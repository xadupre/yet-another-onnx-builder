import operator
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.fx
import torch.utils._pytree as _pytree
from .shape import TracingInt, TracingShape


class TracingTensor(torch.Tensor):
    """
    A :class:`torch.Tensor` subclass that records all dispatch-level operations
    into a :class:`torch.fx.Graph` via ``__torch_dispatch__``.

    :class:`TracingTensor` uses ``__torch_dispatch__`` to intercept all tensor
    operations at the C++ dispatcher level and records them as nodes in a
    :class:`torch.fx.Graph`.  This produces a full computation graph without
    requiring Python-level symbolic proxy objects.

    .. note::

        :class:`TracingTensor` instances are created internally by
        :class:`~yobx.torch.new_tracing.DispatchTracer`.  Use
        :meth:`~yobx.torch.new_tracing.DispatchTracer.trace` to trace a
        callable rather than constructing :class:`TracingTensor` directly.

    It contains two attributes:

    * ``_tracer``: The :class:`~yobx.torch.new_tracing.DispatchTracer`
      managing this tensor's graph.
    * ``_node``: The :class:`torch.fx.Node` corresponding to this tensor in
      the graph.
    """

    @staticmethod
    def __new__(  # noqa: PYI034
        cls,
        size: Union[Tuple[int, ...], "TracingShape"],
        dtype: torch.dtype,
        device: Optional[Union[str, torch.device]] = None,
        requires_grad: bool = False,
        tracer: Optional["GraphTracer"] = None,  # type: ignore # noqa: F821
    ) -> "TracingTensor":
        if isinstance(size, TracingShape):
            # Use concrete values where available; fall back to 1 for purely
            # symbolic dimensions so that _make_wrapper_subclass receives ints.
            sizes = tuple(
                (
                    d.value
                    if isinstance(d, TracingInt) and isinstance(d.value, int)
                    else (int(d) if isinstance(d, int) else d)
                )
                for d in size
            )
            sizes = tuple((i if isinstance(i, int) else 0) for i in sizes)
        else:
            assert isinstance(size, tuple), f"Unexpected type {type(size)} for size"
            assert not size or all(
                isinstance(i, int) for i in size
            ), f"Unexpected type in sizes {[type(s) for s in size]} for size"
            sizes = tuple((i if isinstance(i, int) else 0) for i in size)

        t = torch.Tensor._make_wrapper_subclass(
            cls, sizes, dtype=dtype, device=device, requires_grad=requires_grad  # type: ignore
        )
        t._tracer = tracer
        return t

    def __init__(
        self,
        size: Union[Tuple[int, ...], "TracingShape"],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
        requires_grad: bool = False,
        tracer: Optional["GraphTracer"] = None,  # type: ignore # noqa: F821
    ):
        self._tracer = tracer
        self._node: Optional[torch.fx.Node] = None
        assert isinstance(size, TracingShape) or all(
            isinstance(i, int) for i in size
        ), f"Unexpected type for shape={size!r}"
        self._tracing_shape = size if isinstance(size, TracingShape) else TracingShape(size)

    @property
    def shape(self) -> TracingShape:  # type: ignore
        """Returns the shape as a TracingShape."""
        return self._tracing_shape

    def numel(self) -> Union[int, TracingInt]:  # type: ignore[override]
        """Computes the total number of elements from :attr:`_tracing_shape`.

        Concrete integer dimensions contribute their actual value directly.
        Symbolic (string-valued) :class:`TracingInt` dimensions are folded
        into the product as symbolic terms, yielding a :class:`TracingInt`
        return value.

        This ensures that guards of the form ``if x.numel() == 0:`` can be
        resolved at trace time via the :class:`TracingBool` mechanism: models
        should use ``torch._check(x.numel() != 0)`` to register the non-empty
        constraint (as with :class:`~yobx.torch.testing._model_eval_cases.ControlFlowShapeCheck`),
        after which :meth:`~yobx.torch.new_tracing.shape.TracingBool.__bool__`
        resolves the equality to ``False`` via its negation lookup.

        A concrete dimension of ``0`` still causes an immediate return of ``0``
        so that genuinely empty static shapes are identified correctly.

        When a tracer is active and the result is symbolic, FX nodes are
        emitted for the numel computation and stored on the returned
        :class:`TracingInt` (see :meth:`_emit_numel_node`).  Subsequent
        comparisons such as ``numel() > 0`` then also emit comparison FX
        nodes, allowing the result to serve as a ``torch.cond`` predicate.

        Returns:
            Union[int, TracingInt]: Plain ``int`` when every dimension is
            concrete; :class:`TracingInt` when any dimension is symbolic.
        """
        result: Union[int, TracingInt] = 1
        for d in self._tracing_shape.dims:
            if isinstance(d, TracingInt):
                if isinstance(d.value, int):
                    if d.value == 0:
                        return 0
                    result = result * d.value
                else:
                    # Purely symbolic dim — fold into the product symbolically.
                    result = d * result
            else:
                d_int = int(d)
                if d_int == 0:
                    return 0
                result = result * d_int
        # When the result is symbolic and a tracer is active, emit FX nodes
        # so that comparisons such as ``numel() > 0`` can produce a proper
        # bool tensor node (required for use as a ``torch.cond`` predicate).
        if isinstance(result, TracingInt) and self._tracer is not None and self._node is not None:
            numel_node = self._emit_numel_node()
            if numel_node is not None:
                result._node = numel_node
                result._tracer = self._tracer
                result._device = self.device
        return result

    def _emit_numel_node(self) -> Optional[torch.fx.Node]:
        """Emits FX nodes that compute ``numel()`` as a scalar ``int64`` tensor.

        For each dimension of this tensor an ``aten.sym_size.int`` node is
        emitted.  All per-dimension nodes are then multiplied together via
        ``aten.mul.Tensor`` dispatch calls.  The final node represents the
        total element count as a 0-dim ``int64`` :class:`TracingTensor`.

        This follows the same pattern as :meth:`_div_by_tracing_int`: shape
        dimensions are materialised as proper FX nodes so that the resulting
        integer value participates correctly in the runtime computation graph.

        Returns:
            The FX :class:`~torch.fx.Node` whose output is the numel scalar,
            or ``None`` when no tracer is active or the tensor has no
            dimensions.
        """
        import traceback

        tracer = self._tracer
        if tracer is None or self._node is None:
            return None

        dims: List[Any] = list(self._tracing_shape.dims)
        if not dims:
            return None

        device = self.device

        def make_sym_size_node(dim_idx: int) -> torch.fx.Node:
            """Emits ``aten.sym_size.int(self, dim_idx)`` and wraps it."""
            node = tracer.graph.call_function(
                torch.ops.aten.sym_size.int, args=(self._node, dim_idx), kwargs={}
            )
            # Create a 0-dim int64 TracingTensor for FX node metadata.
            tt = TracingTensor(TracingShape(()), dtype=torch.int64, device=device, tracer=tracer)
            tt._node = node
            node.meta["val"] = tt
            node.meta["stack_trace"] = "".join(traceback.format_stack())
            return node

        result_node = make_sym_size_node(0)
        result_tt: "TracingTensor" = result_node.meta["val"]

        for i in range(1, len(dims)):
            dim_node = make_sym_size_node(i)
            dim_tt: "TracingTensor" = dim_node.meta["val"]
            # aten.mul.Tensor dispatches through __torch_dispatch__ → dispatch(),
            # creating a proper FX node and returning a new TracingTensor.
            result_tt = torch.ops.aten.mul.Tensor(result_tt, dim_tt)  # type: ignore
            result_node = result_tt._node  # type: ignore

        return result_node

    def __repr__(self) -> str:  # type: ignore
        node_name = self._node.name if self._node is not None else "<unregistered>"
        return (
            f"TracingTensor(node={node_name!r}, shape={self.shape}, "
            f"dtype={self.dtype}, device={self.device})"
        )

    @classmethod
    def from_tensor(
        cls,
        t: torch.Tensor,
        dynamic_shapes: Optional[Dict[int, Any]] = None,
        tracer: Optional["GraphTracer"] = None,  # type: ignore # noqa: F821
    ) -> "TracingTensor":
        """Creates a tracing tensor."""
        return TracingTensor(
            TracingShape.from_existing_shape(t.shape, dynamic_shapes),
            t.dtype,
            t.device,
            tracer=tracer,
        )

    def make_empty_instance(
        self, dyanmic_shape_values: Optional[Dict[str, int]] = None
    ) -> torch.Tensor:
        """
        Allocates an uninitialised :func:`torch.empty` tensor whose dtype and
        device match this :class:`TracingTensor`.

        Concrete integer dimensions are used as-is.  Symbolic (string) dimensions
        must be resolved by supplying *dyanmic_shape_values*, a mapping from
        dimension name to its concrete integer value.  A missing entry for any
        symbolic dimension raises :exc:`AssertionError`.

        :param dyanmic_shape_values: Optional mapping from symbolic dimension
            names (e.g. ``"batch"``) to their concrete integer sizes.
        :returns: A real :class:`torch.Tensor` with the resolved shape,
            the same ``dtype``, and the same ``device`` as this
            :class:`TracingTensor`.  The tensor is uninitialised (contents are
            undefined).
        :raises AssertionError: If a symbolic dimension name is not present in
            *dyanmic_shape_values*.
        :raises NotImplementedError: If a dimension has an unexpected type (i.e.
            neither :class:`int` nor :class:`str`).
        """
        if not dyanmic_shape_values:
            assert all(
                isinstance(i, int) for i in self.shape
            ), f"One shape is dynamic in {self.shape} but dyanmic_shape_values is empty."
            return torch.empty(tuple(self.shape), dtype=self.dtype, device=self.device)  # type: ignore
        new_shape = []
        for s in self.shape:
            if isinstance(s, int):
                new_shape.append(s)
            elif isinstance(s, TracingInt):
                assert (
                    s.value in dyanmic_shape_values
                ), f"Missing value for dynamic dimension {s!r}"
                new_shape.append(dyanmic_shape_values[s.value])
            else:
                raise NotImplementedError(f"Unexpected type {type(s)}")
        return torch.empty(tuple(new_shape), dtype=self.dtype, device=self.device)

    @classmethod
    def __torch_dispatch__(  # type: ignore
        cls,
        func: Any,
        types: Any,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Intercept every dispatched operation, create an FX graph node, and
        return a new :class:`TracingTensor` (or tuple thereof) for the result.
        """
        # Use pytree to find the tracer from any TracingTensor in the
        # (possibly nested) args.  This handles ops like ``aten.cat.default``
        # where the first argument is a *list* of tensors, not a bare tensor.
        tracer = next(
            (a._tracer for a in _pytree.tree_leaves(args) if isinstance(a, TracingTensor)), None
        )
        assert (
            tracer is not None
        ), f"Missing tracer for func={func}, types={types}, {args}, {kwargs=}"
        return tracer.dispatch(func, args, kwargs)

    def _div_by_tracing_int(self, dim: "TracingInt") -> "TracingTensor":
        """
        Handles division of this tensor by a symbolic :class:`TracingInt`
        (e.g. ``tensor / tensor.shape[0]``).

        Instead of passing the :class:`TracingInt` directly to the ATen
        dispatcher (which would fail because it is not a scalar or tensor),
        this method:

        1. Looks up which dimension index in ``self.shape`` corresponds to
           ``dim.value``.
        2. Emits an ``aten.sym_size.int(self, dim_idx)`` FX node that
           computes the dimension at runtime (translates to ONNX
           ``Shape + Squeeze``).
        3. Wraps that node in a scalar ``int64`` :class:`TracingTensor`.
        4. Delegates to ``aten.div.Tensor(self, scalar_tt)`` via the normal
           dispatch path.

        :param dim: The symbolic :class:`TracingInt` representing the
            dimension to divide by.  It must appear in ``self.shape``.
        :returns: A :class:`TracingTensor` representing ``self / dim``.
        :raises ValueError: If *dim* is not found in ``self.shape``.
        """
        tracer = self._tracer
        assert tracer is not None, "_div_by_tracing_int requires a tracer"

        # Find the dimension index in self.shape that matches dim.value.
        dim_idx: Optional[int] = None
        for i, d in enumerate(self._tracing_shape.dims):
            if isinstance(d, TracingInt) and d.value == dim.value:
                dim_idx = i
                break

        if dim_idx is None:
            raise ValueError(
                f"TracingInt({dim.value!r}) not found in tensor shape "
                f"{self._tracing_shape}; cannot create a shape-access node "
                "for division.  Ensure the divisor comes from the same "
                "tensor's shape."
            )

        # Emit aten.sym_size.int(self, dim_idx) — becomes Shape+Squeeze in ONNX.
        sym_node = tracer.graph.call_function(
            torch.ops.aten.sym_size.int, args=(self._node, dim_idx), kwargs={}
        )
        # Create a scalar int64 TracingTensor backed by the shape-access node.
        shape_dim_tt = TracingTensor(
            TracingShape(()),  # 0-dim (scalar)
            dtype=torch.int64,
            device=self.device,
            tracer=tracer,
        )
        shape_dim_tt._node = sym_node
        # Set meta["val"] so that dispatch() can read the TracingTensor's
        # shape/dtype when this node is used as an argument in subsequent
        # FX nodes (e.g. aten.div.Tensor below).  This mirrors the
        # convention used throughout dispatch() for all produced nodes.
        sym_node.meta["val"] = shape_dim_tt

        # Delegate to aten.div.Tensor(self, shape_dim_tt).  Both operands are
        # now TracingTensors, so this routes through __torch_dispatch__ →
        # dispatch() in the tracer as usual.
        return torch.ops.aten.div.Tensor(self, shape_dim_tt)  # type: ignore

    def __truediv__(self, other: Any) -> Any:
        """
        Overrides true-division to handle symbolic :class:`TracingInt`
        divisors (e.g. ``cat / cat.shape[0]``).

        When *other* is a symbolic :class:`TracingInt`, calling
        ``super().__truediv__`` would fail because the C++ dispatcher cannot
        convert a :class:`TracingInt` to a numeric scalar.  Instead, this
        method calls :meth:`_div_by_tracing_int` to emit the required
        shape-access FX nodes and then performs a tensor-tensor division.

        For all other divisors (plain ``int``, ``float``, or
        :class:`TracingTensor`) the default ``torch.Tensor.__truediv__``
        behaviour is preserved.

        :param other: The divisor.
        :returns: Result of ``self / other`` as a :class:`TracingTensor`.
        """
        if isinstance(other, TracingInt) and not other.is_static:
            return self._div_by_tracing_int(other)
        return super().__truediv__(other)

    def __setitem__(self, indices: Any, values: Any) -> None:
        """
        Captures inplace index-assignment as an ``operator.setitem`` FX node.

        When Python evaluates ``x[...] = val`` on a :class:`TracingTensor`,
        this override intercepts the assignment before it reaches the C++
        dispatcher.  It emits a single ``call_function`` node with target
        ``operator.setitem`` and updates ``self._node`` to point to that node
        so that all subsequent uses of this tensor in the graph flow through
        the setitem result.

        This mirrors the behaviour of
        :meth:`~yobx.torch.tracing.CustomProxy.__setitem__` in the old
        symbolic-tracing path, allowing
        :func:`~yobx.torch.tracing.CustomTracer.remove_inplace` and the
        ONNX interpreter's ``aten_setitem`` handler to process the node in
        the same way as the old tracing.

        :param indices: Index expression (slice, tuple of slices,
            integer, boolean mask, or :class:`TracingTensor`).
        :param values: Value(s) to assign.  May be a scalar, a
            :class:`torch.Tensor`, or a :class:`TracingTensor`.
        """
        tracer = self._tracer
        assert tracer is not None, "__setitem__ requires an active tracer"

        def _unwrap(idx: Any) -> Any:
            """Returns the FX node for a TracingTensor index or the index itself."""
            if isinstance(idx, TracingTensor):
                return idx._node
            if isinstance(idx, tuple):
                return tuple(_unwrap(i) for i in idx)
            return idx

        processed_indices = _unwrap(indices)
        values_arg = values._node if isinstance(values, TracingTensor) else values

        node = tracer.graph.call_function(
            operator.setitem, args=(self._node, processed_indices, values_arg), kwargs={}
        )
        # Store a TracingTensor as node metadata so the FX interpreter can
        # infer the output dtype/shape (same as the modified tensor).
        shape, dtype, device = self._tracing_shape, self.dtype, self.device
        meta_tt = TracingTensor.__new__(TracingTensor, shape, dtype=dtype, device=device)
        meta_tt._tracing_shape = shape
        meta_tt._tracer = tracer
        meta_tt._node = node
        node.meta["val"] = meta_tt
        node.meta["stack_trace"] = "".join(traceback.format_stack())
        # Redirect self._node so that all subsequent operations on this tensor
        # in the same forward() call flow through the setitem node rather than
        # the original placeholder.  This ensures eliminate_dead_code() does
        # not discard the node and that downstream ops receive the updated
        # tensor value.
        self._node = node
