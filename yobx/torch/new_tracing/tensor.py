from typing import Any, Dict, Optional, Tuple, Union
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

        Concrete integer dimensions contribute their actual value.  Symbolic
        (string-valued) :class:`TracingInt` dimensions use the trace-time
        concrete value recorded in
        :attr:`~yobx.torch.new_tracing.tracer.GraphTracer._dim_concrete_values`
        when the tensor was created via
        :meth:`~yobx.torch.new_tracing.tracer.GraphTracer.make_tracing_arg`;
        when no tracer or no concrete value is available the symbolic dim is
        folded into a symbolic :class:`TracingInt` product.

        This ensures that guards such as ``if x.numel() == 0:`` evaluate
        correctly at trace time rather than always returning ``True`` due to
        the underlying wrapper tensor storing ``0`` for every symbolic
        dimension.  Any dimension with a concrete value of ``0`` (including
        symbolic dims whose trace-time size was ``0``) causes an immediate
        return of ``0``.

        Returns:
            Union[int, TracingInt]: The product of all dimension values.  A
            plain ``int`` is returned when every dimension is either a concrete
            integer or a symbolic dim with a known concrete value in the tracer.
            A :class:`TracingInt` is returned when at least one symbolic dim
            has no recorded concrete value, so that callers such as
            ``if x.numel() == 0:`` can propagate the symbolic comparison
            correctly via :class:`~yobx.torch.new_tracing.shape.TracingBool`.
        """
        dim_concrete = (
            getattr(self._tracer, "_dim_concrete_values", {}) if self._tracer is not None else {}
        )
        result: Union[int, TracingInt] = 1
        for d in self._tracing_shape.dims:
            if isinstance(d, TracingInt):
                if isinstance(d.value, int):
                    if d.value == 0:
                        return 0
                    result = result * d.value
                else:
                    # Symbolic dim: look up the concrete value recorded by the
                    # tracer when this tensor was created from a concrete input.
                    cv = dim_concrete.get(d.value)
                    if cv is not None:
                        if cv == 0:
                            return 0
                        result = result * cv
                    else:
                        # No concrete value available; keep the dim symbolic.
                        result = d * result
            else:
                d_int = int(d)
                if d_int == 0:
                    return 0
                result = result * d_int
        return result

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
