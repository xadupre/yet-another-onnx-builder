import inspect
import operator
import traceback
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch
from ...xexpressions import rename_expression
from .shape import TracingShape, TracingInt, TracingBool, register_condition, clear_conditions
from .tensor import TracingTensor

# Mapping from ATen inplace op overloads to the transformation name string
# recognised by :func:`~yobx.torch.tracing.setitem_with_transformation`.
# Extend this dict to support additional elementwise inplace functions.
_ATEN_INPLACE_TO_TRANSFORM_NAME: Dict[Any, str] = {
    torch.ops.aten.exp_.default: "exp",
    torch.ops.aten.sigmoid_.default: "sigmoid",
}


class GraphTracer:
    """
    Traces a callable by intercepting all tensor operations via
    ``__torch_dispatch__`` and records them into a :class:`torch.fx.Graph`.

    .. runpython::
        :showcode:
        :process:

        import torch
        from yobx.torch.new_tracing.tracer import GraphTracer

        def add(x, y):
            return x + y

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        tracer = GraphTracer()
        graph = tracer.trace(add, (x, y))
        print(graph)

    The class creates an empty :class:`torch.fx.Graph`
    populated by a :meth:`trace` call. We do the same
    with dynamic shapes:

    .. runpython::
        :showcode:
        :process:

        import torch
        from yobx.torch.new_tracing.tracer import GraphTracer

        def add(x, y):
            return x + y

        x = torch.randn(3, 4)
        y = torch.randn(1, 4)
        tracer = GraphTracer()
        graph = tracer.trace(add, (x, y), {}, ({0:"batch"}, {}))
        print(graph)

    **Leaf modules** — sub-modules that should not be traced into but instead
    appear as a single ``call_function`` node in the graph — can be declared
    via the *module_leaves* constructor argument:

    .. runpython::
        :showcode:
        :process:

        import torch
        from yobx.torch.new_tracing.tracer import GraphTracer

        class MyLeaf(torch.nn.Module):
            def forward(self, x):
                return x * 2

        class Outer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = MyLeaf()
            def forward(self, x):
                return self.leaf(x) + 1

        tracer = GraphTracer(
            module_leaves={MyLeaf: lambda m, module_qualified_name=None: True}
        )
        graph = tracer.trace(Outer(), (torch.randn(2, 4),))
        # print(graph fails)
        graph.print_tabular()

    :param verbose: Verbosity level (0 = silent).
    :param module_leaves: Optional mapping from module *type* to a predicate
        ``f(module, module_qualified_name=name) -> bool``.  When the predicate
        returns ``True`` for a given sub-module instance, that sub-module is
        treated as a leaf: a single ``call_function`` node is emitted for its
        call site and the tracer does not descend into its ``forward`` method.
        Internal parameters and buffers of leaf modules are also excluded from
        the graph's placeholder nodes.
    """

    def __init__(
        self, verbose: int = 0, module_leaves: Optional[Dict[type, Callable[..., bool]]] = None
    ) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        self.graph: torch.fx.Graph = torch.fx.Graph()
        self._external_tensor_to_node: Dict[int, torch.fx.Node] = {}
        self._placeholder_count: int = 0
        self.verbose = verbose
        #
        self._shape_env = ShapeEnv()
        self._fake_mode = FakeTensorMode(shape_env=self._shape_env)
        self._mapped_dimension: Dict[str, torch.SymInt] = {}
        self._sym_int_to_dynamic_dimension: Dict[str, str] = {}
        self._unique_count = 0
        # Storage for torch.cond branches: maps unique name -> callable / sub-tracer.
        self._callables: Dict[str, Callable] = {}
        self._sub_tracers: Dict[str, "GraphTracer"] = {}
        # Mapping from module *type* to a predicate
        # ``f(module, module_qualified_name=name) -> bool`` that decides whether
        # a given module instance should be treated as a leaf.  Leaf modules are
        # not traced into; instead a single ``call_function`` node is emitted for
        # the whole module call.
        self.module_leaves: Optional[Dict[type, Callable[..., bool]]] = module_leaves

    def _is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        Return ``True`` if *m* should be treated as a leaf module.

        A module is a leaf when :attr:`module_leaves` maps its ``type`` to a
        predicate ``f(m, module_qualified_name=name) -> bool`` and that predicate
        returns ``True`` for *m*.

        :param m: The :class:`~torch.nn.Module` to test.
        :param module_qualified_name: The dot-separated path from the root
            module to *m* (e.g. ``"encoder.layer.0"``).
        :return: ``True`` when *m* is a leaf, ``False`` otherwise.
        """
        if not self.module_leaves:
            return False
        if type(m) not in self.module_leaves:
            return False
        return bool(self.module_leaves[type(m)](m, module_qualified_name=module_qualified_name))

    def _make_tracing_tensor(
        self,
        shape: Union[Tuple[int, ...], "TracingShape"],
        dtype: torch.dtype,
        device: Union[str, torch.device],
        node: torch.fx.Node,
    ) -> "TracingTensor":
        """
        Allocate a :class:`TracingTensor` and bind it to *node*.

        The tensor is created via :meth:`TracingTensor.__new__` / ``__init__``
        so that no real storage is allocated, then its ``_node`` attribute is
        set to *node* so that subsequent calls to :meth:`_get_node` can
        retrieve the corresponding graph node.

        :param shape: Tensor shape expressed as a plain ``tuple[int, ...]`` or
            a :class:`TracingShape` (which may contain symbolic string
            dimensions).
        :param dtype: Element dtype of the tensor.
        :param device: Target device (string such as ``"cpu"`` or a
            :class:`torch.device` instance).
        :param node: The :class:`torch.fx.Node` that produces this tensor in
            the graph.
        :return: A :class:`TracingTensor` whose ``_node`` is *node* and whose
            ``_tracer`` is ``self``.
        """
        t = TracingTensor.__new__(TracingTensor, shape, dtype=dtype, device=device)
        t.__init__(shape, dtype=dtype, device=device, tracer=self)  # type: ignore
        t._node = node
        return t

    def _get_node(self, t: Union[TracingTensor, torch.Tensor]) -> torch.fx.Node:
        """Return the :class:`torch.fx.Node` associated with *t*."""
        if isinstance(t, TracingTensor):
            assert t._node, f"Tensor {t} has no node."
            assert t._tracer is self, "The tensor seems traced by another graph."
            return t._node
        return self._node_for_external_tensor(t)

    def is_not_tensor(self, value: Any) -> bool:
        """
        Return ``True`` if *value* contains no :class:`torch.Tensor` leaves.

        Scalars (``int``, ``float``, ``str``), ``None``, and empty collections
        are treated as non-tensor.  Lists and tuples are inspected recursively.
        Dicts are inspected by their values.
        :class:`~yobx.torch.new_tracing.shape.TracingInt` instances are
        treated as scalar integers (non-tensor).

        :param value: The value to inspect.  May be a scalar, tensor,
            list, tuple, dict, or ``None``.
        :return: ``True`` when *value* has no tensor leaves; ``False`` when
            any leaf is a :class:`torch.Tensor` (including
            :class:`TracingTensor`).
        :raises TypeError: If *value* has a type that cannot be classified.
        """
        if value is None:
            return True
        if isinstance(value, (int, float, str)):
            return True
        if isinstance(value, torch.Tensor):
            return False
        # TracingInt represents a scalar integer (a shape dimension value);
        # it is not a tensor.
        if isinstance(value, TracingInt):
            return True
        if isinstance(value, (list, tuple)):
            if not value:
                return True
            return all(self.is_not_tensor(v) for v in value)
        if isinstance(value, dict):
            if not value:
                return True
            return all(self.is_not_tensor(v) for v in value.values())
        if isinstance(value, torch.dtype):
            return True
        if isinstance(value, torch.device):
            return True

        from ...helpers import string_type

        raise TypeError(
            f"Cannot determine if type {type(value)} "
            f"is a constant argument {string_type(value)}"
        )

    def _tracing_int_to_fake(self, ti: TracingInt) -> Union[int, "torch.SymInt"]:
        """
        Convert a :class:`~yobx.torch.new_tracing.shape.TracingInt` to a
        value suitable for use in :class:`~torch._subclasses.FakeTensor`
        computations.

        For static (integer-valued) :class:`TracingInt` instances the plain
        ``int`` value is returned.  For symbolic instances the mapped
        :class:`torch.SymInt` is returned if one exists; otherwise ``0`` is
        used as a last-resort fallback.

        :param ti: The :class:`TracingInt` to convert.
        :returns: An ``int`` or :class:`torch.SymInt` suitable for fake
            tensor arithmetic.
        """
        if ti.is_static:
            return int(ti)
        if ti.value in self._mapped_dimension:
            return self._mapped_dimension[ti.value]
        return 0  # Last-resort fallback for unknown symbolic dims.

    def _tracing_int_to_const(self, ti: TracingInt) -> int:
        """
        Convert a :class:`~yobx.torch.new_tracing.shape.TracingInt` to a
        constant integer for embedding in an FX graph node argument.

        For static instances the integer value is returned directly.
        Symbolic instances that have not been intercepted by a higher-level
        override fall back to ``0``.

        .. note::
            This helper is a fallback for cases where a
            :class:`~yobx.torch.new_tracing.shape.TracingInt` reaches
            :meth:`dispatch` without being intercepted by a higher-level
            override (e.g. :meth:`~TracingTensor.__truediv__`).  Callers
            should prefer to intercept :class:`TracingInt`
            divisors/multipliers *before* dispatching to produce correct
            dynamic ONNX graphs.

        :param ti: The :class:`TracingInt` to convert.
        :returns: A concrete ``int`` constant.
        """
        if ti.is_static:
            return int(ti)
        return 0  # Last-resort fallback.

    def _node_for_external_tensor(self, t: torch.Tensor) -> torch.fx.Node:
        """
        Return (or lazily create) a placeholder node for a non-traced tensor
        (e.g. a module parameter or bias encountered during dispatch).

        If the tensor was pre-registered by :meth:`_register_module_parameters`
        its named placeholder is returned; otherwise a generic ``param_N``
        placeholder is created on the fly.
        """
        key = id(t)
        if key not in self._external_tensor_to_node:
            self._placeholder_count += 1
            name = f"param_{self._placeholder_count}"
            node = self.graph.placeholder(name)
            node.meta["val"] = TracingTensor.from_tensor(t)
            node.meta["torch_value"] = t
            self._external_tensor_to_node[key] = node
        return self._external_tensor_to_node[key]

    def register_module_parameters(self, module: torch.nn.Module) -> None:
        """
        Pre-register all named parameters and buffers of *module* as
        placeholder nodes in the graph.

        This gives each parameter a meaningful name in the graph (e.g.
        ``linear_weight`` instead of ``param_1``) and ensures that shared
        tensors (the same :class:`torch.Tensor` referenced under multiple
        names) map to exactly one placeholder node.

        Parameters that belong to leaf sub-modules (see :attr:`module_leaves`)
        are **skipped**: leaf modules are treated as black boxes and their
        internal parameters are not exposed as graph inputs.

        Each placeholder node receives two extra metadata entries:

        * ``node.meta["torch_name"]``: the original dotted parameter name as
          returned by :meth:`torch.nn.Module.named_parameters` (e.g.
          ``"linear.weight"``).
        * ``node.meta["torch_value"]``: the actual :class:`torch.Tensor`
          object (useful for retrieving concrete weight values later).

        :param module: The :class:`torch.nn.Module` whose parameters and
            buffers should be pre-registered.
        """
        # Collect dot-separated paths of all leaf sub-modules.  Parameters
        # whose module path has a leaf ancestor are internal to that leaf and
        # should not be registered as graph placeholders.
        _leaf_module_paths: Set[str] = set()
        if self.module_leaves:
            for subname, submod in module.named_modules():
                if subname and self._is_leaf_module(submod, subname):
                    _leaf_module_paths.add(subname)

        seen_ids: Set[int] = set()
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            if tensor is None:
                continue
            # Skip parameters that belong to a leaf sub-module.
            if _leaf_module_paths:
                parts = name.split(".")
                # Check every module-path prefix (excluding the bare param name).
                if any(".".join(parts[:i]) in _leaf_module_paths for i in range(1, len(parts))):
                    if self.verbose:
                        print(
                            f"[GraphTracer.register_module_parameters] skip {name!r}"
                            " (belongs to a leaf sub-module)"
                        )
                    continue
            key = id(tensor)
            if key in seen_ids or key in self._external_tensor_to_node:
                continue
            seen_ids.add(key)
            # Replace "." with "_" so the FX node name is a valid identifier.
            sanitized = name.replace(".", "_")
            node = self.graph.placeholder(sanitized)
            node.meta["val"] = TracingTensor.from_tensor(tensor)
            node.meta["torch_name"] = name
            node.meta["torch_value"] = tensor
            if self.verbose:
                print(f"[GraphTracer.register_module_parameters] + {name!r}")
            self._external_tensor_to_node[key] = node

    def _collect_and_replace_module_tensor_attrs(
        self, module: torch.nn.Module
    ) -> List[Tuple[torch.nn.Module, str, torch.Tensor]]:
        """
        Scans every (sub-)module for plain :class:`torch.Tensor` attributes
        that are **not** registered parameters or buffers, registers each as a
        placeholder node, and temporarily replaces the attribute with a
        :class:`TracingTensor`.

        This enables operations such as ``self.params.clone()`` — where
        ``params`` is assigned as a plain tensor in ``__init__`` rather than
        via :meth:`torch.nn.Module.register_buffer` or wrapped in
        :class:`torch.nn.Parameter` — to pass through ``__torch_dispatch__``
        during tracing so they produce proper FX graph nodes.

        Each replaced attribute is recorded in the returned list so that
        :meth:`trace` can restore the originals after the forward pass
        completes.

        :param module: The root :class:`torch.nn.Module` to scan.
        :returns: A list of ``(submodule, attr_name, original_tensor)``
            triples that must be restored after tracing.
        """
        replaced_attrs: List[Tuple[torch.nn.Module, str, torch.Tensor]] = []
        for subname, submod in module.named_modules():
            # Build the set of names already handled by named_parameters /
            # named_buffers so we do not double-register them.
            _registered: Set[str] = (
                set(submod._parameters.keys())
                | set(submod._buffers.keys())
                | set(submod._modules.keys())
            )
            for attr_name, attr_value in list(vars(submod).items()):
                if attr_name in _registered:
                    continue
                if not isinstance(attr_value, torch.Tensor):
                    continue
                if isinstance(attr_value, torch.nn.Parameter):
                    continue
                key = id(attr_value)
                if key in self._external_tensor_to_node:
                    # Already has a placeholder — create a TracingTensor for
                    # the existing node and replace the attribute.
                    node = self._external_tensor_to_node[key]
                    tt = self._make_tracing_tensor(
                        TracingShape(tuple(attr_value.shape)),
                        attr_value.dtype,
                        attr_value.device,
                        node,
                    )
                    object.__setattr__(submod, attr_name, tt)
                    replaced_attrs.append((submod, attr_name, attr_value))
                    continue
                # Register a new placeholder for this plain tensor attribute.
                fq_name = f"{subname}.{attr_name}" if subname else attr_name
                sanitized = fq_name.replace(".", "_")
                node = self.graph.placeholder(sanitized)
                tt = self._make_tracing_tensor(
                    TracingShape(tuple(attr_value.shape)),
                    attr_value.dtype,
                    attr_value.device,
                    node,
                )
                node.meta["val"] = tt
                node.meta["torch_name"] = fq_name
                node.meta["torch_value"] = attr_value
                self._external_tensor_to_node[key] = node
                if self.verbose:
                    print(
                        f"[GraphTracer._collect_and_replace_module_tensor_attrs]"
                        f" + {fq_name!r}"
                    )
                object.__setattr__(submod, attr_name, tt)
                replaced_attrs.append((submod, attr_name, attr_value))
        return replaced_attrs

    def place(self, tt: TracingTensor, name: Optional[str] = None) -> TracingTensor:
        """
        Ensure *tt* is registered in this tracer's graph as a placeholder.

        If *tt* already has a ``_node`` (i.e. it was produced by a previous
        call to :meth:`placeholder` or :meth:`dispatch`), it is returned
        unchanged.  Otherwise a new placeholder node is created, *tt* is
        bound to it, and the updated tensor is returned.

        :param tt: A :class:`TracingTensor` to register.  It must not be
            owned by a *different* :class:`GraphTracer` instance.
        :param name: Unused placeholder for a future name override.  The
            actual node name is always generated as ``tt_<counter>``.
        :return: *tt* with ``_tracer`` set to ``self`` and ``_node`` set to
            the newly created (or pre-existing) placeholder node.
        :raises AssertionError: If *tt* already belongs to a different tracer.
        """
        assert (
            not tt._tracer or tt._tracer is self
        ), "A TracingTensor is already traced by another tracer."
        if tt._node:
            return tt
        name = f"tt_{self._unique_count}"
        self._unique_count += 1
        tt._tracer = self
        node = self.graph.placeholder(name)
        node.meta["val"] = tt
        tt._node = node
        return tt

    def placeholder(
        self,
        name: str,
        shape: Union[Tuple[int, ...], "TracingShape"],
        dtype: torch.dtype,
        device: Union[str, torch.device],
    ) -> "TracingTensor":
        """
        Add a placeholder (input) node to the graph and return the
        corresponding :class:`TracingTensor`.

        :param name: The name of the placeholder in the graph.
        :param shape: Tensor shape (concrete sizes or :class:`TracingShape`).
        :param dtype: Tensor dtype.
        :param device: Target device (string or :class:`torch.device`).
        :return: A :class:`TracingTensor` representing the graph input.
        """
        node = self.graph.placeholder(name)
        # Store a meta tensor so downstream shape inference can use it.
        assert isinstance(shape, TracingShape), f"Unexpected type {type(shape)} for this operator"
        tt = self._make_tracing_tensor(shape, dtype, device, node)
        node.meta["val"] = tt
        # Annotate symbolic dims with source info so they can be lazily
        # materialised as aten.sym_size.int FX nodes when used as slice
        # endpoints (e.g. t[:, :y.shape[1]]).  Also eagerly populate
        # _mapped_dimension so _tracing_int_to_fake works even if the
        # placeholder tensor is never directly passed to a dispatched op.
        device_str = str(device)
        for i, d in enumerate(tt._tracing_shape.dims):
            if isinstance(d, TracingInt) and not d.is_static:
                d._source_node = node
                d._source_dim = i
                d._tracer = self
                d._device = device_str
                if d.value not in self._mapped_dimension:
                    symd = self._shape_env.create_unbacked_symint()
                    self._mapped_dimension[d.value] = symd  # type: ignore
                    symd_name = self._sym_int_to_str(symd)
                    if isinstance(symd_name, str):
                        self._sym_int_to_dynamic_dimension[symd_name] = d.value  # type: ignore
        return tt

    def make_fake(self, a: Union[TracingTensor, torch.Tensor]) -> "FakeTensor":  # type: ignore # noqa: F821
        """
        Convert *a* into a :class:`~torch._subclasses.fake_tensor.FakeTensor`
        for shape/dtype inference inside :meth:`dispatch`.

        For a :class:`TracingTensor`, each symbolic (string) dimension is
        mapped to a :class:`~torch.SymInt` backed by the tracer's
        :class:`~torch.fx.experimental.symbolic_shapes.ShapeEnv`; previously
        seen dimension names are reused so that the same symbol appears
        wherever the same dynamic dimension is referenced.

        For a plain :class:`torch.Tensor`, a fake tensor with the same shape,
        dtype, and device is created directly.

        :param a: Either a :class:`TracingTensor` (possibly with symbolic
            dimensions) or a plain :class:`torch.Tensor`.
        :return: A :class:`~torch._subclasses.fake_tensor.FakeTensor` suitable
            for passing to ATen operations in ``FakeTensorMode``.
        """
        if isinstance(a, TracingTensor):
            new_shape: List[Union[int, torch.SymInt]] = []
            for d in a.shape:
                if isinstance(d, TracingInt):
                    if d.is_static:
                        new_shape.append(d.value)  # type: ignore
                        continue
                    if d.value in self._mapped_dimension:
                        symd = self._mapped_dimension[d.value]
                    else:
                        symd = self._shape_env.create_unbacked_symint()
                        self._mapped_dimension[d.value] = symd  # type: ignore
                        symd_name = self._sym_int_to_str(symd)
                        assert isinstance(symd_name, str), "type checking"
                        self._sym_int_to_dynamic_dimension[symd_name] = d.value  # type: ignore
                    new_shape.append(symd)
                    continue
                assert isinstance(d, int), f"Unexpected type for a dimension {type(d)}"
                new_shape.append(d)
            with self._fake_mode:
                return torch.empty(tuple(new_shape), dtype=a.dtype, device=a.device)
        with self._fake_mode:
            return torch.empty(a.shape, dtype=a.dtype, device=a.device)

    def _handle_select_int(self, x: TracingTensor, dim: int, index: int) -> TracingTensor:
        """
        Handle ``aten.select.int(x, dim, index)`` without calling
        :meth:`dispatch` or :class:`~torch._subclasses.fake_tensor.FakeTensorMode`.

        When *x* has a symbolic (dynamic) dimension at *dim*, the fake-mode
        meta kernel for ``aten.select.int`` would try to prove the bounds check
        ``-size <= index < size`` for an unbacked :class:`~torch.SymInt`.  That
        guard cannot be statically discharged, raising
        :exc:`torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode`.

        This method bypasses the issue entirely by computing the output shape
        directly from :attr:`~TracingTensor._tracing_shape` (dropping dimension
        *dim*) and emitting the ``aten.select.int`` FX node directly.

        It is called from :meth:`~yobx.torch.new_tracing.tensor.TracingTensor.__getitem__`
        whenever a :class:`TracingTensor` is indexed with a single integer.

        :param x: The input :class:`TracingTensor`.
        :param dim: The dimension to select along.  Negative values are
            normalised internally to their equivalent non-negative index.
        :param index: The integer index to select.
        :returns: A :class:`TracingTensor` with the selected dimension dropped.
        """
        ndim = len(x.shape)
        if dim < 0:
            dim = dim + ndim
        out_shape = TracingShape(tuple(d for i, d in enumerate(x.shape.dims) if i != dim))
        node = self.graph.call_function(
            torch.ops.aten.select.int, args=(self._get_node(x), dim, index), kwargs={}
        )
        result = self._make_tracing_tensor(out_shape, x.dtype, x.device, node)
        node.meta["val"] = result
        node.meta["stack_trace"] = "".join(traceback.format_stack())
        return result

    def _emit_sym_size_node(self, ti: TracingInt) -> torch.fx.Node:
        """
        Emit (or return a cached) ``aten.sym_size.int`` FX node for *ti*.

        If *ti* already has a ``_node`` cached, that is returned immediately.
        Otherwise a new ``aten.sym_size.int(source_node, source_dim)`` node is
        created in the graph, wrapped in a 0-dim ``int64``
        :class:`~yobx.torch.new_tracing.tensor.TracingTensor` for metadata, and
        the result is cached on ``ti._node`` before being returned.

        :param ti: A symbolic :class:`TracingInt` whose ``_source_node`` and
            ``_source_dim`` are set (populated by :meth:`placeholder`).
        :returns: The FX :class:`~torch.fx.Node` that computes the dimension
            value as a scalar ``int64`` tensor.
        :raises AssertionError: If *ti* has no source node / dim information.
        """
        if ti._node is not None:
            return ti._node
        assert ti._source_node is not None and ti._source_dim is not None, (
            f"TracingInt({ti.value!r}) has no _source_node/_source_dim; "
            "cannot emit aten.sym_size.int node.  Ensure the TracingInt was "
            "created by GraphTracer.placeholder()."
        )
        sym_node = self.graph.call_function(
            torch.ops.aten.sym_size.int, args=(ti._source_node, ti._source_dim), kwargs={}
        )
        device = ti._device or "cpu"
        tt_meta = TracingTensor(TracingShape(()), dtype=torch.int64, device=device, tracer=self)
        tt_meta._node = sym_node
        sym_node.meta["val"] = tt_meta
        sym_node.meta["stack_trace"] = "".join(traceback.format_stack())
        ti._node = sym_node
        return sym_node

    def _tracing_int_to_slice_arg(self, x: Any) -> Any:
        """
        Convert a slice start/stop/step value to an FX graph argument.

        * ``None`` is returned as-is.
        * Plain ``int`` values are returned as-is.
        * Static :class:`TracingInt` values are returned as their ``int``
          equivalent.
        * Symbolic :class:`TracingInt` values trigger emission of an
          ``aten.sym_size.int`` FX node (via :meth:`_emit_sym_size_node`) and
          that node is returned.

        :param x: A slice member value.
        :returns: ``None``, ``int``, or a :class:`~torch.fx.Node`.
        """
        if x is None:
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, TracingInt):
            if x.is_static:
                return int(x)
            return self._emit_sym_size_node(x)
        return x

    def _compute_slice_output_dim(self, input_dim: Any, start: Any, stop: Any) -> Any:
        """
        Compute the output dimension size for ``input[start:stop]`` (step=1).

        The computation is kept symbolic when *stop* is a symbolic
        :class:`TracingInt` and avoids calling ``FakeTensorMode``, which would
        raise :exc:`~torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode`
        for unbacked symbols.

        The returned value is either a plain ``int`` (fully static slice) or a
        :class:`TracingInt` (when *stop* or *input_dim* is symbolic).

        :param input_dim: The current size of the dimension being sliced
            (:class:`TracingInt` or ``int``).
        :param start: The slice start (``None``, ``int``, or
            :class:`TracingInt`).
        :param stop: The slice stop (``None``, ``int``, or
            :class:`TracingInt`).
        :returns: Output dimension size.
        """
        if isinstance(stop, TracingInt) and not stop.is_static:
            # Symbolic stop — assume stop <= input_dim (caller's responsibility).
            start_is_zero = (
                start is None
                or (isinstance(start, int) and start == 0)
                or (isinstance(start, TracingInt) and start.is_static and int(start) == 0)
            )
            if start_is_zero:
                return stop
            # Non-zero start: output size = stop - start
            if isinstance(start, int):
                return TracingInt(f"({stop.value}-{start})")
            # Both symbolic — return stop as a conservative upper bound.
            return stop
        if stop is None:
            # Full slice: output matches input dimension unchanged.
            return input_dim
        # Fully static stop value.
        stop_v: int = int(stop) if isinstance(stop, TracingInt) else stop
        start_v: int = 0
        if start is not None and start != 0:
            start_v = int(start) if isinstance(start, TracingInt) else start
        if isinstance(input_dim, int):
            return max(0, min(input_dim, stop_v) - start_v)
        # Dynamic input_dim with static stop — output size is min(input_dim, stop_v) - start_v.
        # Return the static slice length as a plain int; the ONNX Slice op computes
        # the correct clamped size at runtime.
        return stop_v - start_v

    def _handle_symbolic_getitem(self, tensor: TracingTensor, index: Any) -> TracingTensor:
        """
        Handle ``tensor[index]`` when *index* contains symbolic
        :class:`~yobx.torch.new_tracing.shape.TracingInt` values inside
        :class:`slice` objects.

        This method is called from
        :meth:`~yobx.torch.new_tracing.tensor.TracingTensor.__getitem__`
        instead of ``super().__getitem__`` to avoid the C++ dispatcher
        calling ``__index__()`` on symbolic :class:`TracingInt` values,
        which would raise :exc:`ValueError`.

        Each symbolic :class:`TracingInt` in a slice start/stop is replaced by
        an ``aten.sym_size.int`` FX node emitted via
        :meth:`_emit_sym_size_node`.  Output shapes are computed directly
        from :attr:`~TracingTensor._tracing_shape` without
        :class:`~torch._subclasses.fake_tensor.FakeTensorMode`.

        :param tensor: The input :class:`TracingTensor`.
        :param index: The index expression (single slice or tuple of slices /
            :data:`Ellipsis`).
        :returns: A :class:`TracingTensor` representing the sliced result.
        """
        if not isinstance(index, tuple):
            index = (index,)

        ndim = len(tensor._tracing_shape)
        result: TracingTensor = tensor

        # Determine how many non-Ellipsis elements there are so we can
        # figure out the actual dimension each index element covers.
        n_ellipsis = sum(1 for k in index if k is Ellipsis)
        n_non_ellipsis = len(index) - n_ellipsis
        # Number of dims each Ellipsis absorbs.
        ellipsis_dims = ndim - n_non_ellipsis if n_ellipsis else 0

        actual_dim = 0
        for key in index:
            if key is Ellipsis:
                actual_dim += ellipsis_dims
                continue
            if isinstance(key, int):
                # Integer indexing selects a single element and removes the dim.
                out_shape = TracingShape(
                    tuple(d for i, d in enumerate(result._tracing_shape.dims) if i != actual_dim)
                )
                sel_node = self.graph.call_function(
                    torch.ops.aten.select.int, args=(result._node, actual_dim, key), kwargs={}
                )
                result = self._make_tracing_tensor(
                    out_shape, tensor.dtype, tensor.device, sel_node
                )
                sel_node.meta["val"] = result
                sel_node.meta["stack_trace"] = "".join(traceback.format_stack())
                # actual_dim stays the same: the next key operates on what is
                # now dimension actual_dim after the removed axis.
                continue
            if not isinstance(key, slice):
                # Other index types (tensor indices, etc.) are not handled here.
                raise NotImplementedError(
                    f"_handle_symbolic_getitem does not support index type "
                    f"{type(key)!r} within a tuple that also contains symbolic "
                    "TracingInt slice endpoints."
                )

            start = key.start
            stop = key.stop
            step = key.step

            # Identity slice — no node needed.
            if stop is None and (start is None or start == 0) and (step is None or step == 1):
                actual_dim += 1
                continue

            start_arg = self._tracing_int_to_slice_arg(start)
            stop_arg = self._tracing_int_to_slice_arg(stop)
            step_val: int = (
                1 if step is None else (int(step) if isinstance(step, TracingInt) else step)
            )

            input_dim = result._tracing_shape.dims[actual_dim]
            output_dim = self._compute_slice_output_dim(input_dim, start, stop)

            new_dims = list(result._tracing_shape.dims)
            new_dims[actual_dim] = output_dim
            new_shape = TracingShape(tuple(new_dims))

            slice_node = self.graph.call_function(
                torch.ops.aten.slice.Tensor,
                args=(result._node, actual_dim, start_arg, stop_arg, step_val),
                kwargs={},
            )
            result = self._make_tracing_tensor(new_shape, tensor.dtype, tensor.device, slice_node)
            slice_node.meta["val"] = result
            slice_node.meta["stack_trace"] = "".join(traceback.format_stack())

            actual_dim += 1

        return result

    def _sym_shape_to_str_shape(
        self, sym_shape: Tuple[Union[int, torch.SymInt], ...]
    ) -> TracingShape:
        """
        Convert a shape tuple whose elements may be :class:`torch.SymInt` into
        a shape tuple of plain ``int`` or ``str`` (symbolic dimension names).

        Each :class:`torch.SymInt` element is first stringified via
        :meth:`_sym_int_to_str`, then canonicalised via :meth:`_token_replace`
        (which substitutes known ``SymInt``-to-name mappings).  The resulting
        string is also registered in ``self._mapped_dimension`` so future
        references to the same symbol resolve to the same name.

        :param sym_shape: An iterable of dimension values that may be plain
            ``int``, ``str``, or :class:`torch.SymInt`.
        :return: A ``tuple`` of ``int`` / ``str`` suitable for constructing a
            :class:`TracingShape`.
        """
        new_shape: List[Union[int, TracingInt]] = []
        for s in sym_shape:
            if isinstance(s, (int, TracingInt)):
                new_shape.append(s)
                continue
            assert not isinstance(s, str), f"unexpected type {type(s)}"
            ss = self._sym_int_to_str(s)
            assert isinstance(ss, str), f"unexpected type {type(ss)}"
            ns = self._token_replace(ss)
            assert isinstance(
                ns, str
            ), f"unexpected type {type(ns)}, ns={ns!r}, s={s!r}, ss={ss!r}"
            self._mapped_dimension[ns] = s
            new_shape.append(TracingInt(ns))
        return TracingShape(tuple(new_shape))

    def _sym_int_to_str(self, value):
        """
        Stringify a symbolic integer into a plain ``str`` or ``int``.

        The conversion follows this priority:
        1. A plain ``str`` is returned as-is.
        2. A :class:`torch.SymInt` whose ``.node`` attribute is a ``str`` is
           returned as that string.
        3. A :class:`torch.SymInt` backed by a :class:`torch.fx.experimental.sym_node.SymNode`
           is serialised via its internal ``_expr`` (spaces stripped).
        4. Anything else is cast to ``int`` with :func:`int`.

        :param value: A ``str``, ``int``, or :class:`torch.SymInt`.
        :return: A ``str`` (symbolic name / expression) or ``int`` (concrete
            value).
        """
        if isinstance(value, str):
            return value
        if hasattr(value, "node") and isinstance(value.node, str):
            return f"{value.node}"
        from torch.fx.experimental.sym_node import SymNode

        if hasattr(value, "node") and isinstance(value.node, SymNode):
            # '_expr' is safer than expr
            return str(value.node._expr).replace(" ", "")
        val_int = int(value)
        return val_int

    def _token_replace(self, expr: Union[str, int]) -> Union[str, int]:
        """
        Replace symbolic expression tokens with their user-visible dimension
        names.

        If *expr* is an ``int``, it is returned unchanged.  Otherwise the
        method first checks ``self._sym_int_to_dynamic_dimension`` for an
        exact match; failing that it delegates to
        :func:`~yobx.xexpressions.rename_expression` to perform a token-level
        substitution across the whole expression string.

        :param expr: A symbolic expression string produced by
            :meth:`_sym_int_to_str`, or a concrete ``int``.
        :return: The renamed expression (``str``) or the original ``int``.
        """
        if isinstance(expr, int):
            return expr
        if expr in self._sym_int_to_dynamic_dimension:
            return self._sym_int_to_dynamic_dimension[expr]
        return rename_expression(expr, self._sym_int_to_dynamic_dimension)

    def dispatch(self, func: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Handle one dispatched operation:

        1. Run the op on *meta* tensors for shape/dtype inference.
        2. Build FX node args (replacing :class:`TracingTensor` with their
           nodes, and non-traced tensors with auto-placeholder nodes).
        3. Emit a ``call_function`` node in the graph.
        4. Return wrapped :class:`TracingTensor` output(s).
        """
        if self.verbose > 1:
            print(f"[GraphTracer.dispatch] call {func}")
        kwargs = kwargs or {}

        combined_args, treespec = torch.utils._pytree.tree_flatten((args, kwargs))

        # We could recompute the dynamic shapes without FakeTensor
        # but this is the first approach
        fake_combined = [
            (
                self._tracing_int_to_fake(a)
                if isinstance(a, TracingInt)
                else (a if self.is_not_tensor(a) else self.make_fake(a))
            )
            for a in combined_args
        ]
        unflat = torch.utils._pytree.tree_unflatten(fake_combined, treespec)
        fake_args, fake_kwargs = unflat

        if self.verbose > 2:
            from ...helpers import string_type

            print(f"[GraphTracer.dispatch] >>>> args={string_type(args, with_shape=True)}")
            print(f"[GraphTracer.dispatch] fake_args={string_type(fake_args, with_shape=True)}")
            print(f"[GraphTracer.dispatch] >>>> kwargs={string_type(kwargs, with_shape=True)}")
            print(
                f"[GraphTracer.dispatch] fake_kwargs={string_type(fake_kwargs, with_shape=True)}"
            )

        # Early-exit for inplace ops on slice views.  Running the FakeTensor
        # computation for ``exp_`` on a view of a dynamically-shaped tensor
        # triggers GuardOnDataDependentSymNode (PyTorch tries to evaluate a
        # symbolic shape equality such as ``batch-4 == 1`` which cannot be
        # resolved statically).  Detects this case up-front, emits the
        # ``setitem_with_transformation`` FX node immediately, and returns the
        # view TracingTensor, skipping the fake computation entirely.
        _transform_name = _ATEN_INPLACE_TO_TRANSFORM_NAME.get(func)
        if (
            _transform_name is not None
            and args
            and isinstance(args[0], TracingTensor)
            and hasattr(func, "_schema")
            and func._schema.is_mutable
            and func._schema.arguments
            and func._schema.arguments[0].alias_info is not None
            and func._schema.arguments[0].alias_info.is_write
        ):
            view_tt = args[0]
            _view_source = getattr(view_tt, "_view_source", None)
            _view_indices = getattr(view_tt, "_view_indices", None)
            if _view_source is not None and _view_indices is not None:
                from ..tracing import setitem_with_transformation

                swt_node = self.graph.call_function(
                    setitem_with_transformation,
                    args=(_view_source._node, _view_indices, ((_transform_name, ()),)),
                    kwargs={},
                )
                meta_tt = self._make_tracing_tensor(
                    _view_source._tracing_shape, _view_source.dtype, _view_source.device, swt_node
                )
                swt_node.meta["val"] = meta_tt
                swt_node.meta["stack_trace"] = "".join(traceback.format_stack())
                _view_source._node = swt_node
                if self.verbose > 1:
                    print(
                        f"[GraphTracer.dispatch] view inplace {func!r}: "
                        f"emitted setitem_with_transformation node "
                        f"{swt_node.name!r} for source {_view_source!r}"
                    )
                return view_tt

        # running the function
        # Temporarily restore the torch.full patch to its original so that
        # FakeTensor kernel implementations (e.g. _like constructors inside
        # ``aten.batch_norm``) call the real ``torch.full`` and do not route
        # through our tracing handler, which would create TracingTensors
        # incompatible with FakeTensor.
        from ._patches import _ORIGINAL_TORCH_FULL

        _tracing_full = torch.full
        torch.full = _ORIGINAL_TORCH_FULL
        try:
            with self._fake_mode:
                fake_res = func(*fake_args, **fake_kwargs)
        finally:
            torch.full = _tracing_full
        assert type(fake_res) is not torch.Tensor, f"Unexpected type {type(fake_res)} for output."

        # add a node in the graph
        node_combined = [
            (
                self._tracing_int_to_const(a)
                if isinstance(a, TracingInt)
                else (a if self.is_not_tensor(a) else self._get_node(a))
            )
            for a in combined_args
        ]
        unflat_nodes = torch.utils._pytree.tree_unflatten(node_combined, treespec)
        node_args, node_kwargs = unflat_nodes
        # We need to add nodes.
        node = self.graph.call_function(func, args=node_args, kwargs=node_kwargs)

        # For inplace operations (e.g. aten.add_.Tensor), the first tensor
        # argument is modified in place.  Python's __iadd__ and friends return
        # ``self``, so the caller keeps holding the *original* TracingTensor
        # whose ``_node`` still points to the placeholder.  Redirecting
        # that ``_node`` to the new graph node ensures that subsequent uses of
        # the tensor — including the function's return value — see the result
        # of the inplace op rather than the original input.
        if (
            args
            and isinstance(args[0], TracingTensor)
            and hasattr(func, "_schema")
            and func._schema.is_mutable
            and func._schema.arguments
            and func._schema.arguments[0].alias_info is not None
            and func._schema.arguments[0].alias_info.is_write
        ):
            args[0]._node = node
            if self.verbose > 1:
                print(
                    f"[GraphTracer.dispatch] inplace op {func!r}: "
                    f"redirected {args[0]!r} _node -> {node.name!r}"
                )

        flat_fake_res, treespec_res = torch.utils._pytree.tree_flatten(fake_res)
        flat_res = [
            (
                a
                if self.is_not_tensor(a)
                else self._make_tracing_tensor(
                    self._sym_shape_to_str_shape(a.shape), a.dtype, a.device, node  # type: ignore
                )
            )
            for a in flat_fake_res
        ]
        unflat_res = torch.utils._pytree.tree_unflatten(flat_res, treespec_res)

        if self.verbose > 2:
            from ...helpers import string_type

            print(f"[GraphTracer.dispatch] fake_res={string_type(fake_res, with_shape=True)}")
            print(f"[GraphTracer.dispatch] <<<< res={string_type(unflat_res, with_shape=True)}")

        if node:
            node.meta["val"] = unflat_res
            node.meta["fake_val"] = fake_res
            node.meta["stack_trace"] = "".join(traceback.format_stack())
            _calling_module = None
            for _frame_info in inspect.stack():
                _frame_self = _frame_info.frame.f_locals.get("self")
                if isinstance(_frame_self, torch.nn.Module):
                    _calling_module = _frame_self
                    break
            node.meta["fn"] = _calling_module

        assert (
            func not in {torch.ops.aten.split.Tensor} or unflat_res is not None
        ), f"res is None but func is {func}, this is not possible, args={args}, kwargs={kwargs}"

        if isinstance(unflat_res, (TracingTensor, int, float, str)):
            return unflat_res

        if isinstance(unflat_res, (list, tuple)):
            results: List[Any] = []
            for i, item in enumerate(unflat_res):
                if isinstance(item, TracingTensor):
                    get_node = self.graph.call_function(
                        operator.getitem, args=(node, i), kwargs={}
                    )
                    get_node.meta["val"] = item
                    if "stack_trace" in node.meta:
                        get_node.meta["stack_trace"] = node.meta["stack_trace"]
                    results.append(
                        self._make_tracing_tensor(item.shape, item.dtype, item.device, get_node)
                    )
                else:
                    assert isinstance(
                        item, (float, int, str)
                    ), f"Unexpected type for one item={item}"
                    results.append(item)
            return type(unflat_res)(results)

        raise NotImplementedError(f"Unexpected type {type(unflat_res)}")

    # ------------------------------------------------------------------
    # Leaf module support
    # ------------------------------------------------------------------

    def _make_leaf_forward(self, module: torch.nn.Module, qualified_name: str) -> Callable:
        """
        Return a replacement ``forward`` for *module* that emits a single
        ``call_function`` graph node instead of tracing through the module's
        internals.

        When the leaf-forward wrapper is called during tracing the input
        arguments are :class:`TracingTensor` instances.  The wrapper:

        1. Creates fresh :class:`~torch._subclasses.fake_tensor.FakeTensor`
           inputs inside a new :class:`~torch._subclasses.fake_tensor.FakeTensorMode`
           (sharing the tracer's :class:`~torch.fx.experimental.symbolic_shapes.ShapeEnv`
           so that symbolic dimensions are consistent).
        2. Runs the **original** ``forward`` in that mode with
           ``allow_non_fake_inputs=True`` so the module's own real parameters
           are accepted alongside the fake inputs.
        3. Records a single ``call_function(module, node_args, node_kwargs)``
           FX node and returns the corresponding :class:`TracingTensor`
           output(s).

        :param module: The leaf :class:`~torch.nn.Module` whose forward should
            be wrapped.
        :param qualified_name: Dot-separated path to *module* inside the root
            module being traced (e.g. ``"encoder.layer"``) — used for error
            messages only.
        :return: A callable that accepts the same positional/keyword arguments
            as ``module.forward`` but emits a single graph node.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode

        original_forward = module.forward
        tracer = self

        def _leaf_forward(*args: Any, **kwargs: Any) -> Any:
            # -- shape inference -----------------------------------------------
            # Build a fresh FakeTensorMode that shares the tracer's ShapeEnv
            # so symbolic dimensions propagate correctly.  allow_non_fake_inputs
            # is required because the module's own parameters are real tensors.
            _infer_mode = FakeTensorMode(shape_env=tracer._shape_env, allow_non_fake_inputs=True)
            with _infer_mode:
                # Create fake tensors *inside* the mode so they are owned by
                # _infer_mode (avoids "fake-tensor mode mismatch" errors).
                def _to_fake_in_mode(a: Any) -> Any:
                    if isinstance(a, TracingTensor):
                        new_shape: List[Any] = []
                        for d in a.shape:
                            if isinstance(d, str):
                                if d in tracer._mapped_dimension:
                                    new_shape.append(tracer._mapped_dimension[d])
                                else:
                                    new_shape.append(tracer._shape_env.create_unbacked_symint())
                            else:
                                assert isinstance(d, int), (
                                    f"Unexpected dimension type {type(d)} in leaf "
                                    f"module {qualified_name!r}"
                                )
                                new_shape.append(d)
                        return torch.empty(tuple(new_shape), dtype=a.dtype, device=a.device)
                    if isinstance(a, torch.Tensor):
                        return torch.empty(a.shape, dtype=a.dtype, device=a.device)
                    return a

                fake_args = tuple(_to_fake_in_mode(a) for a in args)
                fake_kwargs = {k: _to_fake_in_mode(v) for k, v in kwargs.items()}
                fake_out = original_forward(*fake_args, **fake_kwargs)

            # -- FX node construction ------------------------------------------
            def _to_node(a: Any) -> Any:
                if isinstance(a, TracingTensor):
                    return tracer._get_node(a)
                if isinstance(a, torch.Tensor):
                    return tracer._node_for_external_tensor(a)
                return a

            node_args = tuple(_to_node(a) for a in args)
            node_kwargs = {k: _to_node(v) for k, v in kwargs.items()}

            # Use create_node directly (instead of call_function) so we can
            # supply an explicit name for modules that lack a ``__name__``
            # attribute (all nn.Module instances).
            node_name = f"leaf_{qualified_name.replace('.', '_')}"
            node = tracer.graph.create_node(
                "call_function", module, args=node_args, kwargs=node_kwargs, name=node_name
            )
            node.meta["stack_trace"] = "".join(traceback.format_stack())
            node.meta["fn"] = module

            # -- wrap output(s) in TracingTensor --------------------------------
            if isinstance(fake_out, torch.Tensor):
                tt = tracer._make_tracing_tensor(
                    tracer._sym_shape_to_str_shape(fake_out.shape),
                    fake_out.dtype,
                    fake_out.device,
                    node,
                )
                node.meta["val"] = tt
                return tt

            if isinstance(fake_out, (list, tuple)):
                results: List[Any] = []
                for i, item in enumerate(fake_out):
                    if isinstance(item, torch.Tensor):
                        get_node = tracer.graph.call_function(
                            operator.getitem, args=(node, i), kwargs={}
                        )
                        tt = tracer._make_tracing_tensor(
                            tracer._sym_shape_to_str_shape(item.shape),
                            item.dtype,
                            item.device,
                            get_node,
                        )
                        get_node.meta["val"] = tt
                        results.append(tt)
                    else:
                        assert isinstance(item, (int, float, str)), (
                            f"Unexpected non-tensor output item type {type(item)} "
                            f"in leaf module {qualified_name!r}"
                        )
                        results.append(item)
                node.meta["val"] = type(fake_out)(results)
                return type(fake_out)(results)

            raise NotImplementedError(
                f"Leaf module {qualified_name!r} returned unexpected output type "
                f"{type(fake_out)}"
            )

        return _leaf_forward

    # ------------------------------------------------------------------
    # torch.cond support
    # ------------------------------------------------------------------

    def _register_callable(self, prefix: str, fn: Callable) -> str:
        """
        Register *fn* under a unique name derived from *prefix* and the
        function's ``__name__`` attribute, and return that name.

        The registered name is unique within :attr:`_callables` and follows
        the ``"_cb_<prefix>_<fn_name>_<counter>"`` pattern.

        :param prefix: Short tag describing the higher-order operator
            (e.g. ``"cond"``).
        :param fn: The Python callable to register.
        :return: The unique string key used to store *fn* in
            :attr:`_callables`.
        """
        fn_name = getattr(fn, "__name__", "fn")
        base = f"_cb_{prefix}_{fn_name}"
        cand = f"{base}_0"
        i = 0
        while cand in self._callables:
            i += 1
            cand = f"{base}_{i}"
        self._callables[cand] = fn
        return cand

    def _trace_branch(
        self, fn: Callable, operands: Union[List[Any], Tuple[Any, ...]]
    ) -> Tuple["GraphTracer", Any]:
        """
        Trace a branch function (*true_fn* or *false_fn* of ``torch.cond``)
        in an isolated sub-:class:`GraphTracer`.

        Input placeholders for the sub-graph are derived from *operands*:
        each :class:`TracingTensor` in *operands* produces one placeholder
        with matching shape, dtype, and device.  Non-tensor values are
        forwarded unchanged.

        After tracing, an ``output`` node is appended to the sub-graph.

        :param fn: The callable to trace (one branch of ``torch.cond``).
        :param operands: The operands that were passed to ``torch.cond``'s
            fourth argument.
        :return: A ``(sub_tracer, out)`` tuple where *sub_tracer* is the
            fully-traced sub-:class:`GraphTracer` and *out* is the raw return
            value of *fn* (containing :class:`TracingTensor` instances from
            the sub-tracer).
        """
        from ._patches import (
            _cond_replacement_ctx,
            _check_replacement_ctx,
            _full_replacement_ctx,
            _zeros_replacement_ctx,
            _ones_replacement_ctx,
            _while_loop_replacement_ctx,
        )

        sub = GraphTracer(verbose=self.verbose)
        # Share symbolic dimension mappings so the same dynamic-dim names are
        # reused in branch graphs.
        sub._mapped_dimension = dict(self._mapped_dimension)
        sub._sym_int_to_dynamic_dimension = dict(self._sym_int_to_dynamic_dimension)
        # Share the parent's ShapeEnv and FakeTensorMode so that unbacked SymInts
        # stored in _mapped_dimension (which were created by the parent's ShapeEnv)
        # remain valid when sub.make_fake() uses them inside sub._fake_mode.  Without
        # this sharing, checks like is_contiguous for multi-dimensional tensors with
        # dynamic shapes raise "vr must not be None for symbol uN" because the SymInt
        # belongs to the parent's ShapeEnv while the FakeTensorMode runs under the
        # sub-tracer's independent ShapeEnv.
        sub._shape_env = self._shape_env
        sub._fake_mode = self._fake_mode

        sub_operands: List[Any] = []
        for i, op in enumerate(operands):
            if isinstance(op, TracingTensor):
                node = sub.graph.placeholder(f"operand_{i}")
                node.meta["val"] = op
                fake_op = sub._make_tracing_tensor(op.shape, op.dtype, op.device, node)
                sub_operands.append(fake_op)
            else:
                sub_operands.append(op)

        with (
            _cond_replacement_ctx(sub),
            _check_replacement_ctx(sub),
            _full_replacement_ctx(sub),
            _zeros_replacement_ctx(sub),
            _ones_replacement_ctx(sub),
            _while_loop_replacement_ctx(sub),
        ):
            out = fn(*sub_operands)

        def _to_node(x: Any) -> Any:
            if isinstance(x, TracingTensor):
                return sub._get_node(x)
            return x

        import torch.utils._pytree as pytree

        out_val = pytree.tree_map(_to_node, out)
        sub.graph.output(out_val)
        return sub, out

    def _handle_cond(
        self,
        pred: Any,
        true_fn: Callable,
        false_fn: Callable,
        operands: Union[List[Any], Tuple[Any, ...]] = (),
    ) -> Any:
        """
        Handle a ``torch.cond`` call intercepted during graph tracing.

        This method is invoked by :func:`_cond_replacement_ctx`'s handler
        whenever user code calls ``torch.cond`` while a :meth:`trace` is in
        progress.  It:

        1. Registers *true_fn* and *false_fn* in :attr:`_callables`.
        2. Traces each branch in a private sub-:class:`GraphTracer` and
           stores the result in :attr:`_sub_tracers`.
        3. Emits a ``call_function`` node for ``torch.cond`` in the **main**
           graph, passing the branch callables directly as node arguments
           (FX allows non-tensor constant args for ``call_function`` nodes).
        4. Wraps the outputs in fresh :class:`TracingTensor` instances bound
           to the new node and returns them.

        :param pred: The predicate—either a scalar bool/int or a
            :class:`TracingTensor` of shape ``()`` and dtype ``bool``.
        :param true_fn: Branch called when *pred* is ``True``.
        :param false_fn: Branch called when *pred* is ``False``.
        :param operands: Positional operands forwarded to the selected branch.
        :return: A :class:`TracingTensor` (single output) or a ``list`` /
            ``tuple`` of :class:`TracingTensor` instances (multiple outputs).
        """
        # Always use the real torch.cond (captured at import time) as the FX
        # node target so that nested tracing contexts do not accidentally record
        # the shim function instead.
        from ._patches import _ORIGINAL_TORCH_COND

        cond_target = _ORIGINAL_TORCH_COND
        # --- node for the predicate ---
        if isinstance(pred, TracingTensor):
            pred_node: Any = self._get_node(pred)
        elif isinstance(pred, TracingBool) and pred._node is not None:
            # Predicate is a symbolic boolean that also carries an FX node
            # (e.g. produced by ``tensor.numel() > 0`` during tracing).
            # Use the FX node directly so the ONNX If node receives a proper
            # bool tensor instead of a raw TracingBool.
            pred_node = pred._node
        else:
            pred_node = pred

        # --- nodes for each operand ---
        operand_nodes: List[Any] = [
            (self._get_node(op) if isinstance(op, TracingTensor) else op) for op in operands
        ]

        # --- register callables + trace branches ---
        true_name = self._register_callable("cond", true_fn)
        false_name = self._register_callable("cond", false_fn)

        sub_true, true_out = self._trace_branch(true_fn, list(operands))
        sub_false, _ = self._trace_branch(false_fn, list(operands))

        self._sub_tracers[true_name] = sub_true
        self._sub_tracers[false_name] = sub_false

        # --- create callable attribute nodes ---
        true_fn_node = self.graph.get_attr(true_name)
        true_fn_node.meta["stack_trace"] = "".join(traceback.format_stack())
        true_fn_node.meta["callable"] = true_fn
        false_fn_node = self.graph.get_attr(false_name)
        false_fn_node.meta["stack_trace"] = "".join(traceback.format_stack())
        false_fn_node.meta["callable"] = false_fn

        # --- emit the main cond node ---
        node = self.graph.call_function(
            cond_target, args=(pred_node, true_fn_node, false_fn_node, operand_nodes), kwargs={}  # type: ignore
        )
        node.meta["stack_trace"] = "".join(traceback.format_stack())

        # --- build output TracingTensors from the true branch's shapes ---
        if isinstance(true_out, TracingTensor):
            tt = self._make_tracing_tensor(true_out.shape, true_out.dtype, true_out.device, node)
            node.meta["val"] = tt
            return tt

        if isinstance(true_out, (list, tuple)):
            results: List[Any] = []
            for i, item in enumerate(true_out):
                if isinstance(item, TracingTensor):
                    get_node = self.graph.call_function(
                        operator.getitem, args=(node, i), kwargs={}
                    )
                    tt = self._make_tracing_tensor(item.shape, item.dtype, item.device, get_node)
                    get_node.meta["val"] = tt
                    results.append(tt)
                else:
                    assert isinstance(item, (int, float, str)), (
                        f"Expected int, float, or str for non-tensor cond output item, "
                        f"got: {type(item)}. Tensor outputs should have been handled as "
                        f"TracingTensor instances earlier in the conditional flow."
                    )
                    results.append(item)
            node.meta["val"] = type(true_out)(results)
            return type(true_out)(results)

        raise NotImplementedError(
            f"torch.cond: unexpected output type from branch function: {type(true_out)}"
        )

    # ------------------------------------------------------------------
    # torch.ops.higher_order.scan support
    # ------------------------------------------------------------------

    def _handle_scan(
        self,
        f: Callable,
        init_states: List[Any],
        scan_inputs: List[Any],
        additional_inputs: Optional[List[Any]] = None,
        dim: int = 0,
        reverse: bool = False,
    ) -> Any:
        """
        Handles a ``torch.ops.higher_order.scan`` call intercepted during graph
        tracing.

        This method is invoked by :func:`_scan_replacement_ctx`'s handler
        whenever user code calls ``torch.ops.higher_order.scan`` while a
        :meth:`trace` is in progress.  It:

        1. Registers *f* in :attr:`_callables` and emits a ``get_attr`` node.
        2. Traces the scan body in a private sub-:class:`GraphTracer` to
           determine output shapes and stores the result in :attr:`_sub_tracers`.
        3. Emits a ``call_function`` node for ``torch.ops.higher_order.scan``
           in the **main** graph.
        4. Wraps the outputs in fresh :class:`TracingTensor` instances,
           returning them as a flat tuple
           ``(carry_0_final, ..., carry_n_final, scan_out_0, ..., scan_out_m)``,
           mirroring what the real scan operator returns.

        :param f: The scan body callable.  Receives
            ``(*carry_states, *scan_elements, *additional_inputs)`` and returns
            a list ``[new_carry_0, ..., new_carry_n, scan_out_0, ..., scan_out_m]``.
        :param init_states: Initial carry states (list of
            :class:`TracingTensor` instances or concrete tensors).
        :param scan_inputs: Tensors to scan over (first dimension is the scan
            dimension).
        :param additional_inputs: Additional inputs forwarded to *f* unchanged
            on every step.
        :param dim: Scan dimension (must be 0 for current PyTorch ≥ 2.7).
        :param reverse: Whether to scan in reverse order.
        :return: A flat tuple of :class:`TracingTensor` instances
            ``(carry_0_final, ..., scan_out_0_accum, ...)``.
        """
        from ._patches import _scan_replacement_ctx, _check_replacement_ctx, _ORIGINAL_TORCH_SCAN

        additional_inputs = list(additional_inputs) if additional_inputs else []
        n_carry = len(init_states)

        # --- Register the scan body callable and create a get_attr node ---
        fn_name = self._register_callable("scan", f)
        get_attr_node = self.graph.get_attr(fn_name)
        get_attr_node.meta["stack_trace"] = "".join(traceback.format_stack())
        get_attr_node.meta["callable"] = f

        # --- Trace the scan body to determine output shapes ---
        sub = GraphTracer(verbose=self.verbose)
        # Share symbolic dimension mappings so symbolic dim names are consistent.
        sub._mapped_dimension = dict(self._mapped_dimension)
        sub._sym_int_to_dynamic_dimension = dict(self._sym_int_to_dynamic_dimension)
        # Share the parent's ShapeEnv and FakeTensorMode (same rationale as
        # _trace_branch) so that parent SymInts remain valid inside sub._fake_mode.
        sub._shape_env = self._shape_env
        sub._fake_mode = self._fake_mode

        sub_operands: List[Any] = []

        # Carry (init state) inputs: same shape as init_states.
        for i, s in enumerate(init_states):
            if isinstance(s, TracingTensor):
                ph = sub.graph.placeholder(f"carry_{i}")
                tt = sub._make_tracing_tensor(s.shape, s.dtype, s.device, ph)
                ph.meta["val"] = tt
                sub_operands.append(tt)
            else:
                sub_operands.append(s)

        # Scan-element inputs: first dimension (scan dim) stripped.
        for i, s in enumerate(scan_inputs):
            if isinstance(s, TracingTensor):
                stripped_shape = (
                    TracingShape(s.shape[1:]) if len(s.shape) > 1 else TracingShape(())
                )
                ph = sub.graph.placeholder(f"scan_elem_{i}")
                tt = sub._make_tracing_tensor(stripped_shape, s.dtype, s.device, ph)
                ph.meta["val"] = tt
                sub_operands.append(tt)
            else:
                sub_operands.append(s)

        # Additional inputs: passed unchanged (full shape).
        for i, s in enumerate(additional_inputs):
            if isinstance(s, TracingTensor):
                ph = sub.graph.placeholder(f"additional_{i}")
                tt = sub._make_tracing_tensor(s.shape, s.dtype, s.device, ph)
                ph.meta["val"] = tt
                sub_operands.append(tt)
            else:
                sub_operands.append(s)

        with _scan_replacement_ctx(sub), _check_replacement_ctx(sub):
            body_out = f(*sub_operands)

        body_out_list: List[Any] = (
            list(body_out) if isinstance(body_out, (list, tuple)) else [body_out]
        )

        def _body_to_node(x: Any) -> Any:
            if isinstance(x, TracingTensor):
                return sub._get_node(x)
            return x

        import torch.utils._pytree as pytree

        out_val = pytree.tree_map(_body_to_node, body_out)
        sub.graph.output(out_val)
        self._sub_tracers[fn_name] = sub

        # --- Get FX nodes for all scan arguments ---
        def _to_node(s: Any) -> Any:
            if isinstance(s, TracingTensor):
                return self._get_node(s)
            return s

        init_nodes = [_to_node(s) for s in init_states]
        scan_input_nodes = [_to_node(s) for s in scan_inputs]
        add_input_nodes = [_to_node(s) for s in additional_inputs]

        # --- Emit the scan call_function node ---
        scan_op = _ORIGINAL_TORCH_SCAN
        node = self.graph.call_function(
            scan_op,  # type: ignore
            args=(get_attr_node, init_nodes, scan_input_nodes, add_input_nodes),
            kwargs={},
        )
        node.meta["stack_trace"] = "".join(traceback.format_stack())

        # --- Build output TracingTensors ---
        # Outputs follow the flat layout:
        #   (carry_0_final, ..., carry_n_final, scan_out_0_accum, ..., scan_out_m_accum)
        # - carry_i_final.shape  = init_states[i].shape
        # - scan_out_j.shape     = (scan_dim, *body_out_list[n_carry+j].shape)
        #   where scan_dim = scan_inputs[k].shape[0] for any valid k.
        results: List[Any] = []

        # Carry outputs.
        for i in range(n_carry):
            s = init_states[i]
            if isinstance(s, TracingTensor):
                get_node = self.graph.call_function(operator.getitem, args=(node, i), kwargs={})
                tt = self._make_tracing_tensor(s.shape, s.dtype, s.device, get_node)
                get_node.meta["val"] = tt
                results.append(tt)
            else:
                results.append(s)

        # Scan-accumulation outputs.
        n_body_scan_outs = len(body_out_list) - n_carry
        # Determine the scan dimension size from the first scan input.
        if (
            scan_inputs
            and isinstance(scan_inputs[0], TracingTensor)
            and len(scan_inputs[0].shape) > 0
        ):
            scan_dim_val: Any = scan_inputs[0].shape[0]
        else:
            scan_dim_val = 0

        for j in range(n_body_scan_outs):
            body_scan_out = body_out_list[n_carry + j]
            out_idx = n_carry + j
            if isinstance(body_scan_out, TracingTensor):
                accum_shape = TracingShape((scan_dim_val, *body_scan_out.shape.dims))
                get_node = self.graph.call_function(
                    operator.getitem, args=(node, out_idx), kwargs={}
                )
                tt = self._make_tracing_tensor(
                    accum_shape, body_scan_out.dtype, body_scan_out.device, get_node
                )
                get_node.meta["val"] = tt
                results.append(tt)
            else:
                results.append(None)

        node.meta["val"] = tuple(results)
        return tuple(results)

    # ------------------------------------------------------------------
    # torch._higher_order_ops.while_loop support
    # ------------------------------------------------------------------

    def _handle_while_loop(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        carried_inputs: Union[List[Any], Tuple[Any, ...]],
        additional_inputs: Optional[Union[List[Any], Tuple[Any, ...]]] = None,
    ) -> Any:
        """
        Handles a ``torch._higher_order_ops.while_loop`` call intercepted
        during graph tracing.

        This method is invoked by :func:`_while_loop_replacement_ctx`'s
        handler whenever user code calls ``while_loop`` while a
        :meth:`trace` is in progress.  It:

        1. Registers *cond_fn* and *body_fn* in :attr:`_callables` and emits
           ``get_attr`` nodes for them.
        2. Traces both functions in private sub-:class:`GraphTracer` instances
           (using *carried_inputs* + *additional_inputs* as placeholders) and
           stores the results in :attr:`_sub_tracers`.
        3. Emits a ``call_function`` node for
           ``torch._higher_order_ops.while_loop`` in the **main** graph.
        4. Wraps the outputs in fresh :class:`TracingTensor` instances whose
           shapes are taken from the *body_fn* traced outputs (which match the
           shapes of *carried_inputs*).

        :param cond_fn: Condition callable.  Receives
            ``(*carried_inputs, *additional_inputs)`` and returns a scalar
            bool tensor.
        :param body_fn: Body callable.  Receives
            ``(*carried_inputs, *additional_inputs)`` and returns a tuple of
            tensors with the same shapes as *carried_inputs*.
        :param carried_inputs: The initial loop-variable tensors.
        :param additional_inputs: Extra read-only tensors forwarded to both
            *cond_fn* and *body_fn* unchanged.
        :returns: A tuple of :class:`TracingTensor` instances corresponding to
            the final loop-variable values.
        """
        from ._patches import _ORIGINAL_TORCH_WHILE_LOOP

        additional_inputs = list(additional_inputs) if additional_inputs else []
        while_loop_target = _ORIGINAL_TORCH_WHILE_LOOP
        assert (
            while_loop_target is not None
        ), "torch._higher_order_ops.while_loop is not available on this PyTorch version"

        # --- Register cond_fn and body_fn as callables ---
        cond_name = self._register_callable("while_cond", cond_fn)
        body_name = self._register_callable("while_body", body_fn)

        # --- Trace both functions (all operands = carried + additional) ---
        all_operands: List[Any] = list(carried_inputs) + additional_inputs
        sub_cond, _ = self._trace_branch(cond_fn, all_operands)
        sub_body, body_out = self._trace_branch(body_fn, all_operands)

        self._sub_tracers[cond_name] = sub_cond
        self._sub_tracers[body_name] = sub_body

        # --- Create get_attr nodes for the callables ---
        cond_fn_node = self.graph.get_attr(cond_name)
        cond_fn_node.meta["stack_trace"] = "".join(traceback.format_stack())
        cond_fn_node.meta["callable"] = cond_fn
        body_fn_node = self.graph.get_attr(body_name)
        body_fn_node.meta["stack_trace"] = "".join(traceback.format_stack())
        body_fn_node.meta["callable"] = body_fn

        # --- Get FX nodes for all arguments ---
        def _to_node(x: Any) -> Any:
            if isinstance(x, TracingTensor):
                return self._get_node(x)
            return x

        carried_nodes = [_to_node(x) for x in carried_inputs]
        additional_nodes = [_to_node(x) for x in additional_inputs]

        # --- Emit the while_loop call_function node ---
        node = self.graph.call_function(
            while_loop_target,
            args=(cond_fn_node, body_fn_node, carried_nodes, additional_nodes),
            kwargs={},
        )
        node.meta["stack_trace"] = "".join(traceback.format_stack())

        # --- Build output TracingTensors from body_fn output shapes ---
        # while_loop outputs have the same shapes/dtypes as carried_inputs.
        body_out_list: List[Any] = (
            list(body_out) if isinstance(body_out, (list, tuple)) else [body_out]
        )
        results: List[Any] = []
        for i, item in enumerate(body_out_list):
            if isinstance(item, TracingTensor):
                get_node = self.graph.call_function(operator.getitem, args=(node, i), kwargs={})
                tt = self._make_tracing_tensor(item.shape, item.dtype, item.device, get_node)
                get_node.meta["val"] = tt
                results.append(tt)
            else:
                results.append(item)

        node.meta["val"] = tuple(results)
        return tuple(results)

    def _handle_check(self, cond: Any, msg: Any = None) -> None:
        """
        Tracing-aware replacement for ``torch._check`` called via
        :func:`_check_replacement_ctx`.

        ``torch._check`` is a runtime assertion that *cond* holds.  During
        :class:`GraphTracer` tracing the condition is frequently a
        :class:`TracingBool` (produced by comparing a symbolic
        :class:`TracingInt` dimension, e.g. ``tensor.shape[0] > 0``) or a
        concrete ``False`` caused by symbolic dimensions being stored as ``0``
        in the underlying tensor storage.

        This method:

        1. **Silently accepts** any :class:`TracingBool` condition without
           calling ``bool()`` on it (which would raise :exc:`ValueError`).
        2. **Registers** symbolic :class:`TracingBool` conditions in the
           module-level :data:`~yobx.torch.new_tracing.shape._known_true_conditions`
           set so that later uses of the same comparison (e.g. ``if
           tensor.shape[0] > 0:``) can resolve to ``True`` via
           :meth:`TracingBool.__bool__`.
        3. **Silently skips** concrete ``False`` values — these occur when a
           symbolic dimension (stored as ``0``) makes a comparison that is
           normally True at runtime evaluate to ``False`` at trace time.
        4. **No-ops** on concrete ``True`` values.

        :param cond: The condition passed to ``torch._check``.  May be a
            concrete :class:`bool`, a :class:`TracingBool`, or any other value.
        :param msg: Optional error message (ignored during tracing).
        """
        if isinstance(cond, TracingBool):
            register_condition(cond)
            return
        if isinstance(cond, bool):
            # Silently accept both True and False during tracing; a False
            # value typically arises when a symbolic dimension stored as 0
            # makes an otherwise-True condition evaluate to False.
            return
        # For any other type (including ints from shape ops), accept silently.

    def _handle_full(self, size: Any, fill_value: Any, **kwargs: Any) -> Any:
        """
        Tracing-aware replacement for ``torch.full`` called via
        :func:`_full_replacement_ctx`.

        Intercepts constructor calls and emits a corresponding FX node so
        that the result is always a :class:`TracingTensor`, whether *size*
        contains symbolic :class:`TracingInt` values or plain Python ``int``
        values.  This ensures that ``torch.full`` calls whose dimensions are
        fully resolved to concrete integers (e.g. the static branch of a
        guard like ``if x.shape[0] != 0``) still appear in the FX graph.

        FakeTensor kernel invocations within :meth:`dispatch` never reach this
        handler because :meth:`dispatch` temporarily restores the original
        ``torch.full`` before entering the FakeTensorMode context.  The handler
        is only reached for ``torch.full`` calls made from user model code
        during normal tracing.

        :param size: Full size argument passed to ``torch.full``.
        :param fill_value: Fill value passed to ``torch.full``.
        :param kwargs: Additional keyword arguments for ``torch.full``.

        Returns:
            Returns a :class:`TracingTensor` wrapping an FX node for the call.
        """
        from ._patches import _ORIGINAL_TORCH_FULL

        if isinstance(size, torch.Size):
            size = tuple(size)
        if not isinstance(size, (tuple, list)):
            return _ORIGINAL_TORCH_FULL(size, fill_value, **kwargs)

        traced_size: List[Union[int, torch.SymInt]] = []
        node_size: List[Any] = []
        for dim in size:
            if isinstance(dim, TracingInt):
                if dim.is_static:
                    traced_size.append(dim.value)  # type: ignore[arg-type]
                    node_size.append(dim)
                    continue
                assert isinstance(
                    dim.value, str
                ), f"Expected string symbolic dimension value, got {type(dim.value)}"
                dim_key = self._token_replace(dim.value)
                assert isinstance(
                    dim_key, str
                ), f"Expected string type for symbolic dimension key, got {type(dim_key)}"
                if dim_key in self._mapped_dimension:
                    symd = self._mapped_dimension[dim_key]
                else:
                    symd = self._shape_env.create_unbacked_symint()
                    self._mapped_dimension[dim_key] = symd
                    symd_name = self._sym_int_to_str(symd)
                    assert isinstance(symd_name, str), "type checking"
                    self._sym_int_to_dynamic_dimension[symd_name] = dim_key
                traced_size.append(symd)
                node_size.append(TracingInt(dim_key))
            else:
                assert isinstance(dim, int), f"Unexpected full size element type {type(dim)}"
                traced_size.append(dim)
                node_size.append(dim)

        with self._fake_mode:
            fake_res = _ORIGINAL_TORCH_FULL(tuple(traced_size), fill_value, **kwargs)

        node = self.graph.call_function(
            _ORIGINAL_TORCH_FULL, args=(tuple(node_size), fill_value), kwargs=kwargs
        )
        res = self._make_tracing_tensor(
            self._sym_shape_to_str_shape(fake_res.shape), fake_res.dtype, fake_res.device, node
        )
        node.meta["val"] = res
        node.meta["fake_val"] = fake_res
        node.meta["stack_trace"] = "".join(traceback.format_stack())
        return res

    def _handle_zeros(self, size: Any, **kwargs: Any) -> Any:
        """
        Tracing-aware replacement for ``torch.zeros`` called via
        :func:`_zeros_replacement_ctx`.

        Intercepts constructor calls where *size* contains symbolic
        :class:`TracingInt` values and emits a corresponding FX node without
        requiring eager execution with concrete Python ``int`` dimensions.

        :param size: Size argument passed to ``torch.zeros``.
        :param kwargs: Additional keyword arguments for ``torch.zeros``.

        Returns:
            Returns a :class:`TracingTensor` when symbolic dimensions are present,
            otherwise returns the eager ``torch.zeros`` result.
        """
        from ._patches import _ORIGINAL_TORCH_ZEROS

        if isinstance(size, torch.Size):
            size = tuple(size)
        if not isinstance(size, (tuple, list)):
            return _ORIGINAL_TORCH_ZEROS(size, **kwargs)
        if not any(isinstance(dim, TracingInt) for dim in size):
            return _ORIGINAL_TORCH_ZEROS(size, **kwargs)

        traced_size: List[Union[int, torch.SymInt]] = []
        node_size: List[Any] = []
        for dim in size:
            if isinstance(dim, TracingInt):
                if dim.is_static:
                    traced_size.append(dim.value)  # type: ignore[arg-type]
                    node_size.append(dim)
                    continue
                assert isinstance(
                    dim.value, str
                ), f"Expected string symbolic dimension value, got {type(dim.value)}"
                dim_key = self._token_replace(dim.value)
                assert isinstance(
                    dim_key, str
                ), f"Expected string type for symbolic dimension key, got {type(dim_key)}"
                if dim_key in self._mapped_dimension:
                    symd = self._mapped_dimension[dim_key]
                else:
                    symd = self._shape_env.create_unbacked_symint()
                    self._mapped_dimension[dim_key] = symd
                    symd_name = self._sym_int_to_str(symd)
                    assert isinstance(symd_name, str), "type checking"
                    self._sym_int_to_dynamic_dimension[symd_name] = dim_key
                traced_size.append(symd)
                node_size.append(TracingInt(dim_key))
            else:
                assert isinstance(dim, int), f"Unexpected full size element type {type(dim)}"
                traced_size.append(dim)
                node_size.append(dim)

        with self._fake_mode:
            fake_res = _ORIGINAL_TORCH_ZEROS(tuple(traced_size), **kwargs)

        node = self.graph.call_function(
            _ORIGINAL_TORCH_ZEROS, args=(tuple(node_size),), kwargs=kwargs
        )
        res = self._make_tracing_tensor(
            self._sym_shape_to_str_shape(fake_res.shape), fake_res.dtype, fake_res.device, node
        )
        node.meta["val"] = res
        node.meta["fake_val"] = fake_res
        node.meta["stack_trace"] = "".join(traceback.format_stack())
        return res

    def _handle_ones(self, size: Any, **kwargs: Any) -> Any:
        """
        Tracing-aware replacement for ``torch.ones`` called via
        :func:`_ones_replacement_ctx`.

        Intercepts constructor calls where *size* contains symbolic
        :class:`TracingInt` values and emits a corresponding FX node without
        requiring eager execution with concrete Python ``int`` dimensions.

        :param size: Size argument passed to ``torch.ones``.
        :param kwargs: Additional keyword arguments for ``torch.ones``.

        Returns:
            Returns a :class:`TracingTensor` when symbolic dimensions are present,
            otherwise returns the eager ``torch.ones`` result.
        """
        from ._patches import _ORIGINAL_TORCH_ONES

        if isinstance(size, torch.Size):
            size = tuple(size)
        if not isinstance(size, (tuple, list)):
            return _ORIGINAL_TORCH_ONES(size, **kwargs)
        if not any(isinstance(dim, TracingInt) for dim in size):
            return _ORIGINAL_TORCH_ONES(size, **kwargs)

        traced_size: List[Union[int, torch.SymInt]] = []
        node_size: List[Any] = []
        for dim in size:
            if isinstance(dim, TracingInt):
                if dim.is_static:
                    traced_size.append(dim.value)  # type: ignore[arg-type]
                    node_size.append(dim)
                    continue
                assert isinstance(
                    dim.value, str
                ), f"Expected string symbolic dimension value, got {type(dim.value)}"
                dim_key = self._token_replace(dim.value)
                assert isinstance(
                    dim_key, str
                ), f"Expected string type for symbolic dimension key, got {type(dim_key)}"
                if dim_key in self._mapped_dimension:
                    symd = self._mapped_dimension[dim_key]
                else:
                    symd = self._shape_env.create_unbacked_symint()
                    self._mapped_dimension[dim_key] = symd
                    symd_name = self._sym_int_to_str(symd)
                    assert isinstance(symd_name, str), "type checking"
                    self._sym_int_to_dynamic_dimension[symd_name] = dim_key
                traced_size.append(symd)
                node_size.append(TracingInt(dim_key))
            else:
                assert isinstance(dim, int), f"Unexpected full size element type {type(dim)}"
                traced_size.append(dim)
                node_size.append(dim)

        with self._fake_mode:
            fake_res = _ORIGINAL_TORCH_ONES(tuple(traced_size), **kwargs)

        node = self.graph.call_function(
            _ORIGINAL_TORCH_ONES, args=(tuple(node_size),), kwargs=kwargs
        )
        res = self._make_tracing_tensor(
            self._sym_shape_to_str_shape(fake_res.shape), fake_res.dtype, fake_res.device, node
        )
        node.meta["val"] = res
        node.meta["fake_val"] = fake_res
        node.meta["stack_trace"] = "".join(traceback.format_stack())
        return res

    def _handle_arange(self, *args: Any, **kwargs: Any) -> Any:
        """
        Tracing-aware replacement for ``torch.arange`` called via
        :func:`_arange_replacement_ctx`.

        Intercepts calls where *start*, *end*, or *step* is a symbolic
        :class:`TracingInt` and emits a corresponding FX node without
        requiring eager execution with concrete Python ``int`` values.

        The signature mirrors ``torch.arange``:

        * ``arange(end, ...)``
        * ``arange(start, end, ...)``
        * ``arange(start, end, step, ...)``

        For symbolic :class:`TracingInt` arguments that originate from a
        placeholder tensor dimension (``_source_node`` is set), an
        ``aten.sym_size.int`` FX node is emitted and used as the graph
        argument, producing a proper ONNX scalar with known rank/type.
        For other symbolic TracingInts a ``TracingInt`` value string is
        recorded as a fallback.

        :param args: Positional arguments as passed to ``torch.arange``.
        :param kwargs: Keyword arguments forwarded to ``torch.arange``.

        Returns:
            A :class:`TracingTensor` when any of *start*, *end*, or *step* is
            a symbolic :class:`TracingInt`; otherwise the eager
            ``torch.arange`` result.
        """
        from ._patches import _ORIGINAL_TORCH_ARANGE

        has_tracing_int = any(isinstance(a, TracingInt) for a in args)
        if not has_tracing_int:
            return _ORIGINAL_TORCH_ARANGE(*args, **kwargs)

        # Resolve TracingInt arguments:
        # - fake_args: SymInt or int values for FakeTensorMode shape inference.
        # - node_args: FX node args for the recorded call_function node.
        #   Symbolic TracingInts with a known source placeholder emit an
        #   aten.sym_size.int node so the ONNX interpreter can materialize
        #   the dimension as a typed scalar.  Other TracingInts fall back to
        #   recording the TracingInt directly (handled as a dimension string
        #   by the ONNX interpreter).
        fake_args: List[Any] = []
        node_args: List[Any] = []
        for a in args:
            if isinstance(a, TracingInt):
                fake_args.append(self._tracing_int_to_fake(a))
                if a.is_static:
                    node_args.append(a.value)
                elif a._source_node is not None:
                    # Emit aten.sym_size.int so the ONNX interpreter gets a
                    # properly-typed scalar node whose rank is known.
                    node_args.append(self._emit_sym_size_node(a))
                else:
                    dim_key = self._token_replace(a.value)
                    assert isinstance(
                        dim_key, str
                    ), f"Expected string type for symbolic dimension key, got {type(dim_key)}"
                    if dim_key not in self._mapped_dimension:
                        symd = self._shape_env.create_unbacked_symint()
                        self._mapped_dimension[dim_key] = symd
                        symd_name = self._sym_int_to_str(symd)
                        assert isinstance(symd_name, str), "type checking"
                        self._sym_int_to_dynamic_dimension[symd_name] = dim_key
                    node_args.append(TracingInt(dim_key))
            else:
                assert isinstance(
                    a, (int, float)
                ), f"Unexpected type {type(a)} for arange argument"
                fake_args.append(a)
                node_args.append(a)

        with self._fake_mode:
            fake_res = _ORIGINAL_TORCH_ARANGE(*fake_args, **kwargs)

        node = self.graph.call_function(
            _ORIGINAL_TORCH_ARANGE, args=tuple(node_args), kwargs=kwargs
        )
        res = self._make_tracing_tensor(
            self._sym_shape_to_str_shape(fake_res.shape), fake_res.dtype, fake_res.device, node
        )
        node.meta["val"] = res
        node.meta["fake_val"] = fake_res
        node.meta["stack_trace"] = "".join(traceback.format_stack())
        return res

    def _handle_tensor_split(self, input: Any, indices_or_sections: Any, dim: int = 0) -> Any:
        """
        Tracing-aware replacement for ``torch.tensor_split`` called via
        :func:`_tensor_split_replacement_ctx`.

        The native C++ kernel for ``aten::tensor_split`` reads the *values*
        from ``indices_or_sections`` (via ``.item()`` or ``.tolist()``) before
        dispatching, which fails for
        :class:`~yobx.torch.new_tracing.tensor.TracingTensor` instances that
        carry no backing storage.

        When ``indices_or_sections`` is a :class:`TracingTensor` with a
        statically-known 1-D shape ``(n,)``, this method:

        1. Builds concrete surrogate tensors (no symbolic/tracing tensors) and
           calls the original ``torch.tensor_split`` to infer the output shapes.
        2. Emits a ``call_function`` FX node for
           ``aten.tensor_split.Tensor_indices_or_sections``.
        3. Emits one ``operator.getitem`` FX node per output chunk.
        4. Returns a tuple of ``n + 1`` :class:`TracingTensor` instances.

        When ``indices_or_sections`` is a plain ``int`` or ``list`` (or a
        concrete :class:`torch.Tensor`), the call is forwarded to the standard
        :meth:`dispatch` path which handles those cases correctly.

        :param input: The tensor to split.  Must be a
            :class:`TracingTensor`.
        :param indices_or_sections: An ``int``, a sequence of ``int``, a
            concrete :class:`torch.Tensor`, or a
            :class:`TracingTensor` of shape ``(n,)``.
        :param dim: The dimension along which to split.

        Returns:
            A tuple of :class:`TracingTensor` instances, one per output chunk.
        """
        # ------------------------------------------------------------------ #
        # Guard: if the input tensor is NOT a TracingTensor (e.g. it is a   #
        # FakeTensor produced inside FakeTensorMode during dispatch()'s       #
        # fake evaluation of another op), fall back to the original           #
        # implementation immediately.  The patch is only meant for user-level #
        # calls where the input is a genuine TracingTensor.                   #
        # ------------------------------------------------------------------ #
        if not isinstance(input, TracingTensor):
            from ._patches import _ORIGINAL_TORCH_TENSOR_SPLIT

            return _ORIGINAL_TORCH_TENSOR_SPLIT(input, indices_or_sections, dim)
        # ------------------------------------------------------------------ #
        # Case 1: indices_or_sections is NOT a TracingTensor.                 #
        # Route through the standard dispatch path which handles int/list and #
        # concrete tensor arguments correctly.                                 #
        # ------------------------------------------------------------------ #
        if not isinstance(indices_or_sections, TracingTensor):
            if isinstance(indices_or_sections, int):
                return self.dispatch(
                    torch.ops.aten.tensor_split.sections, (input, indices_or_sections, dim), {}
                )
            if isinstance(indices_or_sections, (list, tuple)):
                return self.dispatch(
                    torch.ops.aten.tensor_split.indices, (input, indices_or_sections, dim), {}
                )
            if isinstance(indices_or_sections, torch.Tensor):
                if indices_or_sections.dim() == 0:
                    return self.dispatch(
                        torch.ops.aten.tensor_split.sections,
                        (input, int(indices_or_sections.item()), dim),
                        {},
                    )
                return self.dispatch(
                    torch.ops.aten.tensor_split.indices,
                    (input, indices_or_sections.tolist(), dim),
                    {},
                )
            # Fall back to original for anything else (should not normally occur).
            return _ORIGINAL_TORCH_TENSOR_SPLIT(input, indices_or_sections, dim)  # type: ignore

        # ------------------------------------------------------------------ #
        # Case 2: indices_or_sections IS a TracingTensor.                     #
        # The C++ kernel would try to read the tensor values here; intercept. #
        # ------------------------------------------------------------------ #
        indices_shape = indices_or_sections.shape
        assert (
            indices_or_sections.dim() == 1
        ), f"TracingTensor indices_or_sections must be 1-D, got shape {indices_shape}"

        n_indices_dim = indices_shape[0]
        assert isinstance(n_indices_dim, int), (
            f"Dynamic number of split indices is not supported "
            f"in new-tracing mode; shape[0]={n_indices_dim!r}"
        )
        n_indices = n_indices_dim
        n_outputs = n_indices + 1

        # Build a concrete surrogate for ``input`` for shape inference.
        concrete_x_shape = tuple(d if isinstance(d, int) else 1 for d in input.shape)
        concrete_x = torch.zeros(concrete_x_shape, dtype=input.dtype)

        # Build concrete surrogate indices (equidistant within the split dim).
        concrete_dim_size = concrete_x_shape[dim]
        step = max(1, concrete_dim_size // n_outputs)
        concrete_indices = torch.tensor(
            [min(concrete_dim_size, step * (i + 1)) for i in range(n_indices)], dtype=torch.int64
        )

        # Shape inference using real (non-tracing, non-fake) tensors.
        concrete_results = _ORIGINAL_TORCH_TENSOR_SPLIT(concrete_x, concrete_indices, dim)  # type: ignore

        # Emit FX node for the split op.
        func = torch.ops.aten.tensor_split.Tensor_indices_or_sections
        x_node = self._get_node(input)
        indices_node = self._get_node(indices_or_sections)
        node = self.graph.call_function(func, args=(x_node, indices_node, dim), kwargs={})

        # Emit one getitem node per output chunk and wrap in TracingTensor.
        results = []
        for i, concrete_out in enumerate(concrete_results):
            get_node = self.graph.call_function(operator.getitem, args=(node, i), kwargs={})
            tt = self._make_tracing_tensor(
                TracingShape(tuple(concrete_out.shape)),
                concrete_out.dtype,
                concrete_out.device,
                get_node,
            )
            get_node.meta["val"] = tt
            results.append(tt)

        node.meta["val"] = tuple(results)
        node.meta["stack_trace"] = "".join(traceback.format_stack())
        return tuple(results)

    # ------------------------------------------------------------------
    # Public tracing entry point
    # ------------------------------------------------------------------

    def make_names(self, n: int, name: str, arg, treespec):
        """
        Generate a list of *n* unique child names derived from *name*.

        For a list or tuple *arg*, names are ``"<name>_0"``, ``"<name>_1"``,
        …, ``"<name>_{n-1}"``.  For a dict *arg* whose length equals *n*,
        names are ``"<name>_<key>"`` for each key.

        :param n: Number of names to generate (must equal ``len(arg)``).
        :param name: Base name (typically the parameter name from the function
            signature).
        :param arg: The original argument (list, tuple, or dict).
        :param treespec: The :class:`torch.utils._pytree.TreeSpec` of *arg*;
            included for error messages only.
        :return: A ``list`` of ``str`` names of length *n*.
        :raises NotImplementedError: If *arg* has an unsupported type.
        """
        if isinstance(arg, (list, tuple)):
            return [f"{name}_{i}" for i in range(n)]
        if isinstance(arg, dict) and len(arg) == n:
            return [f"{name}_{k}" for k in arg]
        raise NotImplementedError(
            f"make_names is not implemented for type {type(arg)}, n={n}, {treespec=}"
        )

    def make_tracing_arg(self, arg, dynamic_shapes, name: Union[int, str]):
        if isinstance(name, int):
            name = f"arg_{name}"
        if isinstance(arg, torch.Tensor):
            tt = self.placeholder(
                name,
                TracingShape.from_existing_shape(arg.shape, dynamic_shapes),
                arg.dtype,
                device=arg.device,
            )
            return tt

        new_arg, treespec_arg = torch.utils._pytree.tree_flatten(arg)
        if dynamic_shapes:
            flat_ds, _ = torch.utils._pytree.tree_flatten(
                dynamic_shapes,
                is_leaf=lambda x: isinstance(x, dict) and all(isinstance(k, int) for k in x),
            )
            assert len(flat_ds) == len(new_arg), (
                f"Length mismatch between arg ({len(new_arg)}) and "
                f"the dynamic_shapes ({len(flat_ds)}), name={name!r}"
            )
        else:
            flat_ds = [None for _ in new_arg]
        names = self.make_names(len(new_arg), name, arg, treespec_arg)
        flat_args = [
            self.placeholder(
                nom, TracingShape.from_existing_shape(a.shape, ds), a.dtype, device=a.device
            )
            for nom, a, ds in zip(names, new_arg, flat_ds)
        ]
        return torch.utils._pytree.tree_unflatten(flat_args, treespec_arg)

    def make_tracing_args(
        self,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
        dynamic_shapes: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None,
        sig_names: Optional[List[str]] = None,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Convert *args* / *kwargs* into tracing counterparts.

        Every :class:`torch.Tensor` (or container of tensors) in *args* and
        *kwargs* is replaced by the corresponding :class:`TracingTensor`
        placeholder(s).  Non-tensor values are forwarded unchanged.

        :param args: Positional arguments as provided to :meth:`trace`.
        :param kwargs: Keyword arguments as provided to :meth:`trace`.
        :param dynamic_shapes: Optional per-argument dynamic shape mapping;
            passed through to :meth:`make_tracing_arg`.
        :param sig_names: Parameter names extracted from the traced function's
            signature; used to name placeholders and look up *dynamic_shapes*
            by name rather than by index.
        :return: A ``(tracing_args, tracing_kwargs)`` tuple whose tensor leaves
            are :class:`TracingTensor` instances.
        """
        assert sig_names, "sig_names cannot be empty."
        tracing_args = []
        tracing_kwargs = {}
        if args:
            for i, a in enumerate(args):
                if self.is_not_tensor(a):
                    tracing_args.append(a)
                    continue
                arg_name = sig_names[i] if i < len(sig_names) else f"vararg_{i - len(sig_names)}"
                ds = (
                    (
                        dynamic_shapes[i]
                        if isinstance(dynamic_shapes, tuple)
                        else dynamic_shapes.get(arg_name, None)
                    )
                    if dynamic_shapes
                    else None
                )
                x = self.make_tracing_arg(a, ds, name=arg_name)
                tracing_args.append(x)
        if kwargs:
            for k, v in kwargs.items():
                if self.is_not_tensor(v):
                    tracing_kwargs[k] = v
                    continue
                assert dynamic_shapes is None or isinstance(dynamic_shapes, dict), (
                    f"Unexpected type {type(dynamic_shapes)} for dynamic shapes, "
                    f"dict is mandatory when kwargs is not empty."
                )
                ds = dynamic_shapes[k] if dynamic_shapes else None
                tracing_kwargs[k] = self.make_tracing_arg(v, ds, name=k)
        return tuple(tracing_args), tracing_kwargs

    def trace(
        self,
        func: Callable,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
        dynamic_shapes: Optional[Dict[str, Any]] = None,
    ) -> torch.fx.Graph:
        """
        Trace *func* with the provided *args* and return the resulting
        :class:`torch.fx.Graph`.

        Tensor arguments are replaced by :class:`TracingTensor` placeholders
        and every dispatched ATen operation is recorded as a graph node.
        Non-tensor arguments are forwarded as-is.

        :param func: The callable to trace (e.g. an :class:`torch.nn.Module`
            instance or a plain Python function).
        :param args: Positional arguments to *func*.  Real
            :class:`torch.Tensor` values should be supplied; their shapes and
            dtypes are used for placeholder metadata.
        :param kwargs: Optional keyword arguments to *func*.
        :param dynamic_shapes: Optional mapping from argument *name*
            (parameter name from *func*'s signature for positional args;
            key name for keyword args) to a list/tuple of
            :class:`TracingInt` / int describing the dimensions symbolically.
            When provided, the corresponding placeholder is given a
            :class:`TracingShape` instead of a concrete :class:`torch.Size`.
        :return: A :class:`torch.fx.Graph` representing the full computation.

        .. runpython::
            :showcode:
            :process:

            import torch
            from yobx.torch.new_tracing.tracer import GraphTracer

            def add(x, y):
                return x + y

            graph = GraphTracer().trace(add, (torch.randn(3, 4), torch.randn(3, 4)))
            print(graph)
        """
        from ._patches import _trace_replacement_ctx

        if self.verbose:
            s = str(func).split("\n")[0]
            print(f"[GraphTracer.trace] trace {s}")

        if isinstance(func, torch.nn.Module):
            self.register_module_parameters(func)

        # Temporarily replace plain tensor module attributes (not parameters or
        # buffers) with TracingTensors so that operations like ``self.params.clone()``
        # dispatch through ``__torch_dispatch__`` and produce proper FX nodes.
        _replaced_attrs: List[Tuple[torch.nn.Module, str, torch.Tensor]] = []
        if isinstance(func, torch.nn.Module):
            _replaced_attrs = self._collect_and_replace_module_tensor_attrs(func)

        # Patch the ``forward`` of every top-level leaf sub-module so that
        # calling the sub-module during tracing emits a single
        # ``call_function`` node rather than tracing into its internals.
        # Only *top-level* leaves are patched: sub-modules that are nested
        # inside another leaf are skipped because the parent's wrapper
        # short-circuits before their code would run.
        _patched: List[Tuple[torch.nn.Module, Callable]] = []
        if isinstance(func, torch.nn.Module) and self.module_leaves:
            _patched_names: Set[str] = set()
            for subname, submod in func.named_modules():
                if not subname:
                    continue  # skip the root module
                # Skip if an ancestor module is already patched as a leaf.
                parts = subname.split(".")
                if any(".".join(parts[:i]) in _patched_names for i in range(1, len(parts))):
                    continue
                if self._is_leaf_module(submod, subname):
                    _patched.append((submod, submod.forward))
                    submod.forward = self._make_leaf_forward(submod, subname)
                    _patched_names.add(subname)
                    if self.verbose:
                        print(f"[GraphTracer.trace] patched leaf sub-module {subname!r}")

        sig_params = [
            p.name
            for p in inspect.signature(
                func.forward if hasattr(func, "forward") else func
            ).parameters.values()
            if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]

        tracing_args, tracing_kwargs = self.make_tracing_args(
            args, kwargs, dynamic_shapes=dynamic_shapes, sig_names=sig_params
        )

        if self.verbose:
            print("[GraphTracer.trace] call...")
        clear_conditions()
        with _trace_replacement_ctx(self):
            out = func(*tracing_args, **tracing_kwargs)

        for submod, orig_fwd in _patched:
            submod.forward = orig_fwd

        # Restore plain tensor module attributes replaced before forward.
        for submod, attr_name, orig_tensor in _replaced_attrs:
            object.__setattr__(submod, attr_name, orig_tensor)

        # Expose any registered callables (e.g. scan body functions) as
        # attributes on the traced module so that the downstream interpreter's
        # ``get_attr`` handler can retrieve them.
        if isinstance(func, torch.nn.Module) and self._callables:
            for k, v in self._callables.items():
                setattr(func, k, v)

        def _to_output_node(x: Any) -> Any:
            if isinstance(x, TracingTensor):
                return self._get_node(x)
            assert not isinstance(x, torch.Tensor), (
                f"Function {func} returned a real torch.Tensor. "
                "All tensor outputs must be TracingTensor instances produced during tracing."
            )
            return x

        import torch.utils._pytree as pytree

        output_val = pytree.tree_map(_to_output_node, out)
        self.graph.output(output_val)
        self.graph.eliminate_dead_code()
        # Remove placeholder nodes introduced by _collect_and_replace_module_tensor_attrs
        # that were never used in the graph (e.g. when the attribute was only accessed
        # for its .shape property, not dispatched as a tensor argument).
        if _replaced_attrs:
            for _, _, orig_tensor in _replaced_attrs:
                key = id(orig_tensor)
                node = self._external_tensor_to_node.get(key)
                if node is not None and node.op == "placeholder" and len(node.users) == 0:
                    self.graph.erase_node(node)
                    del self._external_tensor_to_node[key]
        return self.graph


def trace_model(
    func: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
    module_leaves: Optional[Dict[type, Callable[..., bool]]] = None,
) -> torch.fx.Graph:
    """
    Convenience wrapper: create a :class:`DispatchTracer` and trace *func*.

    :param func: Callable to trace.
    :param args: Positional tensor arguments (real tensors; shapes/dtypes
        are used for placeholder metadata).
    :param kwargs: Optional keyword tensor arguments.
    :param dynamic_shapes: Optional dynamic shape specifications; see
        :meth:`DispatchTracer.trace` for the format.
    :param verbose: verbosity level
    :param module_leaves: Optional mapping from module *type* to a predicate
        ``f(module, module_qualified_name=name) -> bool``.  Modules whose type
        appears in this mapping and whose predicate returns ``True`` are treated
        as leaves: the tracer emits a single ``call_function`` node for the
        whole module call instead of tracing through its internals.  See
        :class:`GraphTracer` for details.
    :return: A :class:`torch.fx.Graph` representing the computation.

    .. runpython::
        :showcode:
        :process:

        import torch
        from yobx.torch.new_tracing import trace_model

        graph = trace_model(
            torch.nn.Linear(4, 4),
            (torch.randn(2, 4),),
        )
        print(graph)
    """
    return GraphTracer(verbose=verbose, module_leaves=module_leaves).trace(
        func, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes
    )
