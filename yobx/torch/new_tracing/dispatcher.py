"""
Dispatcher-related classes and helpers for dispatch-level tracing.

Defines :class:`DispatchTracer` and the convenience function
:func:`trace_model`.
"""

import operator
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.fx
import torch.utils._pytree as pytree

from .shape import TracingInt, TracingShape
from .tensor import TracingTensor


class DispatchTracer:
    """
    Traces a callable by intercepting all tensor operations via
    ``__torch_dispatch__`` and records them into a :class:`torch.fx.Graph`.

    Example::

        import torch
        from yobx.torch.new_tracing import DispatchTracer

        def add(x, y):
            return x + y

        tracer = DispatchTracer()
        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        graph = tracer.trace(add, (x, y))
        print(graph)

    The class creates an empty :class:`torch.fx.Graph`
    populated by a :meth:`trace` call.
    """

    def __init__(self) -> None:
        self.graph: torch.fx.Graph = torch.fx.Graph()
        # id(TracingTensor) -> torch.fx.Node
        self._tensor_id_to_node: Dict[int, torch.fx.Node] = {}
        # id(regular torch.Tensor) -> torch.fx.Node  (auto-placeholders)
        self._external_tensor_to_node: Dict[int, torch.fx.Node] = {}
        # Counter for auto-generated placeholder names
        self._placeholder_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_tracing_tensor(
        self,
        shape: Union[Tuple[int, ...], "TracingShape"],
        dtype: torch.dtype,
        device: Union[str, torch.device],
        node: torch.fx.Node,
        tracing_shape: Optional["TracingShape"] = None,
    ) -> "TracingTensor":
        """Create a :class:`TracingTensor` and register it with *node*.

        :param tracing_shape: Optional symbolic :class:`TracingShape` to attach
            to the tensor.  When *None* (the default), ``shape`` itself is used
            if it is already a :class:`TracingShape`.
        """
        t = TracingTensor.__new__(TracingTensor, shape, dtype=dtype, device=device)
        t.__init__(shape, dtype=dtype, device=device)  # type: ignore
        t._tracer = self
        t._node = node
        self._tensor_id_to_node[id(t)] = node
        # Attach symbolic shape if available.
        sym = tracing_shape if tracing_shape is not None else (
            shape if isinstance(shape, TracingShape) else None
        )
        if sym is not None:
            t._tracing_shape = sym
            node.meta["tensor_shape"] = sym
        return t

    def _get_node(self, t: "TracingTensor") -> torch.fx.Node:
        """Return the :class:`torch.fx.Node` associated with *t*."""
        node = self._tensor_id_to_node.get(id(t))
        assert node is not None, (
            f"TracingTensor {t!r} is not registered with this tracer. "
            "Ensure all tensors were created (or passed to) this DispatchTracer."
        )
        return node

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
            node.meta["val"] = t.detach().to(device="meta")
            self._external_tensor_to_node[key] = node
        return self._external_tensor_to_node[key]

    def _infer_output_tracing_shape(
        self,
        meta_out: torch.Tensor,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Optional["TracingShape"]:
        """
        Attempt to derive a symbolic :class:`TracingShape` for an output tensor
        by matching each output dimension position against the corresponding
        position in the :class:`TracingShape` of every :class:`TracingTensor`
        input that has the same number of dimensions.

        A symbolic :class:`TracingInt` is propagated to an output dimension
        when *all* input :class:`TracingShape` objects with the same rank agree
        on the same symbolic name at that position (or have only concrete values
        there).  This handles element-wise operations correctly.

        :param meta_out: The meta output tensor produced during shape inference.
        :param args: Positional arguments to the dispatched operation.
        :param kwargs: Keyword arguments to the dispatched operation.
        :return: A :class:`TracingShape` when at least one dimension is
            symbolic, or ``None`` when no symbolic information is available.
        """
        n = len(meta_out.shape)
        # candidates[i] = TracingInt if all inputs agree on a symbolic dim at
        # position i, or None if there is a conflict.
        candidates: Dict[int, Optional[TracingInt]] = {}

        for leaf in pytree.tree_leaves((*args, *kwargs.values())):
            if not isinstance(leaf, TracingTensor):
                continue
            sym = leaf._tracing_shape
            if sym is None or len(sym) != n:
                continue
            for i, d in enumerate(sym):
                if isinstance(d, TracingInt) and d.is_symbolic:
                    # Symbolic dimension at position i (d.value is a str).
                    # candidates[i] is also always a symbolic TracingInt (str value)
                    # when set, so value comparison is always str vs str here.
                    if i not in candidates:
                        candidates[i] = d
                    elif candidates[i] is not None and candidates[i].value != d.value:
                        # Conflict: two inputs disagree on the symbolic name.
                        candidates[i] = None

        sym_dims = {i: d for i, d in candidates.items() if d is not None}
        if not sym_dims:
            return None

        dims: List[Union[TracingInt, int]] = []
        for i, s in enumerate(meta_out.shape):
            if i in sym_dims:
                dims.append(sym_dims[i])
            else:
                dims.append(int(s))
        return TracingShape(dims)

    def _register_module_parameters(self, module: torch.nn.Module) -> None:
        """
        Pre-register all named parameters and buffers of *module* as
        placeholder nodes in the graph.

        This gives each parameter a meaningful name in the graph (e.g.
        ``linear_weight`` instead of ``param_1``) and ensures that shared
        tensors (the same :class:`torch.Tensor` referenced under multiple
        names) map to exactly one placeholder node.

        :param module: The :class:`torch.nn.Module` whose parameters and
            buffers should be pre-registered.
        """
        seen_ids: set = set()
        for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
            if tensor is None:
                continue
            key = id(tensor)
            if key in seen_ids or key in self._external_tensor_to_node:
                continue
            seen_ids.add(key)
            # Replace "." with "_" so the FX node name is a valid identifier.
            sanitized = name.replace(".", "_")
            node = self.graph.placeholder(sanitized)
            node.meta["val"] = tensor.detach().to(device="meta")
            self._external_tensor_to_node[key] = node

    # ------------------------------------------------------------------
    # Graph construction helpers
    # ------------------------------------------------------------------

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
        if isinstance(shape, TracingShape):
            concrete = (
                shape.to_torch_size()
                if shape.is_concrete
                else torch.Size(
                    (
                        d.value
                        if isinstance(d, TracingInt) and isinstance(d.value, int)
                        else int(d) if isinstance(d, int) else 1
                    )
                    for d in shape
                )
            )
        else:
            concrete = torch.Size(shape)
        node.meta["val"] = torch.empty(concrete, dtype=dtype, device="meta")
        return self._make_tracing_tensor(shape, dtype, device, node)

    # ------------------------------------------------------------------
    # Dispatch handler
    # ------------------------------------------------------------------

    def _dispatch(self, op: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Handle one dispatched operation:

        1. Run the op on *meta* tensors for shape/dtype inference.
        2. Build FX node args (replacing :class:`TracingTensor` with their
           nodes, and non-traced tensors with auto-placeholder nodes).
        3. Emit a ``call_function`` node in the graph.
        4. Return wrapped :class:`TracingTensor` output(s).
        """
        # Determine the device from the first TracingTensor arg (used for
        # the wrapper subclass device attribute of output tensors).
        device: Union[str, torch.device] = "cpu"
        all_inputs = (*args, *kwargs.values())
        for leaf in pytree.tree_leaves(all_inputs):
            if isinstance(leaf, TracingTensor):
                device = leaf.device
                break

        # --- 1. shape inference using meta tensors ---
        def _to_meta(x: Any) -> Any:
            if isinstance(x, TracingTensor):
                node = self._tensor_id_to_node.get(id(x))
                if node is not None and "val" in node.meta:
                    return node.meta["val"]
                return torch.empty(x.shape, dtype=x.dtype, device="meta")
            if isinstance(x, torch.Tensor):
                return x.detach().to(device="meta")
            return x

        meta_args = pytree.tree_map(_to_meta, args)
        meta_kwargs = pytree.tree_map(_to_meta, kwargs)

        with torch.no_grad():
            try:
                meta_out = op(*meta_args, **meta_kwargs)
            except Exception:
                meta_out = None

        # --- 2. build FX node args ---
        def _to_node(x: Any) -> Any:
            if isinstance(x, TracingTensor):
                return self._get_node(x)
            if isinstance(x, torch.Tensor):
                # Non-traced tensor: auto-create a placeholder node.
                return self._node_for_external_tensor(x)
            return x

        fx_args = pytree.tree_map(_to_node, args)
        fx_kwargs = pytree.tree_map(_to_node, kwargs)

        # --- 3. emit FX node ---
        node = self.graph.call_function(op, args=fx_args, kwargs=fx_kwargs)
        node.meta["stack_trace"] = "".join(traceback.format_stack())
        if isinstance(meta_out, torch.Tensor):
            node.meta["val"] = meta_out

        # --- 4. infer symbolic output shape and wrap output ---
        out_tracing_shape: Optional[TracingShape] = None
        if isinstance(meta_out, torch.Tensor):
            out_tracing_shape = self._infer_output_tracing_shape(meta_out, args, kwargs)
        return self._wrap_output(meta_out, node, device, out_tracing_shape, args, kwargs)

    def _wrap_output(
        self,
        meta_out: Any,
        node: torch.fx.Node,
        device: Union[str, torch.device],
        tracing_shape: Optional["TracingShape"] = None,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Wrap meta-tensor output(s) as :class:`TracingTensor` instances.

        * Single tensor → one :class:`TracingTensor`.
        * List/tuple of tensors → same container type with
          ``operator.getitem`` nodes interleaved.
        * Anything else (scalar, ``None``) → returned as-is.

        :param tracing_shape: Pre-computed symbolic shape for a single-tensor
            output.  Ignored for list/tuple outputs (shape is inferred
            per-element when *args* and *kwargs* are provided).
        :param args: Original op args forwarded for per-element shape inference
            in multi-output operations.
        :param kwargs: Original op kwargs forwarded for per-element shape
            inference in multi-output operations.
        """
        if isinstance(meta_out, torch.Tensor):
            return self._make_tracing_tensor(
                meta_out.shape, meta_out.dtype, device, node, tracing_shape
            )

        if isinstance(meta_out, (list, tuple)):
            results: List[Any] = []
            for i, item in enumerate(meta_out):
                if isinstance(item, torch.Tensor):
                    get_node = self.graph.call_function(
                        operator.getitem, args=(node, i), kwargs={}
                    )
                    get_node.meta["val"] = item
                    if "stack_trace" in node.meta:
                        get_node.meta["stack_trace"] = node.meta["stack_trace"]
                    # Infer symbolic shape for this element individually.
                    elem_tracing_shape: Optional[TracingShape] = None
                    if args is not None and kwargs is not None:
                        elem_tracing_shape = self._infer_output_tracing_shape(
                            item, args, kwargs
                        )
                    results.append(
                        self._make_tracing_tensor(
                            item.shape, item.dtype, device, get_node, elem_tracing_shape
                        )
                    )
                else:
                    results.append(item)
            return type(meta_out)(results)

        return meta_out

    # ------------------------------------------------------------------
    # Public tracing entry point
    # ------------------------------------------------------------------

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
            (``"x_0"``, ``"x_1"``, … for positional args; key name for
            keyword args) to a list/tuple of :class:`TracingInt` / int
            describing the dimensions symbolically.  When provided, the
            corresponding placeholder is given a :class:`TracingShape` instead
            of a concrete :class:`torch.Size`.
        :return: A :class:`torch.fx.Graph` representing the full computation.

        Example::

            import torch
            from yobx.torch.new_tracing import DispatchTracer, TracingInt

            def linear(x, w, b):
                return x @ w.t() + b

            tracer = DispatchTracer()
            graph = tracer.trace(
                linear,
                (torch.randn(4, 8), torch.randn(16, 8), torch.randn(16)),
                dynamic_shapes={"x_0": [TracingInt("batch"), 8]},
            )
            print(graph)
        """
        if kwargs is None:
            kwargs = {}
        dynamic_shapes = dynamic_shapes or {}

        # Reset state for a fresh trace.
        self.graph = torch.fx.Graph()
        self._tensor_id_to_node = {}
        self._external_tensor_to_node = {}
        self._placeholder_count = 0

        # ------------------------------------------------------------------
        # Pre-register nn.Module parameters and buffers as named placeholders.
        # This must happen before building input placeholders so that the
        # parameter placeholder nodes appear first in the graph.
        # ------------------------------------------------------------------
        if isinstance(func, torch.nn.Module):
            self._register_module_parameters(func)

        # ------------------------------------------------------------------
        # Build placeholder TracingTensors for each tensor input.
        # Nested structures (list, tuple, dict) are traversed recursively so
        # that every tensor leaf gets its own placeholder node.
        # ------------------------------------------------------------------
        def _make_placeholder(arg: Any, name: str) -> Any:
            """Recursively replace tensors in a nested structure with placeholders."""
            if isinstance(arg, torch.Tensor):
                if name in dynamic_shapes:
                    shape: Union[Tuple[int, ...], TracingShape] = TracingShape(
                        dynamic_shapes[name]
                    )
                else:
                    shape = arg.shape
                return self.placeholder(name, shape, arg.dtype, arg.device)
            if isinstance(arg, dict):
                return {k: _make_placeholder(v, f"{name}_{k}") for k, v in arg.items()}
            if isinstance(arg, (list, tuple)):
                items = [_make_placeholder(item, f"{name}_{j}") for j, item in enumerate(arg)]
                return type(arg)(items)
            # Non-tensor scalars / non-container objects pass through unchanged.
            return arg

        tracing_args = tuple(_make_placeholder(arg, f"x_{i}") for i, arg in enumerate(args))
        tracing_kwargs = {k: _make_placeholder(v, k) for k, v in kwargs.items()}

        # ------------------------------------------------------------------
        # Execute the function under tracing.
        # ------------------------------------------------------------------
        out = func(*tracing_args, **tracing_kwargs)

        # ------------------------------------------------------------------
        # Emit output node.
        # ------------------------------------------------------------------
        def _to_output_node(x: Any) -> Any:
            if isinstance(x, TracingTensor):
                return self._get_node(x)
            return x

        output_val = pytree.tree_map(_to_output_node, out)
        self.graph.output(output_val)

        return self.graph


def trace_model(
    func: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Dict[str, Any]] = None,
) -> torch.fx.Graph:
    """
    Convenience wrapper: create a :class:`DispatchTracer` and trace *func*.

    :param func: Callable to trace.
    :param args: Positional tensor arguments (real tensors; shapes/dtypes
        are used for placeholder metadata).
    :param kwargs: Optional keyword tensor arguments.
    :param dynamic_shapes: Optional dynamic shape specifications; see
        :meth:`DispatchTracer.trace` for the format.
    :return: A :class:`torch.fx.Graph` representing the computation.

    Example::

        import torch
        from yobx.torch.new_tracing import trace_model

        graph = trace_model(
            torch.nn.Linear(4, 4),
            (torch.randn(2, 4),),
        )
        print(graph)
    """
    return DispatchTracer().trace(func, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes)
