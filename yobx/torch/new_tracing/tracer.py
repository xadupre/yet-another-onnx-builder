import inspect
import operator
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.fx
import torch.utils._pytree as pytree
from .shape import TracingShape
from .tensor import TracingTensor


class GraphTracer:
    """
    Traces a callable by intercepting all tensor operations via
    ``__torch_dispatch__`` and records them into a :class:`torch.fx.Graph`.

    .. runpython::
        :showcode:

        import torch
        from yobx.torch.new_tracing import GraphTracer

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

        import torch
        from yobx.torch.new_tracing import GraphTracer

        def add(x, y):
            return x + y

        x = torch.randn(3, 4)
        y = torch.randn(1, 4)
        tracer = GraphTracer()
        graph = tracer.trace(add, (x, y), ({0:"batch"}, {}))
        print(graph)

    """

    def __init__(self) -> None:
        self.graph: torch.fx.Graph = torch.fx.Graph()
        self._external_tensor_to_node: Dict[int, torch.fx.Node] = {}
        self._placeholder_count: int = 0

    def _make_tracing_tensor(
        self,
        shape: Union[Tuple[int, ...], "TracingShape"],
        dtype: torch.dtype,
        device: Union[str, torch.device],
        node: torch.fx.Node,
    ) -> "TracingTensor":
        """Create a :class:`TracingTensor` and register it with *node*."""
        t = TracingTensor.__new__(TracingTensor, shape, dtype=dtype, device=device)
        t.__init__(shape, dtype=dtype, device=device, tracer=self)  # type: ignore
        t._node = node
        return t

    def _get_node(self, t: "TracingTensor") -> torch.fx.Node:
        """Return the :class:`torch.fx.Node` associated with *t*."""
        assert isinstance(t, TracingTensor), f"Unexpected type {type(t)} for a tracing tensor."
        assert t._node, f"Tensor {t} has no node."
        assert t._tracer is self, "The tensor seems traced by another graph."
        return t._node

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

    def register_module_parameters(self, module: torch.nn.Module, verbose: int = 0) -> None:
        """
        Pre-register all named parameters and buffers of *module* as
        placeholder nodes in the graph.

        This gives each parameter a meaningful name in the graph (e.g.
        ``linear_weight`` instead of ``param_1``) and ensures that shared
        tensors (the same :class:`torch.Tensor` referenced under multiple
        names) map to exactly one placeholder node.

        :param module: The :class:`torch.nn.Module` whose parameters and
            buffers should be pre-registered.
        :param verbose: verbosity
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
            node.meta["torch_name"] = name
            if verbose:
                print(f"[GraphTracer.register_module_parameters] + {name!r}")
            self._external_tensor_to_node[key] = node

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
        return tt

    def wrap_output(
        self, meta_out: Any, node: torch.fx.Node, device: Union[str, torch.device]
    ) -> Any:
        """
        Wrap meta-tensor output(s) as :class:`TracingTensor` instances.

        * Single tensor → one :class:`TracingTensor`.
        * List/tuple of tensors → same container type with
          ``operator.getitem`` nodes interleaved.
        * Anything else (scalar, ``None``) → returned as-is.
        """
        if isinstance(meta_out, torch.Tensor):
            return self._make_tracing_tensor(meta_out.shape, meta_out.dtype, device, node)

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
                    results.append(
                        self._make_tracing_tensor(item.shape, item.dtype, device, get_node)
                    )
                else:
                    results.append(item)
            return type(meta_out)(results)

        return meta_out

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

        # --- 1. shape inference using meta tensors ---
        def _to_meta(x: Any) -> Any:
            assert isinstance(
                x, (TracingTensor, int, float, str)
            ), f"unexpected type {type(x)} for a tensor or a constant"
            return x

        meta_args = pytree.tree_map(_to_meta, args)
        meta_kwargs = pytree.tree_map(_to_meta, kwargs)

        with torch.no_grad():
            meta_out = op(*meta_args, **meta_kwargs)

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

        # --- 4. wrap output ---
        return self._wrap_output(meta_out, node, device)

    # ------------------------------------------------------------------
    # Public tracing entry point
    # ------------------------------------------------------------------

    def trace(
        self,
        func: Callable,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
        dynamic_shapes: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
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
        :param verbose: verbosity
        :return: A :class:`torch.fx.Graph` representing the full computation.
        """
        if isinstance(func, torch.nn.Module):
            self._register_module_parameters(func, verbose=verbose)

        new_args, treespec_args = torch.utils._pytree.tree_flatten(args)
        new_kwargs, treespec_kwargs = torch.utils._pytree.tree_flatten(kwargs or {})

        assert all(
            isinstance(t, (TracingTensor, int, float, str)) for t in new_args
        ), f"Unexpected type in args {[type(t) for t in new_args]}"
        assert all(
            isinstance(t, (TracingTensor, int, float, str)) for t in new_kwargs
        ), f"Unexpected type in args {[type(t) for t in new_kwargs]}"

        # Collect positional parameter names from func's signature.
        _sig_params = [
            p.name
            for p in inspect.signature(func).parameters.values()
            if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]

        def _arg_name(i: int) -> str:
            """Return the parameter name for positional argument *i*."""
            if i < len(_sig_params):
                return _sig_params[i]
            return f"x_{i}"

        tracing_args = []
        tracing_kwargs = {}
        if dynamic_shapes:
            # walk through along new_args, kwargs and dynamic_shapes to create TracingTensor
            pass
        else:
            # walk through along new_args, kwargs to create TracingTensor
            pass

        # Replace all torch.Tensor inputs with TracingTensor placeholders.
        # tracing_args = tuple(
        #    self._make_placeholder(arg, _arg_name(i), dynamic_shapes)
        #    for i, arg in enumerate(args)
        # )
        # tracing_kwargs = {
        #    k: self._make_placeholder(v, k, dynamic_shapes) for k, v in kwargs.items()
        # }

        if verbose:
            print(f"[GraphTracer.trace] trace {func}")
        out = func(*tracing_args, **tracing_kwargs)

        def _to_output_node(x: Any) -> Any:
            if isinstance(x, TracingTensor):
                return self._get_node(x)
            if isinstance(x, torch.Tensor):
                raise RuntimeError(
                    f"Function returned a real torch.Tensor: {x!r}. "
                    "All tensor outputs must be TracingTensor instances produced during tracing."
                )
            return x

        output_val = pytree.tree_map(_to_output_node, out)
        self.graph.output(output_val)

        return self.graph


def trace_model(
    func: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
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
    return GraphTracer().trace(func, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes)
