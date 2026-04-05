import inspect
import operator
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.experimental.sym_node import SymNode
from ...xexpressions import rename_expression
from .shape import TracingShape
from .tensor import TracingTensor


class GraphTracer:
    """
    Traces a callable by intercepting all tensor operations via
    ``__torch_dispatch__`` and records them into a :class:`torch.fx.Graph`.

    .. runpython::
        :showcode:

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

        import torch
        from yobx.torch.new_tracing.tracer import GraphTracer

        def add(x, y):
            return x + y

        x = torch.randn(3, 4)
        y = torch.randn(1, 4)
        tracer = GraphTracer()
        graph = tracer.trace(add, (x, y), {}, ({0:"batch"}, {}))
        print(graph)

    """

    def __init__(self, verbose: int = 0) -> None:
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

        Scalars (``int``, ``float``, ``str``) and empty collections are
        treated as non-tensor.  Lists and tuples are inspected recursively.
        Dicts are inspected by their values.

        :param value: The value to inspect.  May be a scalar, tensor,
            list, tuple, or dict.
        :return: ``True`` when *value* has no tensor leaves; ``False`` when
            any leaf is a :class:`torch.Tensor` (including
            :class:`TracingTensor`).
        :raises TypeError: If *value* has a type that cannot be classified.
        """
        if isinstance(value, (int, float, str)):
            return True
        if isinstance(value, torch.Tensor):
            return False
        if isinstance(value, (list, tuple)):
            if not value:
                return True
            return all(self.is_not_tensor(v) for v in value)
        if isinstance(value, dict):
            if not value:
                return True
            return all(self.is_not_tensor(v) for v in value.values())

        from ...helpers import string_type

        raise TypeError(f"Cannot determine if it is a constant argument {string_type(value)}")

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

    def register_module_parameters(self, module: torch.nn.Module) -> None:
        """
        Pre-register all named parameters and buffers of *module* as
        placeholder nodes in the graph.

        This gives each parameter a meaningful name in the graph (e.g.
        ``linear_weight`` instead of ``param_1``) and ensures that shared
        tensors (the same :class:`torch.Tensor` referenced under multiple
        names) map to exactly one placeholder node.

        Each placeholder node receives two extra metadata entries:

        * ``node.meta["torch_name"]``: the original dotted parameter name as
          returned by :meth:`torch.nn.Module.named_parameters` (e.g.
          ``"linear.weight"``).
        * ``node.meta["torch_value"]``: the actual :class:`torch.Tensor`
          object (useful for retrieving concrete weight values later).

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
            node.meta["val"] = TracingTensor.from_tensor(tensor)
            node.meta["torch_name"] = name
            node.meta["torch_value"] = tensor
            if self.verbose:
                print(f"[GraphTracer.register_module_parameters] + {name!r}")
            self._external_tensor_to_node[key] = node

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
                if isinstance(d, str):
                    if d in self._mapped_dimension:
                        symd = self._mapped_dimension[d]
                    else:
                        symd = self._shape_env.create_unbacked_symint()
                        self._mapped_dimension[d] = symd
                        symd_name = self._sym_int_to_str(symd)
                        assert isinstance(symd_name, str), "type checking"
                        self._sym_int_to_dynamic_dimension[symd_name] = d
                    new_shape.append(symd)
                else:
                    assert isinstance(d, int), f"Unexpected type for a dimension {type(d)}"
                    new_shape.append(d)
            with self._fake_mode:
                return torch.empty(tuple(new_shape), dtype=a.dtype, device=a.device)
        with self._fake_mode:
            return torch.empty(a.shape, dtype=a.dtype, device=a.device)

    def _sym_shape_to_str_shape(self, sym_shape):
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
        new_shape = []
        for s in sym_shape:
            if isinstance(s, (int, str)):
                new_shape.append(s)
                continue
            ss = self._sym_int_to_str(s)
            ns = self._token_replace(ss)
            assert isinstance(ns, str), "simple type checking"
            self._mapped_dimension[ns] = s
            new_shape.append(ns)
        return tuple(new_shape)

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
            (a if self.is_not_tensor(a) else self.make_fake(a)) for a in combined_args
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

        # running the function
        with self._fake_mode:
            fake_res = func(*fake_args, **fake_kwargs)
        assert type(fake_res) is not torch.Tensor, f"Unexpected type {type(fake_res)} for output."

        # add a node in the graph
        node_combined = [
            (a if self.is_not_tensor(a) else self._get_node(a)) for a in combined_args
        ]
        unflat_nodes = torch.utils._pytree.tree_unflatten(node_combined, treespec)
        node_args, node_kwargs = unflat_nodes
        # We need to add nodes.
        node = self.graph.call_function(func, args=node_args, kwargs=node_kwargs)

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
            node.meta["fn"] = node.target

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
            flat_ds, _ = torch.utils._pytree.tree_flatten(dynamic_shapes)
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
                ds = (
                    (
                        dynamic_shapes[i]
                        if isinstance(dynamic_shapes, tuple)
                        else dynamic_shapes[sig_names[i]]
                    )
                    if dynamic_shapes
                    else None
                )
                x = self.make_tracing_arg(a, ds, name=sig_names[i] if sig_names else i)
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

            import torch
            from yobx.torch.new_tracing.tracer import GraphTracer

            def add(x, y):
                return x + y

            graph = GraphTracer().trace(add, (torch.randn(3, 4), torch.randn(3, 4)))
            print(graph)
        """
        if self.verbose:
            s = str(func).split("\n")[0]
            print(f"[GraphTracer.trace] trace {s}")

        if isinstance(func, torch.nn.Module):
            self.register_module_parameters(func)

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
        out = func(*tracing_args, **tracing_kwargs)

        def _to_output_node(x: Any) -> Any:
            if isinstance(x, TracingTensor):
                return self._get_node(x)
            assert not isinstance(x, torch.Tensor), (
                f"Function {func} returned a real torch.Tensor. "
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

    .. runpython::
        :showcode:

        import torch
        from yobx.torch.new_tracing import trace_model

        graph = trace_model(
            torch.nn.Linear(4, 4),
            (torch.randn(2, 4),),
        )
        print(graph)
    """
    return GraphTracer(verbose=verbose).trace(
        func, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes
    )
