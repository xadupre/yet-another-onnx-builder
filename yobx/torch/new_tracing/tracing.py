"""
New tracing mechanism for PyTorch models using ``__torch_dispatch__``.

This module provides a dispatch-level tracing mechanism that produces a
:class:`torch.fx.Graph` by intercepting all tensor operations at the
C++ dispatcher level, similar to how :class:`torch._subclasses.FakeTensor`
works.

Key classes:

- :class:`TracingInt`: A symbolic or concrete integer dimension.
- :class:`TracingBool`: A boolean that may be symbolic when it depends on
  dynamic dimensions.
- :class:`TracingShape`: A container of :class:`TracingInt` / :class:`int`
  dimension values.
- :class:`TracingTensor`: A :class:`torch.Tensor` subclass that records
  all operations into a :class:`torch.fx.Graph` via ``__torch_dispatch__``.
- :class:`DispatchTracer`: Manages tracing context and builds the graph.

Example::

    import torch
    from yobx.torch.new_tracing import DispatchTracer

    class MyModel(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    model = MyModel()
    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    tracer = DispatchTracer()
    graph = tracer.trace(model, (x, y))
    print(graph)
"""

import operator
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.fx
import torch.utils._pytree as pytree


class TracingBool:
    """
    Represents a boolean comparison that may involve symbolic :class:`TracingInt`
    dimensions and therefore cannot always be evaluated at trace time.

    :param value: Either a concrete :class:`bool` or a :class:`str` containing
        a symbolic expression such as ``"(batch==4)"``.

    - ``TracingBool(True)`` / ``TracingBool(False)`` — concrete result.
    - ``TracingBool("(n==4)")`` — symbolic; cannot be used as a Python bool.
    """

    def __init__(self, value: Union[bool, str]):
        self.value = value

    def __repr__(self) -> str:
        return f"TracingBool({self.value!r})"

    def __str__(self) -> str:
        return str(self.value)

    def __bool__(self) -> bool:
        if isinstance(self.value, bool):
            return self.value
        raise ValueError(
            f"TracingBool({self.value!r}) cannot be converted to a Python bool; "
            "the result depends on a symbolic/dynamic dimension"
        )

    def __eq__(self, other: Any) -> bool:  # noqa: PYI032
        if isinstance(other, TracingBool):
            return self.value == other.value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)


class TracingInt:
    """
    Represents a single symbolic or concrete integer (e.g. a tensor dimension).

    :param value: Either a concrete :class:`int` (concrete dimension) or a
        :class:`str` (symbolic/dynamic dimension name such as ``"batch"``).

    Examples::

        # Concrete dimension
        d = TracingInt(4)
        print(int(d))       # 4
        print(d + 2)        # TracingInt(6)

        # Symbolic/dynamic dimension
        s = TracingInt("batch")
        print(str(s))       # batch
        print(s + 2)        # TracingInt('(batch+2)')
        print(s == 4)       # TracingBool('(batch==4)')
    """

    def __init__(self, value: Union[int, str]):
        self.value = value

    def __repr__(self) -> str:
        return f"TracingInt({self.value!r})"

    def __str__(self) -> str:
        return str(self.value)

    def __int__(self) -> int:
        if isinstance(self.value, int):
            return self.value
        raise ValueError(
            f"TracingInt({self.value!r}) has no concrete integer value; "
            "pass a concrete int or check .value"
        )

    def __index__(self) -> int:
        """Enable use as a sequence index (calls :meth:`__int__`)."""
        return int(self)

    def __eq__(self, other: Any) -> Union[bool, "TracingBool"]:  # type: ignore # noqa: PYI032
        """
        Return a plain :class:`bool` when both sides are concrete; return a
        :class:`TracingBool` when at least one side is symbolic.
        """
        if isinstance(other, TracingInt):
            if isinstance(self.value, int) and isinstance(other.value, int):
                return self.value == other.value
            return TracingBool(f"({self.value}=={other.value})")
        if isinstance(other, int):
            if isinstance(self.value, int):
                return self.value == other
            return TracingBool(f"({self.value}=={other})")
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)

    # ------------------------------------------------------------------
    # Arithmetic helpers so TracingInt can participate in shape
    # calculations while keeping symbolic expressions intact.
    # ------------------------------------------------------------------

    def _sym(self) -> str:
        """String representation of *self* for building expressions."""
        return str(self.value)

    def __add__(self, other: Union[int, "TracingInt"]) -> "TracingInt":
        if isinstance(other, TracingInt):
            if isinstance(self.value, int) and isinstance(other.value, int):
                return TracingInt(self.value + other.value)
            return TracingInt(f"({self._sym()}+{other._sym()})")
        if isinstance(other, int):
            if isinstance(self.value, int):
                return TracingInt(self.value + other)
            return TracingInt(f"({self._sym()}+{other})")
        return NotImplemented

    def __radd__(self, other: int) -> "TracingInt":
        if isinstance(other, int):
            if isinstance(self.value, int):
                return TracingInt(other + self.value)
            return TracingInt(f"({other}+{self._sym()})")
        return NotImplemented

    def __sub__(self, other: Union[int, "TracingInt"]) -> "TracingInt":
        if isinstance(other, TracingInt):
            if isinstance(self.value, int) and isinstance(other.value, int):
                return TracingInt(self.value - other.value)
            return TracingInt(f"({self._sym()}-{other._sym()})")
        if isinstance(other, int):
            if isinstance(self.value, int):
                return TracingInt(self.value - other)
            return TracingInt(f"({self._sym()}-{other})")
        return NotImplemented

    def __rsub__(self, other: int) -> "TracingInt":
        if isinstance(other, int):
            if isinstance(self.value, int):
                return TracingInt(other - self.value)
            return TracingInt(f"({other}-{self._sym()})")
        return NotImplemented

    def __mul__(self, other: Union[int, "TracingInt"]) -> "TracingInt":
        if isinstance(other, TracingInt):
            if isinstance(self.value, int) and isinstance(other.value, int):
                return TracingInt(self.value * other.value)
            return TracingInt(f"({self._sym()}*{other._sym()})")
        if isinstance(other, int):
            if isinstance(self.value, int):
                return TracingInt(self.value * other)
            return TracingInt(f"({self._sym()}*{other})")
        return NotImplemented

    def __rmul__(self, other: int) -> "TracingInt":
        if isinstance(other, int):
            if isinstance(self.value, int):
                return TracingInt(other * self.value)
            return TracingInt(f"({other}*{self._sym()})")
        return NotImplemented

    def __floordiv__(self, other: Union[int, "TracingInt"]) -> "TracingInt":
        if isinstance(other, TracingInt):
            if isinstance(self.value, int) and isinstance(other.value, int):
                return TracingInt(self.value // other.value)
            return TracingInt(f"({self._sym()}//{other._sym()})")
        if isinstance(other, int):
            if isinstance(self.value, int):
                return TracingInt(self.value // other)
            return TracingInt(f"({self._sym()}//{other})")
        return NotImplemented

    def __neg__(self) -> "TracingInt":
        if isinstance(self.value, int):
            return TracingInt(-self.value)
        return TracingInt(f"(-{self._sym()})")


# Keep TracingDimension as a backward-compatible alias.
TracingDimension = TracingInt


class TracingShape:
    """
    Represents the shape of a :class:`TracingTensor` as an ordered collection
    of :class:`TracingInt` or :class:`int` dimension values.

    Unlike :class:`torch.Size`, individual dimensions may be symbolic
    (:class:`TracingInt` with a ``str`` value).

    :param dims: Iterable of :class:`TracingInt` or :class:`int` values.

    Example::

        shape = TracingShape([TracingInt(4), 16])
        print(shape)              # TracingShape([TracingInt(4), 16])
        print(shape.numel())      # 64
        print(shape.is_concrete)  # True

        sym_shape = TracingShape([TracingInt("n"), 8])
        print(sym_shape.is_concrete)  # False
    """

    def __init__(self, dims: "Sequence[Union[TracingInt, int]]") -> None:
        self.dims: Tuple[Union["TracingInt", int], ...] = tuple(dims)

    def __repr__(self) -> str:
        return f"TracingShape({list(self.dims)!r})"

    def __str__(self) -> str:
        items = ", ".join(str(d) for d in self.dims)
        return f"TracingShape([{items}])"

    def __len__(self) -> int:
        return len(self.dims)

    def __iter__(self):
        return iter(self.dims)

    def __getitem__(self, idx):
        return self.dims[idx]

    def __eq__(self, other: Any) -> bool:  # noqa: PYI032
        if isinstance(other, TracingShape):
            return self.dims == other.dims
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.dims)

    @property
    def is_concrete(self) -> bool:
        """Return ``True`` if every dimension has a concrete integer value."""
        return all(
            isinstance(d, int) or (isinstance(d, TracingInt) and isinstance(d.value, int))
            for d in self.dims
        )

    def numel(self) -> int:
        """
        Return the total number of elements (product of all dimensions).

        :raises ValueError: If any dimension is purely symbolic (no concrete
            integer value).
        """
        result = 1
        for d in self.dims:
            if isinstance(d, TracingInt):
                if not isinstance(d.value, int):
                    raise ValueError(
                        f"Cannot compute numel: dimension {d.value!r} has no concrete value"
                    )
                result *= d.value
            else:
                result *= int(d)
        return result

    def to_torch_size(self) -> torch.Size:
        """
        Convert to :class:`torch.Size` (requires all dimensions to be concrete).

        :raises ValueError: If any dimension is purely symbolic.
        """
        if not self.is_concrete:
            raise ValueError(
                "Cannot convert TracingShape with purely symbolic dims to torch.Size; "
                "ensure all TracingInt objects have a concrete (int) value"
            )
        return torch.Size(tuple(int(d) for d in self.dims))


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
        :class:`DispatchTracer`.  Use :meth:`DispatchTracer.trace` to trace a
        callable rather than constructing :class:`TracingTensor` directly.

    Attributes:
        _tracer: The :class:`DispatchTracer` managing this tensor's graph.
        _node: The :class:`torch.fx.Node` corresponding to this tensor in the
            graph.
    """

    @staticmethod
    def __new__(  # noqa: PYI034
        cls,
        size: Union[Tuple[int, ...], "TracingShape"],
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
        requires_grad: bool = False,
    ) -> "TracingTensor":
        if isinstance(size, TracingShape):
            # Use concrete values where available; fall back to 1 for purely
            # symbolic dimensions so that _make_wrapper_subclass receives ints.
            sizes = tuple(
                (
                    d.value
                    if isinstance(d, TracingInt) and isinstance(d.value, int)
                    else int(d) if isinstance(d, int) else 1
                )
                for d in size
            )
        else:
            sizes = tuple(int(s) for s in size)

        t = torch.Tensor._make_wrapper_subclass(
            cls, sizes, dtype=dtype, device=device, requires_grad=requires_grad  # type: ignore
        )
        return t

    def __init__(
        self,
        size: Union[Tuple[int, ...], "TracingShape"],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
        requires_grad: bool = False,
    ):
        self._tracer: Optional["DispatchTracer"] = None
        self._node: Optional[torch.fx.Node] = None

    def __repr__(self) -> str:  # type: ignore
        node_name = self._node.name if self._node is not None else "<unregistered>"
        return (
            f"TracingTensor(node={node_name!r}, shape={tuple(self.shape)}, "
            f"dtype={self.dtype}, device={self.device})"
        )

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
        kwargs = kwargs or {}

        # Locate the :class:`DispatchTracer` from the first TracingTensor leaf.
        tracer: Optional["DispatchTracer"] = None
        all_inputs = (*args, *kwargs.values())
        for leaf in pytree.tree_leaves(all_inputs):
            if isinstance(leaf, TracingTensor) and leaf._tracer is not None:
                tracer = leaf._tracer
                break

        if tracer is None:
            # No tracer found — fall back to the default behaviour.
            return NotImplemented

        return tracer._dispatch(func, args, kwargs)


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
    ) -> "TracingTensor":
        """Create a :class:`TracingTensor` and register it with *node*."""
        t = TracingTensor.__new__(TracingTensor, shape, dtype=dtype, device=device)
        t.__init__(shape, dtype=dtype, device=device)  # type: ignore
        t._tracer = self
        t._node = node
        self._tensor_id_to_node[id(t)] = node
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
        if isinstance(meta_out, torch.Tensor):
            node.meta["val"] = meta_out

        # --- 4. wrap output ---
        return self._wrap_output(meta_out, node, device)

    def _wrap_output(
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
                    results.append(
                        self._make_tracing_tensor(item.shape, item.dtype, device, get_node)
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
