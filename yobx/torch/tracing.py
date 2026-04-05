import contextlib
import inspect
import math
import operator
import textwrap
import types
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
from torch.fx import Node
from torch.fx.proxy import TracerBase
from ..helpers import flatten_object, string_type
from .fake_tensor_helper import make_fake_with_dynamic_dimensions
from .torch_helper import torch_deepcopy

_torch_cat = torch.cat

_MISSING: Any = object()  # sentinel for "no concrete value supplied"


def _infer_ndim_from_node(node: "torch.fx.Node", _visited: Optional[set] = None) -> Optional[int]:
    """
    Infer the rank (number of dimensions) of the tensor produced by ``node``.

    For placeholder nodes the rank can be read directly from ``node.meta["val"]``.
    For intermediate computation nodes (e.g. the result of ``x + 1``) the
    metadata is not yet available at trace time, so we walk up the graph and
    return the maximum rank found among all tensor-valued input nodes.  Using
    the maximum is consistent with NumPy/PyTorch broadcasting semantics: when
    operands have different numbers of dimensions the result always has at least
    as many dimensions as the largest input.

    :param node: The FX node whose rank we want to determine.
    :param _visited: Internal set used to avoid infinite loops in cyclic graphs.
    :return: The inferred rank as an :class:`int`, or ``None`` if it cannot be
        determined.
    """
    if _visited is None:
        _visited = set()
    if id(node) in _visited:
        return None
    _visited.add(id(node))

    if "val" in node.meta:
        val = node.meta["val"]
        if isinstance(val, torch.Tensor):
            return val.ndim

    # For intermediate nodes, propagate from tensor-valued arguments.
    # We need to handle list/tuple args (e.g. torch.cat's first argument is a
    # list of tensors) in addition to plain Node arguments.
    best: Optional[int] = None

    def _scan(items):
        nonlocal best
        for arg in items:
            if isinstance(arg, torch.fx.Node):
                d = _infer_ndim_from_node(arg, _visited)
                if d is not None and (best is None or d > best):
                    best = d
            elif isinstance(arg, (list, tuple)):
                _scan(arg)

    _scan(list(node.args) + list(node.kwargs.values()))
    return best


class LEAVE_INPLACE:
    "Constant indicating inplace removal failed."


def setitem_with_transformation(a, b, transformations):
    """Extended version of setitem to deal with inplace modification."""
    function_table = {"exp": torch.exp_, "sigmoid": torch.sigmoid_}
    assert transformations, "transformations is empty, it means identity?"
    for name, args in transformations:
        assert not args, f"Not implemented for name={name!r} and args={args!r}"
        f = function_table[name]
        f(a[b])
    # operator.setitem(a, b, c)
    return a


class CustomProxy(torch.fx.proxy.Proxy):
    """
    Defines a custom proxy to trace the execution of a model
    and converts it into a fx graph.
    Works with :class:`CustomTracer`.
    """

    def __init__(self, node: Node, tracer: Optional["TracerBase"] = None):
        super().__init__(node, tracer=tracer)
        assert isinstance(
            self.tracer, CustomTracer
        ), f"Unexpected type {type(self.tracer)} for the tracer."

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.node.name})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.node.name}), meta={self.node.meta}"

    def _custom_fx_repr_fn(self) -> str:
        "To avoid bugs."
        return f"CustomProxy(%{str(self.node)})"

    def __getattr__(self, k) -> "CustomAttribute":
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        if k in ("dtype", "device"):
            # Return the concrete value so that dtype/device comparisons in
            # control flow (e.g. ``if x.dtype == torch.int64:``) resolve to a
            # plain Python bool instead of a proxy, which would raise
            # ``TraceError: symbolically traced variables cannot be used as
            # inputs to control flow``.
            # Use __dict__ lookup to avoid triggering the lazy node property on
            # CustomAttribute subclasses (which would create unwanted graph nodes).
            node = self.__dict__.get("node")
            if node is not None and "val" in node.meta:
                val = node.meta["val"]
                if isinstance(val, torch.Tensor):
                    return getattr(val, k)
        if k == "shape":
            node = self.__dict__.get("node")
            if node is not None and "val" in node.meta:
                val = node.meta["val"]
                if isinstance(val, torch.Tensor):
                    shape = val.shape
                    if all(isinstance(d, int) for d in shape):
                        # All dimensions are concrete static integers: return
                        # torch.Size directly so that downstream code receives
                        # plain ints.
                        return shape
                    # Dynamic shape: return a CustomProxyShape so that
                    # elements are CustomProxyInt instances.
                    # Comparing an element against a plain int
                    # (e.g. ``x.shape[2] == 0``) then evaluates to a Python
                    # bool without raising TraceError, while cross-tensor
                    # comparisons (``x.shape[0] == y.shape[0]``) still raise
                    # as before.
                    node = self.tracer.create_node(
                        "call_method", "size", args=(self.node,), kwargs={}
                    )
                    shape_proxy = self.tracer.proxy(node, cls=CustomProxyShape)
                    return CustomProxyShape.from_proxy(shape_proxy, shape)
            # In any other case (no meta val or val is not a Tensor), emit a node.
            node = self.tracer.create_node("call_method", "size", args=(self.node,), kwargs={})
            tt = self.tracer.proxy(node, cls=CustomProxyShape)
            return tt
        if k == "ndim":
            node = self.__dict__.get("node")
            if node is not None:
                ndim = _infer_ndim_from_node(node)
                if ndim is not None:
                    return ndim
            raise RuntimeError(
                f"The rank of a tensor is always known. "
                f"k={k!r} - {self=} - {self.node=} - {self.node.meta=}"
            )
        return CustomAttribute(self, k)

    @classmethod
    def __torch_function__(cls, orig_method, types, args=None, kwargs=None):
        if isinstance(orig_method, torch._ops.HigherOrderOperator):
            # not implemented by torch
            if orig_method is torch.cond:
                assert (
                    not kwargs
                ), f"Unexpected kwargs={kwargs}, args={args}, orig_method={orig_method}"
                assert (
                    len(args) == 4
                ), f"Unexpected kwargs={kwargs}, args={args}, orig_method={orig_method}"
                assert isinstance(
                    args[3], (list, tuple)
                ), f"Unexpected type {type(args[3])} for the last argument"
                root = args[0]
                cond_true = root.tracer.register_callable("cond", args[1])
                cond_false = root.tracer.register_callable("cond", args[2])
                node = root.tracer.create_node(
                    "call_function",
                    orig_method,
                    args=(
                        args[0].node,
                        cond_true,
                        cond_false,
                        type(args[3])(a.node for a in args[3]),
                    ),
                    kwargs={},
                )
                return root.tracer.proxy(node)

            if isinstance(orig_method, ScanCCOp):
                # args = (f, init_states, scan_inputs, additional_inputs[, dim, reverse])
                # Find a root proxy from the tensor lists so we can access the tracer.
                root = None
                for lst in (args[1], args[2], args[3] if len(args) > 3 else []):
                    if not isinstance(lst, (list, tuple)):
                        continue
                    for a in lst:
                        if isinstance(a, CustomProxy):
                            root = a
                            break
                    if root is not None:
                        break
                assert (
                    root is not None
                ), f"Unable to find a proxy in scan args={args}, orig_method={orig_method}"
                scan_fn = args[0]
                init_states = args[1]
                scan_inputs = args[2]
                additional_inputs = (
                    args[3] if len(args) > 3 else (kwargs or {}).get("additional_inputs", [])
                )
                scan_fn_node = root.tracer.register_callable("scan", scan_fn)
                init_nodes = [a.node if hasattr(a, "node") else a for a in init_states]
                scan_nodes = [a.node if hasattr(a, "node") else a for a in scan_inputs]
                add_nodes = [
                    a.node if hasattr(a, "node") else a for a in (additional_inputs or [])
                ]
                node = root.tracer.create_node(
                    "call_function",
                    orig_method,
                    args=(scan_fn_node, init_nodes, scan_nodes, add_nodes),
                    kwargs={},
                )
                return root.tracer.proxy(node)

        return torch.fx.proxy.Proxy.__torch_function__(
            orig_method, types, args=args, kwargs=kwargs
        )

    def __setitem__(self, *args, **kwargs):
        assert not kwargs, f"Unexpected not empty kwargs={kwargs!r}"
        assert len(args) == 2, f"Unexpected number of args={len(args)}: {args}"
        indices, values = args
        if isinstance(indices, CustomProxy):
            indices = indices.node
        node = self.tracer.create_node(
            "call_function",
            operator.setitem,
            args=(self.node, indices, values.node if hasattr(values, "node") else values),
            kwargs={},
        )
        # node_to_replace = self.node
        return self.tracer.proxy(node)

    def __len__(self):
        raise RuntimeError(
            f"'len' is not supported in symbolic tracing by default, "
            f"you need to detect if the model is being traced and then "
            f"call 'self.length()'. "
            f"self={self!r}, node={self.node!r}, op={self.node.op!r}, "
            f"node.meta={self.node.meta}. "
        )

    def length(self):
        """Returns a proxy for the length."""
        node = self.tracer.create_node("call_method", "__len__", args=(self.node,), kwargs={})
        tt = self.tracer.proxy(node, cls=CustomProxyInt)
        return tt

    def numel(self, *args: Any, **kwargs: Any) -> "CustomProxyInt":
        """
        Records a ``numel()`` call in the FX graph and returns a
        :class:`CustomProxyInt`.

        When the fake-tensor metadata carries a concrete element count,
        ``concrete_val`` is set on the returned proxy so that comparisons
        such as ``if x.numel() == 0:`` resolve to a plain Python ``bool``
        at trace time without raising ``TraceError``.  All other comparisons
        (e.g. ``x.numel() > 0``) keep the standard FX-proxy behaviour.
        """
        concrete_numel: Any = _MISSING
        _node = self.__dict__.get("node")
        if _node is not None and "val" in _node.meta:
            val = _node.meta["val"]
            if isinstance(val, torch.Tensor):
                concrete_numel = val.numel()
            elif isinstance(val, CustomProxy):
                concrete_numel = val.numel()
        node = self.tracer.create_node(
            "call_method", "numel", args=(self.node, *args), kwargs=kwargs
        )
        if concrete_numel is not _MISSING and isinstance(concrete_numel, int):
            return concrete_numel
        return self.tracer.proxy(node, cls=CustomProxyInt, concrete_val=concrete_numel)

    def size(self, dim: Optional[int] = None):
        """
        Records a ``size()`` or ``size(dim)`` call in the FX graph.

        * ``size()`` — returns a :class:`CustomProxyShape` (dynamic shapes) or a
          plain :class:`torch.Size` (all-static shapes), matching the behaviour
          of accessing ``.shape``.
        * ``size(dim)`` — returns a :class:`CustomProxyInt` whose ``concrete_val``
          is set from fake-tensor metadata so that comparisons such as
          ``if x.size(0) == 0:`` resolve to a plain Python ``bool`` at trace time
          without raising ``TraceError``.
        """
        _node = self.__dict__.get("node")
        concrete_shape: Any = None
        if _node is not None and "val" in _node.meta:
            val = _node.meta["val"]
            if isinstance(val, torch.Tensor):
                concrete_shape = val.shape
        if dim is None:
            if concrete_shape is not None:
                if all(isinstance(d, int) for d in concrete_shape):
                    # All dimensions are concrete static integers: return
                    # torch.Size directly so that downstream code receives plain ints.
                    return concrete_shape
                # Dynamic shape: create a call_method("size") node and wrap in
                # CustomProxyShape so each element is a CustomProxyInt with a
                # concrete_val, enabling ``if x.size()[0] == 0:`` at trace time.
                size_node = self.tracer.create_node(
                    "call_method", "size", args=(self.node,), kwargs={}
                )
                return CustomProxyShape.from_proxy(self.tracer.proxy(size_node), concrete_shape)
            # No fake-tensor metadata: return a plain CustomProxyShape proxy
            # (values will be _MISSING; individual elements can still be traced
            # via getitem when indexed).
            size_node = self.tracer.create_node(
                "call_method", "size", args=(self.node,), kwargs={}
            )
            return self.tracer.proxy(size_node, cls=CustomProxyShape)
        else:
            concrete_val: Any = _MISSING
            if concrete_shape is not None:
                concrete_val = concrete_shape[dim]
            size_node = self.tracer.create_node(
                "call_method", "size", args=(self.node, dim), kwargs={}
            )
            return self.tracer.proxy(size_node, cls=CustomProxyInt, concrete_val=concrete_val)

    @classmethod
    def cat(
        cls, tensors: List["CustomProxy"], dim: int = 0, *, out=None, axis: Optional[int] = None
    ) -> "CustomProxy":
        """Implements cat for tensors."""
        assert out is None, "Tracing is not implementing is out is not None."
        if isinstance(tensors, list):
            if any(isinstance(t, CustomProxy) for t in tensors):
                proxy = next(t for t in tensors if isinstance(t, CustomProxy))
                new_tensors = []
                for t in tensors:
                    if isinstance(t, CustomProxy):
                        new_tensors.append(t)
                        continue
                    if isinstance(t, torch.Tensor):
                        # This is not expected, it may be due to a unit test.
                        ph = proxy.tracer.create_node(
                            "placeholder", f"tcat{id(t)}", args=(), kwargs={}
                        )
                        if not ph.meta:
                            ph.meta = {}
                        ph.meta["val"] = t
                        p = proxy.tracer.proxy(ph, cls=proxy.__class__)
                        new_tensors.append(p)
                        continue

                    raise TypeError(f"A tensor is expected not {type(t)}.")
                node = proxy.tracer.create_node(
                    "call_function", torch.cat, args=(new_tensors, dim), kwargs={}
                )
                return proxy.tracer.proxy(node)
            return _torch_cat(tensors, dim)
        if axis is not None and dim == 0:
            dim = axis
        proxy = tensors
        node = proxy.tracer.create_node(
            "call_function", torch.cat, args=(proxy.node, dim), kwargs={}
        )
        return proxy.tracer.proxy(node)


class CustomProxyBool(CustomProxy):
    "A proxy for a boolean."

    __hash__ = CustomProxy.__hash__  # restore hash after __eq__ override

    def _bool_op(self, op, other):
        """Creates a binary boolean operation node and returns a :class:`CustomProxyBool`."""
        # Constant-fold when the non-proxy operand is a known bool literal.
        if isinstance(other, bool):
            if op is operator.and_:
                return False if not other else self
            if op is operator.or_:
                return True if other else self
            if op is operator.xor:
                return ~self if other else self
        node = self.tracer.create_node(
            "call_function",
            op,
            args=(self.node, other.node if isinstance(other, CustomProxy) else other),
            kwargs={},
        )
        return self.tracer.proxy(node, cls=CustomProxyBool)

    def _bool_op_reversed(self, op, other):
        """Creates a binary boolean operation node
        (reversed) and returns a :class:`CustomProxyBool`."""
        # Constant-fold when the non-proxy operand is a known bool literal.
        if isinstance(other, bool):
            if op is operator.and_:
                return False if not other else self
            if op is operator.or_:
                return True if other else self
            if op is operator.xor:
                return ~self if other else self
        node = self.tracer.create_node(
            "call_function",
            op,
            args=(other.node if isinstance(other, CustomProxy) else other, self.node),
            kwargs={},
        )
        return self.tracer.proxy(node, cls=CustomProxyBool)

    def __and__(self, other):
        return self._bool_op(operator.and_, other)

    def __rand__(self, other):
        return self._bool_op_reversed(operator.and_, other)

    def __or__(self, other):
        return self._bool_op(operator.or_, other)

    def __ror__(self, other):
        return self._bool_op_reversed(operator.or_, other)

    def __xor__(self, other):
        return self._bool_op(operator.xor, other)

    def __rxor__(self, other):
        return self._bool_op_reversed(operator.xor, other)

    def __invert__(self):
        """Logical not (~b)."""
        node = self.tracer.create_node(
            "call_function", operator.not_, args=(self.node,), kwargs={}
        )
        return self.tracer.proxy(node, cls=CustomProxyBool)

    def __eq__(self, other):  # type: ignore[override]
        return self._bool_op(operator.eq, other)

    def __ne__(self, other):  # type: ignore[override]
        return self._bool_op(operator.ne, other)


class CustomProxyInt(CustomProxy):
    """A proxy for an integer.

    When constructed with a *concrete_val* (e.g. a backed ``SymInt`` from
    fake-tensor metadata), ``==`` and ``!=`` against a plain Python
    integer / float are evaluated concretely so that patterns like
    ``if x.shape[2] == 0:`` work during symbolic tracing without raising
    ``TraceError``.  All other comparisons (including proxy-vs-proxy) still
    create FX nodes as usual.

    When *only_positive* is ``True`` (set e.g. after seeing
    ``torch._check(value > 0)``), comparisons against non-positive constants
    are evaluated concretely:

    * ``value > c`` for ``c <= 0`` → ``True``
    * ``value >= c`` for ``c <= 0`` → ``True``  (since value > 0 ≥ c)
    * ``value < c`` for ``c <= 0`` → ``False``
    * ``value <= c`` for ``c <= 0`` → ``False``
    * ``value != c`` for ``c <= 0`` → ``True``

    When *can_be_null* is ``False`` (set e.g. after seeing
    ``torch._check(value != 0)``), zero-comparisons are evaluated concretely:

    * ``value == 0`` → ``False``
    * ``value != 0`` → ``True``
    """

    __hash__ = CustomProxy.__hash__  # restore hash after __eq__ override

    def __repr__(self):
        return f"{self.__class__.__name__}({self._concrete_val})"

    def __init__(
        self,
        node: Node,
        tracer: Optional["TracerBase"] = None,
        concrete_val: Any = _MISSING,
        only_positive: bool = False,
        can_be_null: bool = True,
    ):
        super().__init__(node, tracer=tracer)
        self._concrete_val = concrete_val
        self.only_positive = only_positive
        self.can_be_null = can_be_null
        assert not isinstance(
            concrete_val, int
        ), f"concrete_val is an integer {concrete_val}, CustomProxyInt is not needed"

    def _compare(self, op, other):
        """Creates a comparison node and returns a :class:`CustomProxyBool`."""
        node = self.tracer.create_node(
            "call_function",
            op,
            args=(self.node, other.node if isinstance(other, CustomProxy) else other),
            kwargs={},
        )
        return self.tracer.proxy(node, cls=CustomProxyBool)

    def __eq__(self, other):  # type: ignore[override]
        if (
            self._concrete_val is not _MISSING
            and isinstance(other, (int, float))
            and not isinstance(other, bool)
        ):
            return bool(self._concrete_val == other)
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            if self.only_positive and other <= 0:
                # Strictly positive ⟹ value > 0, cannot equal a non-positive constant.
                return False
            if not self.can_be_null and other == 0:
                return False
        return self._compare(operator.eq, other)

    def __ne__(self, other):  # type: ignore[override]
        if (
            self._concrete_val is not _MISSING
            and isinstance(other, (int, float))
            and not isinstance(other, bool)
        ):
            return not bool(self._concrete_val == other)
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            if self.only_positive and other <= 0:
                # Strictly positive ⟹ value > 0, never equal to a non-positive constant.
                return True
            if not self.can_be_null and other == 0:
                return True
        return self._compare(operator.ne, other)

    def __lt__(self, other):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            if self.only_positive and other <= 0:
                # Strictly positive ⟹ value > 0, so value < other is impossible
                # when other <= 0.
                return False
        return self._compare(operator.lt, other)

    def __le__(self, other):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            if self.only_positive and other <= 0:
                # Strictly positive ⟹ value > 0, so value <= other is impossible
                # when other <= 0.
                return False
        return self._compare(operator.le, other)

    def __gt__(self, other):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            if self.only_positive and other <= 0:
                # Strictly positive ⟹ value > 0 > other (or value > 0 = other),
                # so True.
                return True
        return self._compare(operator.gt, other)

    def __ge__(self, other):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            if self.only_positive and other <= 0:
                # Strictly positive ⟹ value > 0 ≥ other, so True.
                return True
        return self._compare(operator.ge, other)


class CustomProxyFloat(CustomProxy):
    "A proxy for a float."


class CustomAttribute(CustomProxy):
    """To trace attributes."""

    def __init__(self, root: CustomProxy, attr: str):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Optional[Node] = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.attr})"

    @property
    def node(self):
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy(
                "call_function", getattr, (self.root, self.attr), {}
            ).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy("call_method", self.attr, (self.root, *args), kwargs)


class CustomProxyShape(CustomProxy):
    """
    A :class:`tuple` of :class:`CustomProxyInt` instances representing a
    tensor shape with dynamic dimensions.

    Each element is a valid FX proxy node (so dynamic-shape operators such
    as ``torch.full`` are recorded correctly in the graph) *and* supports
    equality comparison against plain integer constants so that
    ``if x.shape[2] == 0:`` evaluates to a Python ``bool`` without raising
    ``TraceError``.

    Use :meth:`from_proxy` to construct an instance from a shape
    :class:`CustomAttribute` proxy and its corresponding concrete
    ``torch.Size``.
    """

    def __init__(
        self, node: Node, tracer: Optional["TracerBase"] = None, concrete_val: Any = _MISSING
    ):
        super().__init__(node, tracer=tracer)
        self.values = concrete_val
        assert concrete_val is _MISSING or all(
            isinstance(v, (int, CustomProxyInt)) for v in concrete_val
        ), (
            f"Unexpected type in values (type(values)={type(concrete_val)}), "
            f"{'MISSING' if concrete_val is _MISSING else [type(v) for v in concrete_val]}"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.values})"

    def __len__(self) -> int:
        if self.values is _MISSING:
            raise NotImplementedError(
                "Cannot compute the length of a CustomProxyShape without concrete values. "
                "Ensure concrete shape metadata is available, or use the proxy node directly."
            )
        return len(self.values)

    def length(self) -> int:
        if self.values is _MISSING:
            raise NotImplementedError(
                "Cannot compute the length of a CustomProxyShape without concrete values. "
                "Ensure concrete shape metadata is available, or use the proxy node directly."
            )
        return len(self.values)

    def __iter__(self):
        if self.values is _MISSING:
            raise NotImplementedError(
                "Cannot iterate a CustomProxyShape without concrete values. "
                "Ensure concrete shape metadata is available."
            )
        yield from self.values

    def __getitem__(self, index):
        if isinstance(index, int):
            if self.values is _MISSING:
                # Let's return a traced int.
                node = self.tracer.create_node(
                    "call_function", operator.getitem, args=(self.node, index), kwargs={}
                )
                return self.tracer.proxy(node, cls=CustomProxyInt)
            return self.values[index]
        if isinstance(index, slice):
            if self.values is _MISSING:
                raise NotImplementedError(
                    "Slicing a CustomProxyShape requires concrete values. "
                    "Ensure concrete shape metadata is available."
                )
            return self.values[index]
        raise TypeError(f"{type(index)=} is unexpected for {self.__class__=}")

    @classmethod
    def from_proxy(
        cls, shape_proxy: "CustomProxy", concrete_shape: "torch.Size"
    ) -> "CustomProxyShape":
        """
        Build a :class:`CustomProxyShape` from a shape/size proxy and
        the corresponding concrete (possibly symbolic) ``torch.Size``.

        Parameters
        ----------
        shape_proxy:
            An FX proxy whose ``[i]`` subscript creates a ``getitem`` graph
            node (e.g. a ``CustomAttribute`` for ``tensor.shape`` or a
            ``CustomProxy`` wrapping a ``call_method("size", …)`` node).
        concrete_shape:
            The concrete ``torch.Size`` from the fake tensor's metadata
            (may contain backed ``SymInt`` values for dynamic dimensions).
        """
        items = []
        for i in range(len(concrete_shape)):
            item_proxy: CustomProxy = shape_proxy[i]  # type: ignore[assignment]
            concrete_val = concrete_shape[i]
            if isinstance(concrete_val, int):
                items.append(concrete_val)
            else:
                cpi = CustomProxyInt(item_proxy.node, item_proxy.tracer, concrete_val)
                # Register in the tracer's node→proxy map so that
                # _torch_check_for_tracing can find this proxy by its node.
                node_proxy_map = getattr(item_proxy.tracer, "_node_proxy_map", None)
                if node_proxy_map is not None:
                    node_proxy_map[item_proxy.node] = cpi
                items.append(cpi)
        # Tie the CustomProxyShape to the underlying proxy node and tracer, then
        # attach the per-dimension proxies as values (bypassing the init assertion
        # so the assignment is made directly after construction).
        shape = cls(shape_proxy.node, shape_proxy.tracer)
        shape.values = tuple(items)
        return shape


class CustomParameterProxy(CustomProxy):
    """
    A special proxy which lets "shape", "size", "dim", and a few other
    attribute accesses pass through to the underlying  module parameter object,
    so that conditional tests on these attributes will not throw exception during tracing.
    """

    def __init__(self, tracer: TracerBase, node: Node, name, param):
        super().__init__(node, tracer)
        assert isinstance(param, torch.nn.Parameter)
        self.param = param
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    @property
    def shape(self):
        return self.param.shape

    def size(self, dim=None):
        return self.param.size() if dim is None else self.param.size(dim)

    def dim(self):
        return self.param.dim()

    @property
    def ndim(self):
        return self.param.ndim

    def numel(self):
        return self.param.numel()

    def nelement(self):
        return self.param.nelement()


def tree_unflatten_with_proxy(tree_spec: pytree.PyTree, leaves: Iterable[Any]) -> Any:
    """
    More robust implementation of ``pytree.tree_unflatten``
    supporting ``DynamicCache``.
    """
    if isinstance(leaves, (list, tuple)):
        leaves = list(leaves)
    assert len(leaves) == tree_spec.num_leaves, (
        f"treespec.unflatten(leaves): `leaves` has length {len(leaves)} "
        f"but the spec refers to a pytree that holds {tree_spec.num_leaves} "
        f"items ({tree_spec}).",
    )
    if tree_spec.is_leaf():
        return leaves[0]

    unflatten_fn = pytree.SUPPORTED_NODES[tree_spec.type].unflatten_fn

    # Recursively unflatten the children
    start = 0
    end = 0
    child_pytrees = []
    for child_spec in tree_spec._children:
        end += child_spec.num_leaves
        assert not child_spec or not child_spec.type or child_spec.unflatten, (
            f"child_spec.unflatten is empty for child_spec.type={child_spec.type}, "
            f"child_spec={child_spec}"
        )
        child_pytrees.append(child_spec.unflatten(leaves[start:end]))
        start = end
    return unflatten_fn(child_pytrees, tree_spec._context)


class CondCCOp(torch._ops.HigherOrderOperator):
    """
    Cannot be imported from torch.ops.higher_order.cond
    (function cond overwrite submodule cond).
    """

    def __init__(self):
        # we cannot use "cond" to avoid confusion with the existing cond
        super().__init__("condcc")

    def __call__(self, pred, true_fn, false_fn, operands):
        # torch._higher_order_ops.utils.validate_subgraph_args_types(operands)
        return super().__call__(pred, true_fn, false_fn, operands)


class ScanCCOp(torch._ops.HigherOrderOperator):
    """
    Replacement for :func:`torch.ops.higher_order.scan` during FX symbolic
    tracing. The real scan operator cannot be called with :class:`CustomProxy`
    arguments because it tries to execute the body function eagerly. This proxy
    operator defers to :meth:`CustomProxy.__torch_function__` which records the
    scan call as a FX ``call_function`` node instead.
    """

    def __init__(self):
        # use "scancc" to avoid shadowing the real "scan" operator
        super().__init__("scancc")

    def __call__(
        self,
        f: Callable,
        init_states: List,
        scan_inputs: List,
        additional_inputs: Optional[List] = None,
        dim: int = 0,
        reverse: bool = False,
    ):
        return super().__call__(
            f, init_states, scan_inputs, additional_inputs or [], dim, reverse
        )


def _vmap_for_tracing(func: Callable, in_dims: Any = 0, out_dims: Any = 0, **kwargs) -> Callable:
    """
    Replacement for :func:`torch.vmap` during FX symbolic tracing.

    During tracing all model arguments are :class:`torch.fx.proxy.Proxy` objects,
    so the real vmap dispatch cannot run. This wrapper traces through the body of
    *func* directly with the (still-batched) proxy arguments.

    :param func: the function to be mapped over the batch dimension
    :param in_dims: accepted for API compatibility with :func:`torch.vmap` but ignored
        during tracing — the full batched tensors are passed directly to *func*
    :param out_dims: accepted for API compatibility with :func:`torch.vmap` but ignored
        during tracing
    :param kwargs: any additional keyword arguments accepted for API compatibility
        but ignored during tracing

    .. warning::
        This replacement is only correct when *func* applies element-wise operations
        so that ``func(x_batch)`` is equivalent to ``vmap(func)(x_batch)`` (i.e. the
        function commutes with batching).  Non-element-wise functions — for example
        those that call :func:`torch.mm` across the batch axis — will produce an
        incorrect trace.
    """

    def wrapped(*args):
        return func(*args)

    return wrapped


_torch_check = getattr(torch, "_check", None)


def _torch_check_for_tracing(cond: Any, msg: Any = None) -> None:
    """
    Replacement for :func:`torch._check` during :class:`CustomTracer` symbolic
    tracing.

    :func:`torch._check` is intended as a runtime assertion that a condition
    holds.  During symbolic tracing with :class:`CustomTracer` the condition
    is often a :class:`CustomProxy` or :class:`CustomProxyBool` — an FX graph
    node whose concrete boolean value is not available.  Calling
    ``bool(proxy)`` on such a node raises ``torch.fx.proxy.TraceError``.

    This wrapper avoids that error by **not** evaluating ``bool(cond)`` when
    *cond* is a proxy.  Instead it:

    1. Silently accepts the constraint (the assertion is assumed to hold, as
       it would at runtime).
    2. If the condition looks like ``proxy_int > 0`` / ``proxy_int >= 1``
       (or the symmetric ``0 < proxy_int``), it sets
       :attr:`CustomProxyInt.only_positive` on the underlying
       :class:`CustomProxyInt` proxy so that downstream comparisons such as
       ``if proxy_int > 0:`` can be resolved to a concrete Python ``True``
       without creating additional FX nodes.
    3. If the condition looks like ``proxy_int != 0``, it clears
       :attr:`CustomProxyInt.can_be_null` on the underlying proxy.

    For concrete Python ``bool`` values the original :func:`torch._check` is
    called unchanged.
    """
    if isinstance(cond, CustomProxy):
        # Try to propagate constraint information to the underlying CustomProxyInt.
        cond_node = cond.node
        if cond_node.op == "call_function" and len(cond_node.args) == 2:
            op = cond_node.target
            lhs_node, rhs = cond_node.args
            # Normalise: if rhs is a Node (proxy on the right), flip for gt/ge/lt/le.
            if isinstance(rhs, torch.fx.Node) and not isinstance(lhs_node, torch.fx.Node):
                # e.g. 0 < proxy → swap to proxy > 0
                lhs_node, rhs = rhs, lhs_node
                op = {
                    operator.gt: operator.lt,
                    operator.lt: operator.gt,
                    operator.ge: operator.le,
                    operator.le: operator.ge,
                    operator.eq: operator.eq,
                    operator.ne: operator.ne,
                }.get(op, op)
            tracer = cond.tracer
            node_proxy_map: Dict[torch.fx.Node, Any] = getattr(tracer, "_node_proxy_map", {})
            lhs_proxy = (
                node_proxy_map.get(lhs_node) if isinstance(lhs_node, torch.fx.Node) else None
            )
            if isinstance(lhs_proxy, CustomProxyInt) and isinstance(rhs, (int, float)):
                if op is operator.gt and rhs == 0:
                    # torch._check(x > 0): x is strictly positive
                    lhs_proxy.only_positive = True
                    lhs_proxy.can_be_null = False
                elif op is operator.ge and rhs == 1:
                    # torch._check(x >= 1): x is strictly positive (integer)
                    lhs_proxy.only_positive = True
                    lhs_proxy.can_be_null = False
                elif op is operator.ne and rhs == 0:
                    # torch._check(x != 0)
                    lhs_proxy.can_be_null = False
        return  # Do not call bool(cond) – it would raise TraceError.
    if _torch_check is not None:
        if msg is not None:
            _torch_check(cond, msg)
        else:
            _torch_check(cond)


@contextlib.contextmanager
def replace_problematic_function_before_tracing() -> Generator:
    """
    Replaces function that cannot be traced with the default tracer
    such as :func:`torch.cat`.
    """
    _scan_op = getattr(torch.ops.higher_order, "scan", None)
    saved = {"cat": torch.cat, "cond": torch.cond, "vmap": torch.vmap}
    if _torch_check is not None:
        saved["_check"] = _torch_check
    newf = {"cat": CustomProxy.cat, "cond": CondCCOp(), "vmap": _vmap_for_tracing}
    if _torch_check is not None:
        newf["_check"] = _torch_check_for_tracing
    if _scan_op is not None:
        saved[(torch.ops.higher_order, "scan")] = _scan_op
        newf[(torch.ops.higher_order, "scan")] = ScanCCOp()
    for k, v in newf.items():
        if isinstance(k, tuple):
            setattr(k[0], k[1], v)
        else:
            setattr(torch, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            if isinstance(k, tuple):
                setattr(k[0], k[1], v)
            else:
                setattr(torch, k, v)


_AUTOWRAP_FUNCTIONS: Tuple[Callable, ...] = (
    torch.ones,
    torch.zeros,
    torch.full,
    torch.empty,
    torch.arange,
)
"""
Default functions to autowrap in :class:`CustomTracer`.
These tensor-creating functions take integer size arguments and would
fail during symbolic tracing when size values are proxy objects.
By autowrapping them, FX records their calls as traced nodes rather
than attempting to execute them immediately.
"""


class CustomTracer(torch.fx.Tracer):
    """
    Defines a custom tracer to trace the execution of a model
    and converts it into a fx graph.
    Works with :class:`CustomProxy`.

    ::
        from yobx.torch.tracing import CustomTracer

        graph = CustomTracer().trace(model)

    :param autowrap_modules: defaults to `(math, )`,
        Python modules whose functions should be wrapped automatically
        without needing to use fx.wrap().
    :param autowrap_functions: defaults to :data:`_AUTOWRAP_FUNCTIONS`,
        Python functions that should be wrapped automatically without
        needing to use fx.wrap(). Includes tensor-creating functions
        (e.g. ``torch.ones``) so that calls with proxy size arguments
        are captured as traced nodes rather than executed immediately.
    :param param_shapes_constant: When this flag is set, calls to shape,
        size and a few other shape like attributes of a module's parameter
        will be evaluated directly, rather than returning a new Proxy value
        for an attribute access.
    :param module_leaves: modules to be considered as leaves,
        mapped to a callable ``f(module, module_qualified_name) -> bool``
        that decides whether a specific module instance is a leaf;
        the tracer does not trace into leaf modules and emits
        ``call_module`` nodes for them instead
    """

    def __init__(
        self,
        autowrap_modules: Tuple[types.ModuleType, ...] = (math,),
        autowrap_functions: Tuple[Callable, ...] = _AUTOWRAP_FUNCTIONS,
        param_shapes_constant: bool = False,
        module_leaves: Optional[Dict[type, Callable[[torch.nn.Module, str], bool]]] = None,
    ):
        super().__init__(
            autowrap_modules=autowrap_modules,
            autowrap_functions=autowrap_functions,
            param_shapes_constant=param_shapes_constant,
        )
        self._callables = {}
        self.module_leaves = module_leaves
        # Maps each FX node to the :class:`CustomProxy` (or subclass) that was
        # created for it.  Used by :func:`_torch_check_for_tracing` to look up
        # the :class:`CustomProxyInt` associated with the operand of a comparison
        # node so that constraint flags such as ``only_positive`` can be set.
        self._node_proxy_map: Dict[torch.fx.Node, CustomProxy] = {}

    @torch.fx._compatibility.compatibility(is_backward_compatible=True)
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.

        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.

        Args:

            m (Module):
                The module being queried about
            module_qualified_name (str):
                The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        is_leaf = super().is_leaf_module(m, module_qualified_name)
        if is_leaf:
            return is_leaf
        if self.module_leaves and type(m) in self.module_leaves:
            f = self.module_leaves[type(m)]
            return f(m, module_qualified_name=module_qualified_name)
        return False

    @torch.fx._compatibility.compatibility(is_backward_compatible=True)
    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Method that specifies the behavior of this ``Tracer`` when it encounters
        a call to an ``nn.Module`` instance.

        By default, the behavior is to check if the called module is a leaf module
        via ``is_leaf_module``. If it is, emit a ``call_module`` node referring to
        ``m`` in the ``Graph``. Otherwise, call the ``Module`` normally, tracing through
        the operations in its ``forward`` function.

        This method can be overridden to--for example--create nested traced
        GraphModules, or any other behavior you would want while tracing across
        ``Module`` boundaries.

        Args:

            m (Module): The module for which a call is being emitted
            forward (Callable): The forward() method of the ``Module`` to be invoked
            args (Tuple): args of the module callsite
            kwargs (Dict): kwargs of the module callsite

        Return:

            The return value from the Module call. In the case that a ``call_module``
            node was emitted, this is a ``Proxy`` value. Otherwise, it is whatever
            value was returned from the ``Module`` invocation.
        """
        return super().call_module(m, forward, args, kwargs)

    def register_callable(self, name: str, fn: Callable) -> torch.fx.Node:
        """
        Registers a function and return a unique name.

        :param name: prefix to prepend to the function name
        :param fn: function
        :return: new_name
        """
        cand = f"_cb_{name}_{fn.__name__}_0"
        if cand in self._callables:
            i = 1
            cand = f"_cb_{name}_{fn.__name__}_{i}"
            while cand in self._callables:
                i += 1
                cand = f"_cb_{name}_{fn.__name__}_{i}"
        self._callables[cand] = fn
        return self.create_node("get_attr", cand, args=(), kwargs={})

    def proxy(
        self, node: torch.fx.Node, cls: type[CustomProxy] = CustomProxy, **kwargs
    ) -> torch.fx.Proxy:
        """Overwrites this method to replace the default Proxy by CustomProxy."""
        p = cls(node, self, **kwargs)
        self._node_proxy_map[node] = p
        return p

    def create_arg(self, a: Any) -> "Argument":  # noqa: F821
        """Overwrites this method to deal with more argument."""
        if a is bool:
            return torch.bool
        if a is int:
            return torch.int64
        if a is float:
            return torch.float32
        if a is complex:
            return torch.complex64
        res = super().create_arg(a)
        return res

    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: Dict[str, Any]):
        """See :meth:`torch.fx.Tracer.getattr`."""

        def maybe_get_proxy_for_attr(attr_val, collection_to_search, parameter_proxy_cache):
            for n, p in collection_to_search:
                if attr_val is p:
                    if n not in parameter_proxy_cache:
                        kwargs = {}
                        if "proxy_factory_fn" in inspect.signature(self.create_proxy).parameters:
                            kwargs["proxy_factory_fn"] = (
                                None
                                if not self.param_shapes_constant
                                else lambda node, n=n, attr_val=attr_val: CustomParameterProxy(
                                    self, node, n, attr_val
                                )
                            )
                        val_proxy = self.create_proxy("get_attr", n, (), {}, **kwargs)  # type: ignore[arg-type]
                        parameter_proxy_cache[n] = val_proxy
                    return parameter_proxy_cache[n]
            return None

        if isinstance(attr_val, torch.nn.Parameter):
            maybe_parameter_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_parameters(), parameter_proxy_cache
            )
            if maybe_parameter_proxy is not None:
                return maybe_parameter_proxy

        if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
            maybe_buffer_proxy = maybe_get_proxy_for_attr(
                attr_val, self.root.named_buffers(), parameter_proxy_cache
            )
            if maybe_buffer_proxy is not None:
                return maybe_buffer_proxy

        return attr_val

    def _proxy_placeholder(self, name, concrete_args, sig, fn_for_analysis):
        res = torch.fx.Tracer._proxy_placeholder(self, name, concrete_args, sig, fn_for_analysis)
        # Pre-populate node meta with the fake tensor so that CustomProxy.__getattr__
        # can resolve static attributes (dtype, device) to concrete values during
        # tracing, enabling dtype/device-based control flow without TraceError.
        # Only handles the dict case; when _traced_concrete_args is a flat list
        # (pytree-wrapped models), the wrapped parameter names differ from the
        # originals and the mapping is handled post-trace in the trace() loop.
        if (
            self._traced_concrete_args is not None
            and isinstance(self._traced_concrete_args, dict)
            and name in self._traced_concrete_args
        ):
            fake = self._traced_concrete_args[name]
            if isinstance(fake, torch.Tensor):
                res.node.meta["val"] = fake
        return res

    def create_args_for_root(self, root_fn, is_module, concrete_args=None):
        root_fn, args = torch.fx.Tracer.create_args_for_root(
            self, root_fn, is_module, concrete_args=concrete_args
        )
        assert (
            self._traced_concrete_args is None or len(self._traced_concrete_args) == len(args) - 1
        ), (
            f"Mismatch between _traced_concrete_args="
            f"{string_type(self._traced_concrete_args, with_shape=True)} and "
            f"args={string_type(args, with_shape=True)}"
        )
        return root_fn, args

    @classmethod
    def make_args_names(cls, concrete_args, flat_concrete_args):
        if not isinstance(concrete_args, dict):
            return [f"a{i}" for i in range(len(flat_concrete_args))]

        if flatten_object is not None:
            flat_conc = {k: flatten_object(v, drop_keys=True) for k, v in concrete_args.items()}
        else:
            flat_conc = {
                k: ([v] if isinstance(v, torch.Tensor) else list(v))
                for k, v in concrete_args.items()
            }
        lengths = [1 if isinstance(v, torch.Tensor) else len(v) for v in flat_conc.values()]
        assert sum(lengths) == len(flat_concrete_args), (
            f"{sum(lengths)} flattened objects != {len(flat_concrete_args)}, lengths={lengths}, "
            f"concrete_args={string_type(concrete_args, with_shape=True)}, "
            f"flat_concrete_args={string_type(flat_concrete_args, with_shape=True)}, "
        )
        names = []
        for k, v in flat_conc.items():
            if isinstance(v, torch.Tensor):
                names.append(k)
            else:
                names.extend([f"{k}_{i}" for i in range(len(v))])
        assert len(names) == len(
            flat_concrete_args
        ), f"len(names)={len(names)} != {len(flat_concrete_args)}, names={names}"
        return names

    @classmethod
    def make_wrapped_model(cls, root, concrete_args):
        flat_concrete_args, spec = pytree.tree_flatten(concrete_args)
        args_names = cls.make_args_names(concrete_args, flat_concrete_args)

        if (
            len(concrete_args) == 2
            and isinstance(concrete_args, dict)
            and set(concrete_args) == {"x", "cache"}
            and isinstance(concrete_args["x"], torch.Tensor)
            and concrete_args["cache"].__class__.__name__ == "DynamicCache"
        ):
            # this is a not generic case to check one unit test
            from .in_transformers.cache_helper import make_dynamic_cache

            def make_method(args_names):
                args = ", ".join(args_names)
                args1 = ", ".join(args_names[1:])
                src = textwrap.dedent(f"""
                    def f(self, {args}):
                        args = [{args1}]
                        cache = make_dynamic_cache(list(zip(args[::2], args[1::2])))
                        return self._traced_m1({args[0]}, cache)
                    """)
                ns = {"torch": torch, "make_dynamic_cache": make_dynamic_cache}
                exec(src, ns)
                return ns["f"]

            class FlatArgWrap(torch.nn.Module):
                def __init__(self, m, spec):
                    super().__init__()
                    self._traced_m1 = m
                    self._spec = spec

                forward = make_method(args_names)

            return FlatArgWrap(root, spec), args_names

        # pytree.tree_unflatten does not work on CustomProxy

        def make_method(args_names):
            args = ", ".join(args_names)
            src = textwrap.dedent(f"""
                def f(self, {args}):
                    res = tree_unflatten_with_proxy(self._spec, [{args}])
                    assert isinstance(res, dict), (
                        "A dictionary is expected but unflattened type is %r" % type(res)
                    )
                    return self._traced_m2(**res)
                """)
            ns = {"tree_unflatten_with_proxy": tree_unflatten_with_proxy}
            exec(src, ns)
            return ns["f"]

        class FlatArgWrap(torch.nn.Module):
            def __init__(self, m, spec):
                super().__init__()
                self._traced_m2 = m
                self._spec = spec

            forward = make_method(args_names)

        return FlatArgWrap(root, spec), args_names

    def trace(
        self,
        root: Union[torch.nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
        remove_inplace: bool = True,
        update_model_with_callable: bool = True,
        dynamic_shapes: Optional[Any] = None,
        verbose: int = 0,
    ) -> torch.fx.Graph:
        """
        Trace ``root`` and return the corresponding FX ``Graph`` representation. ``root``
        can either be an ``nn.Module`` instance or a Python callable.

        Note that after this call, ``self.root`` may be different from the ``root`` passed
        in here. For example, when a free function is passed to ``trace()``, we will
        create an ``nn.Module`` instance to use as the root and add embedded constants to.

        :param root: Either a ``Module`` or a function to be
            traced through. Backwards-compatibility for this parameter is
            guaranteed.
        :param concrete_args: Concrete arguments that should
            not be treated as Proxies. This parameter is experimental and
            its backwards-compatibility is *NOT* guaranteed.
        :param remove_inplace: Removes inplace nodes
        :param update_model_with_callable: in some cases (control flow),
            the model needs to be updated
        :param dynamic_shapes: dynamic shapes
        :param verbose: verbosity
        :return: A ``Graph`` representing the semantics of the passed-in ``root``

        If the model had to be wrapped before being traced, attribute ``traced_model``
        is added to the tracer.
        """
        assert concrete_args is None or isinstance(
            concrete_args, dict
        ), f"Unexpected type for concrete_args: {string_type(concrete_args)}"
        if verbose:
            print(
                f"[CustomTracer.trace] trace with concrete_args="
                f"{string_type(concrete_args, with_shape=True)}"
            )
            print(f"[CustomTracer.trace] trace with dynamic_shapes={dynamic_shapes}")

        traced_model = None
        if concrete_args:
            flat_args = (
                concrete_args.values() if isinstance(concrete_args, dict) else concrete_args
            )

            if any(type(a) in pytree.SUPPORTED_NODES for a in flat_args):
                # tracing does not know the input type so we need to flatten everything.
                if verbose > 0:
                    print("[CustomTracer.trace] wraps for serializable args")
                new_model, new_names = self.make_wrapped_model(root, concrete_args)
                traced_concrete_args, _ = make_fake_with_dynamic_dimensions(
                    torch_deepcopy(concrete_args), dynamic_shapes
                )
                self._traced_concrete_args, _ = pytree.tree_flatten(traced_concrete_args)
                traced_model = new_model
            else:
                new_names = None
                self._traced_concrete_args, _ = make_fake_with_dynamic_dimensions(
                    concrete_args, dynamic_shapes
                )
                new_model = root
            if verbose > 0:
                print(
                    f"[CustomTracer.trace] _traced_concrete_args="
                    f"{string_type(self._traced_concrete_args, with_shape=True)}"
                )
            with replace_problematic_function_before_tracing():
                # concrete arguments are replaced by constants whatever is given to the function
                graph = super().trace(new_model)

        else:
            self._traced_concrete_args = None
            new_names = None

            with replace_problematic_function_before_tracing():
                # concrete arguments are replaced by constants whatever is given to the function
                graph = super().trace(root)  # , concrete_args)

        if verbose >= 10:
            print("[CustomTracer.trace] -- graph")
            print(graph)
        if concrete_args:
            if new_names and verbose > 0:
                print(f"[CustomTracer.trace] -- new_names={new_names}")
            if new_names:
                flat_concrete_args, _spec = pytree.tree_flatten(concrete_args)
                flat_traced_concrete_args, _spec = pytree.tree_flatten(self._traced_concrete_args)
            mapped = set()
            for node in graph.nodes:
                if node.op == "placeholder":
                    if not new_names and node.name in concrete_args:
                        ti = concrete_args[node.name]
                        tif = self._traced_concrete_args[node.name]
                        if verbose:
                            print(
                                f"[CustomTracer.trace] assign.1 {node.name!r} with "
                                f"{string_type(ti, with_shape=True)} or "
                                f"{string_type(tif, with_shape=True)}"
                            )
                        node.meta["example_value"] = ti
                        node.meta["val"] = tif
                        mapped.add(node.name)
                    elif new_names and node.name in new_names:
                        ii = new_names.index(node.name)
                        ti = flat_concrete_args[ii]
                        tif = flat_traced_concrete_args[ii]
                        if verbose:
                            print(
                                f"[CustomTracer.trace] assign.1 {node.name!r} with "
                                f"{string_type(ti, with_shape=True)} or "
                                f"{string_type(tif, with_shape=True)}"
                            )
                        node.meta["example_value"] = ti
                        node.meta["val"] = tif
                        mapped.add(node.name)
            assert new_names or set(mapped) == set(concrete_args), (
                f"Missing mapped inputs, set(concrete_args)={set(concrete_args)}, "
                f"mapped={mapped}\n{graph}\nroot={root}"
            )
            assert not new_names or len(new_names) == len(flat_concrete_args), (
                f"Missing mapped inputs, new_names={new_names}, "
                f"flat_concrete_args={string_type(flat_concrete_args, with_shape=True)}, "
                f"mapped={mapped}\n{graph}\nroot={root}"
            )

        self._replace_problematic_functions(graph)
        if update_model_with_callable and self._callables:
            for k, v in self._callables.items():
                setattr(root, k, v)
        self.remove_unnecessary_slices(graph, verbose=verbose)
        if remove_inplace:
            self.remove_inplace(graph, verbose=verbose)
        graph.lint()
        self.traced_model = traced_model
        return graph

    @classmethod
    def _replace_problematic_functions(cls, graph: torch.fx.Graph, verbose: int = 0) -> int:
        """
        The tracing introduced some problematic functions which need to be replaced.

        :param graph: graph to process
        :param verbose: verbosity level
        :return: number of impacted nodes
        """
        replaces = {CustomProxy.cat: torch.cat}
        n = 0
        for node in graph.nodes:
            if node.op == "call_function":
                if node.target in replaces:
                    n += 1
                    if verbose > 1:
                        print(
                            f"[CustomTracer._replace_problematic_functions] replace "
                            f"{node.target} by {replaces[node.target]}"
                        )
                    node.target = replaces[node.target]
                elif isinstance(node.target, CondCCOp):
                    n += 1
                    node.target = torch.ops.higher_order.cond
                elif isinstance(node.target, ScanCCOp):
                    n += 1
                    node.target = torch.ops.higher_order.scan
        return n

    @classmethod
    def _get_aten_name(cls, node: torch.fx.Node) -> str:
        """Returns the aten name for the target as a string."""
        assert hasattr(node, "target"), f"Unable to return aten name for {node}"
        known = {operator.getitem: "getitem", operator.le: "le", operator.ge: "ge"}
        if node.target in known:
            return known[node.target]
        if isinstance(node.target, torch._ops.OpOverloadPacket):
            if node.target != torch.ops.aten.sym_size:
                raise RuntimeError(f"Unsupported function {node!r}.")
            raise NotImplementedError(f"Unsupported function {node!r} (not implemented).")

        if isinstance(node.target, types.BuiltinFunctionType):
            if node.target is math.ceil:
                # We need to distinguish between math.ceil and torch.ceil.
                # The output type is different.
                return "math_ceil"
            return str(node.target)

        if isinstance(node.target, torch._ops.OpOverload):
            return node.target.name()

        if callable(node.target):
            # a single function
            return f"aten_{node.target.__name__}"

        if isinstance(node.target, str):
            return node.target

        raise NotImplementedError(
            f"Unsupported function {node!r} (not implemented), "
            f"node.target={node.target}, type is {type(node.target)}."
        )

    @classmethod
    def _is_trully_inplace(cls, node: torch.fx.Node) -> bool:
        return True

    @classmethod
    def _inplace_nodes(cls, graph: torch.fx.Graph) -> List[Tuple[int, torch.fx.Node]]:
        """Returns the position and the node involved in inplace modifications."""
        return [
            (i, node)
            for i, node in enumerate(graph.nodes)
            if node.op != "output"
            and len(node.users) == 0
            and node.op.startswith("call_")
            and node.target not in {operator.getitem, operator.or_, operator.and_}
            and cls._get_aten_name(node)
            not in {
                "aten::_assert_scalar",
                "aten::sym_constrain_range_for_size",
                "aten::_log_api_usage_once",
                "aten::_enter_autocast",
                "aten::_exit_autocast",
                "aten::_set_grad_enabled",
                "aten__assert_scalar",
                "aten_sym_constrain_range_for_size",
                "aten__log_api_usage_once",
                "aten__enter_autocast",
                "aten__exit_autocast",
                "aten__set_grad_enabled",
            }
            and cls._is_trully_inplace(node)
        ]

    @classmethod
    def _replace_meth_setitem(cls, graph: torch.fx.Graph) -> int:
        """
        The execution of ``op="call_method", target="__setitem__" `` returns None.
        We replace it by ``op="call_function", target="operator.setitem"``.

        :return: number of impacted nodes
        """
        n = 0
        for node in graph.nodes:
            if node.op == "call_method" and node.target == "__setitem__":
                node.op = "call_function"
                node.target = operator.setitem
                n += 1
        return n

    @classmethod
    def _replace_getattr(cls, graph: torch.fx.Graph) -> int:
        """
        Nodes such as
        ``%_tensor_constant0_1 : [num_users=1] = get_attr[target=_tensor_constant0]``
        are part of the replacement in function ``replace_all_uses_with``.
        Let's remove the duplicates first.

        :return: number of impacted get_attr nodes
        """
        targets = {}
        to_replace = []
        for node in graph.nodes:
            if node.op == "get_attr":
                if node.target in targets:
                    # replacements
                    to_replace.append((node, targets[node.target]))
                else:
                    targets[node.target] = node
        if to_replace:
            for node, by in to_replace:
                node.replace_all_uses_with(by)
                graph.erase_node(node)
        return len(to_replace)

    @classmethod
    def remove_unnecessary_slices(cls, graph: torch.fx.Graph, verbose: int = 0) -> int:
        """
        Removes unnecessary slices and other nodes doing nothing.

        :param graph: graph to modify
        :param verbose: verbosity level
        :return: number of inplace nodes removed

        ::

            %slice_11 : [num_users=1] = call_function[target=torch.ops.aten.slice.Tensor]
                (args = (%clone, 0, 0, 9223372036854775807), kwargs = {})
        """
        # slices
        nodes = list(enumerate(graph.nodes))

        removed = 0
        for pos, node in nodes:
            if not hasattr(node.target, "name"):
                continue
            if node.target.name() != "aten::slice.Tensor":
                continue
            if (
                len(node.args) != 4
                or (node.args[2] != 0 and node.args[2] is not None)
                or (node.args[3] != 9223372036854775807 and node.args[3] is not None)
            ):
                continue

            # The first argument is the node to keep.
            new_name = node.args[0]
            old_name = node

            # Let's replace.
            changed = old_name.replace_all_uses_with(new_name)
            assert changed, (
                f"No change applied, the node [{node}] at position {pos} "
                f"cannot be removed and replaced by {old_name} in \n{graph}."
            )
            graph.erase_node(old_name)
            if verbose > 1:
                print(
                    f"[CustomTracer.remove_unnecessary_slices.1] replace "
                    f"{old_name!r} by {new_name!r}"
                )
            removed += 1

        # copy_.default
        nodes = list(enumerate(graph.nodes))
        max_pos = {}
        for pos, node in nodes:
            for a in node.args:
                if isinstance(a, torch.fx.Node):
                    max_pos[id(a)] = pos

        for pos, node in nodes[::-1]:
            if not hasattr(node.target, "name"):
                continue
            if node.target.name() != "aten::copy_" or len(node.args) != 2 or len(node.users) == 0:
                # Not the expected node, not the expected number of arguments
                # or not used (meaning this is partial inplace modification)
                continue
            if "val" not in node.args[0].meta or "val" not in node.args[1].meta:
                continue
            if node.args[0].meta["val"].shape != node.args[1].meta["val"].shape:
                continue
            # We need to check that node.args[0] is not used after that.
            if max_pos[id(node.args[0])] > pos:
                # The node is used after
                continue

            # The first argument is the node to keep.
            new_name = node.args[1]
            old_name = node

            # Let's replace.
            changed = old_name.replace_all_uses_with(new_name)
            assert changed, (
                f"No change applied, the node [{node}] at position {pos} "
                f"cannot be removed and replaced by {old_name}, "
                f"shape0={node.args[0].meta['val'].shape}, "
                f"shape1={node.args[1].meta['val'].shape}, "
                f" in \n{graph}"
            )
            graph.erase_node(old_name)
            if verbose > 1:
                print(
                    f"[CustomTracer.remove_unnecessary_slices.2] replace "
                    f"{old_name!r} by {new_name!r}"
                )
            removed += 1
        return removed

    @classmethod
    def remove_batch_dim_nodes(cls, graph: torch.fx.Graph, verbose: int = 0) -> int:
        """
        Rewrites ``_add_batch_dim`` and ``_remove_batch_dim`` nodes introduced by
        ``torch.vmap`` lowering using a two-phase batch-dimension tracking strategy.

        **Phase 1 – tag:** ``_add_batch_dim(x, batch_dim, level)`` is replaced by
        ``aten.clone.default(x)`` and the actual batch-dimension position is stored
        in the replacement node's metadata under the key ``"vmap_batch_dim"`` as a
        ``dict`` mapping *level* → *batch_dim*.

        **Phase 2 – propagate:** For every ``call_function`` node whose input
        nodes carry ``"vmap_batch_dim"`` metadata, that metadata is copied onto
        the output node so that downstream consumers can find it.

        **Phase 3 – remove:** ``_remove_batch_dim(x, level, batch_size, out_dim)``
        looks up the actual batch-dimension position of *x* for the given *level*
        from the metadata (falling back to ``level - 1`` if absent), then:

        * If the actual size of the batch dimension in *x* is 1 but
          ``batch_size > 1`` (the tensor was broadcast / "not batched" inside vmap),
          an ``aten.expand.default`` node is emitted first to materialise the full
          batch.
        * erases the node (replacing all uses with *x*) when
          no expand is needed and ``actual_batch_dim == out_dim`` – a no-op case;
        * replaces the node with
          ``aten.movedim.int(x, actual_batch_dim, out_dim)`` otherwise.

        :param graph: FX graph to modify in-place
        :param verbose: verbosity level
        :return: number of nodes replaced or removed
        """
        from torch._functorch.predispatch import _add_batch_dim, _remove_batch_dim

        _VMAP_KEY = "vmap_batch_dim"  # meta key: Dict[level, actual_batch_dim]

        def _get_batch_dim(node: torch.fx.Node) -> Optional[Tuple[int]]:
            """Return the tracked batch-dim position for *level*, or None."""
            return node.meta.get(_VMAP_KEY, None)

        def _set_batch_dim(node: torch.fx.Node, batch_dim: int, level: int):
            """Store the batch-dim position for *level* in *node*'s metadata."""
            assert _VMAP_KEY not in node.meta or node.meta[_VMAP_KEY] == (batch_dim, level), (
                f"Incompatible hatc_dim dimension existing is "
                f"{node.meta[_VMAP_KEY]} != {(batch_dim, level)} (new)"
            )
            node.meta[_VMAP_KEY] = (batch_dim, level)

        def _propagate(src_nodes, dst_node: torch.fx.Node):
            """Copy vmap batch-dim metadata from any src node that carries it."""
            for src in src_nodes:
                if isinstance(src, torch.fx.Node):
                    batch = _get_batch_dim(src)
                    if batch:
                        _set_batch_dim(dst_node, *batch)

        modified = 0
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            target = node.target
            if target is _add_batch_dim:
                # args: (x, batch_dim, level)
                assert len(node.args) == 3, (
                    f"_add_batch_dim expected 3 args, got {len(node.args)}: {node.args} "
                    f"is it expected?"
                )
                x, batch_dim, level = node.args
                # Replace with clone(x) and tag with the actual batch-dim position.
                with graph.inserting_before(node):
                    new_node = graph.call_function(torch.ops.aten.clone.default, args=(x,))
                    x_shape = x.meta["val"].shape
                    node_shape = node.meta["val"].shape
                    if x_shape == node_shape:
                        new_node.meta = node.meta.copy()
                    else:
                        assert len(node_shape) == 0 and len(x_shape) == 1, (
                            f"{x_shape=} and {node_shape=} is not yet implmeented. "
                            f"You should raise an issue."
                        )
                        new_node = graph.call_function(
                            torch.ops.aten.squeeze.default, args=(new_node, 0)
                        )
                        new_node.meta = node.meta.copy()
                _set_batch_dim(new_node, batch_dim, level)
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)
                if verbose:
                    print(
                        f"[CustomTracer.remove_batch_dim_nodes] _add_batch_dim "
                        f"batch_dim={batch_dim} level={level} → clone + tag"
                    )
                modified += 1
            elif target is _remove_batch_dim:
                # args: (x, level, batch_size, out_dim)
                assert (
                    len(node.args) == 4
                ), f"_remove_batch_dim expected 4 args, got {len(node.args)}: {node.args}"
                x, level, batch_size, out_dim = node.args

                # Resolve the actual batch-dim position from tracked metadata.
                assert isinstance(
                    x, torch.fx.Node
                ), f"This is not implemented yet for type {type(x)}"
                x_batch, _x_level = _get_batch_dim(x)
                assert x_batch == out_dim, (
                    f"Not yet implemented when batch_dim(x)={x_batch} and {out_dim=}. "
                    f"You should file an issue."
                )

                with graph.inserting_before(node):
                    # Build expand shape: keep all dims as-is (-1) except
                    # the batch dim which is set to batch_size.
                    x_shape = x.meta["val"].shape
                    rank = len(x_shape)
                    assert rank > 1 or x_batch == 0, (
                        f"Incompatibilities between {x_shape=} and {x_batch=}. "
                        f"A case is not tested?"
                    )
                    if rank == 0:
                        rank = 1
                    expand_shape = [-1] * rank
                    expand_shape[x_batch] = batch_size
                    src = graph.call_function(
                        torch.ops.aten.expand.default, args=(x, expand_shape)
                    )
                    src.meta = node.meta.copy()
                    if x_batch != out_dim:
                        movedim_node = graph.call_function(
                            torch.ops.aten.movedim.int, args=(src, x_batch, out_dim)
                        )
                        movedim_node.meta = node.meta.copy()
                        src = movedim_node
                node.replace_all_uses_with(src)
                graph.erase_node(node)
                if verbose:
                    print(f"[CustomTracer.remove_batch_dim_nodes] _remove_batch_dim {src.name!r}")
                modified += 1
            else:
                # Propagate vmap batch-dim metadata through all other ops so that
                # _remove_batch_dim can find it even after several intervening nodes.
                # Skip quickly when no input node carries vmap metadata.
                if any(isinstance(a, torch.fx.Node) and _VMAP_KEY in a.meta for a in node.args):
                    _propagate(node.args, node)

        return modified

    @classmethod
    def graph_erase_node(cls, graph: torch.fx.Graph, node: torch.fx.Node):
        """
        Removes a node and all predecessors with are only consumed by this one.
        """
        nodes = [node]
        while (
            node.op == "call_function"
            and node.args
            and isinstance(node.args[0], torch.fx.Node)
            and all(isinstance(_, (int, float)) for _ in node.args[1:])
            and len(node.args[0].users) == 1
        ):
            node = node.args[0]
            nodes.append(node)
        for node in nodes:
            graph.erase_node(node)

    @classmethod
    def _modify_graph_clone_copy_(
        cls,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        existing_nodes: List[torch.fx.Node],
        pos: int,
        exported_program: torch.export.ExportedProgram,
        err_graph: str,
        verbose: int = 0,
        exc: bool = True,
    ) -> int:
        """
        Removes inplace node ``clone`` + ``copy_`` (inplace copy).
        Then we tell the nodes using ``%clone`` to use ``%copy_``.

        :param graph: graph to modify
        :param node: node after clone
        :param existing_nodes: list of the nodes in the graph
        :param pos: position of the node in existing nodes
        :param exported_program: for debugging purpose
        :param err_graph: original graph as a string, for debugging purpose
        :param verbose: verbosity
        :param exc: if False, return -1 if unable to reach the goal
        :return: number of removed nodes
        """
        predecessor = node.args[0]
        predecessor_name = (
            predecessor.target.name() if hasattr(predecessor.target, "name") else None
        )

        if not exc and predecessor_name != "aten::clone":
            return -1

        assert predecessor_name == "aten::clone", (
            f"(inplace) Unexpected predecessor {predecessor.target!r} "
            f"(predecessor.target.name()={predecessor_name!r}) "
            f"for node {node.name!r} with args={node.args} at position "
            f"{pos}/{len(graph.nodes)}"
            f"\n--original graph--\n{err_graph}"
            f"\n--graph\n{exported_program or graph}"
        )

        def delete_user_cb(n, nodes_to_leave):
            return n not in nodes_to_leave

        # class Node can be used as a key
        # We also assume a user is placed after this node.
        nodes_to_leave = {n[1] for n in existing_nodes[: pos + 1]}
        node_args = node.args
        p_users = predecessor.users

        # We can replace with expand then.
        with graph.inserting_before(node):
            # We assume the first argument is the one modified inplace.
            new_node = graph.call_method("expand_as", args=(node_args[1], predecessor))
            # let's replace
            changed = predecessor.replace_all_uses_with(
                new_node,
                delete_user_cb=(lambda n, leave=nodes_to_leave: delete_user_cb(n, leave)),
            )
            graph.erase_node(node)
            # new_node is replaced as well so we manually revert the replacement
            new_node.update_arg(1, predecessor)

        assert changed, (
            f"No change applied, the inplace node [{node}] "
            f"at position {pos} with node.args={node_args}, was not replaced "
            f"by [{new_node}] with target {new_node.target!r} and "
            f"new_node.args={new_node.args}, predecessor="
            f"[{predecessor}] with target={predecessor.target!r}, "
            f"p_users={list(p_users)}, "
            f"predecessor.users={list(predecessor.users)}, "
            f"new_node.users={list(new_node.users)} in "
            f"\n{exported_program or graph}"
        )
        return 1

    @classmethod
    def _modify_graph_clone_index_copy_(
        cls,
        graph: torch.fx.Graph,
        node: torch.fx.Node,
        existing_nodes: List[Tuple[int, torch.fx.Node]],
        pos: int,
        exported_program: torch.export.ExportedProgram,
        err_graph: str,
        verbose: int = 0,
        exc: bool = True,
    ) -> int:
        """
        Removes inplace node ``clone`` + ``index.Tensor`` + ``copy_`` (inplace copy).
        Then we tell the nodes using ``%clone`` to use ``%copy_``.

        :param graph: graph to modify
        :param node: node after clone
        :param existing_nodes: list of the nodes in the graph
        :param pos: position of the node in existing nodes
        :param exported_program: for debugging purpose
        :param err_graph: original graph as a string, for debugging purpose
        :param verbose: verbosity
        :param exc: raises an exception if cannot reach the goal
        :return: number of removed nodes
        """
        # Let's find the first clone node.
        given_node = node
        while (
            hasattr(node, "target")
            and node.target not in {"clone"}
            and (
                not hasattr(node.target, "name")
                or node.target.name() not in {"aten::clone", "aten::zeros", "aten::ones"}
            )
            and node.args
        ):
            if isinstance(node.args[0], torch.fx.immutable_collections.immutable_list):
                arg = node.args[0]
                while isinstance(arg, torch.fx.immutable_collections.immutable_list):
                    if verbose > 5:
                        print("+++", node, node.target)
                    # something like
                    # aten.zeros.default](args = ([%sym_size_int_18, %add_179, %add_179],)
                    if len(arg) == 0:
                        break
                    assert hasattr(arg[0], "args"), (
                        f"Unexpected value for arg={arg}, type is {type(arg[0])} "
                        f"node={node!r}, node.target={node.target}"
                    )
                    node = arg[0]
                    if not node.args:
                        break
                    arg = node.args[0]
            else:
                node = node.args[0]
        clone = node
        target_name = (
            (node.target.name() if hasattr(node.target, "name") else node.target)
            if hasattr(node, "target")
            else None
        )
        allowed_names = {"aten::clone", "clone", "aten::zeros", "zeros"}
        if not exc and target_name not in allowed_names:
            return -1
        assert target_name in allowed_names, (
            f"(inplace) Unexpected predecessor {node.target!r} "
            f"(target_name={target_name!r}) "
            f"for node {given_node.name!r} with args={given_node.args} at position "
            f"{pos}/{len(graph.nodes)}"
            f"\n--original graph--\n{err_graph}"
            f"\n--graph\n{exported_program or graph}"
        )

        # Let's find all the users until we find a node copy_
        # assuming this node has no users
        known = set()
        users = []
        unprocessed = [clone]
        while unprocessed:
            current = unprocessed.pop()
            if current.users:
                new_users = [u for u in current.users if u not in known]
                users.extend(new_users)
                unprocessed.extend(new_users)
                known |= set(new_users)

        def _str(pos_node):
            pos, node = pos_node
            return f"-- {pos} {node.args} -> {node.name} -- with {node.target}\n"

        # Let's reorder according to existing nodes
        def delete_user_cb(n, nodes_to_leave):
            return n not in nodes_to_leave

        def _macro_assert_index_(is_getitem=False):
            assert (
                not inplace_functions
            ), f"Unexpected inplace_functions={inplace_functions} before {n.target}"
            assert len(n.args) and n.args[0] in seen_nodes, (
                f"Unexpected node {n} at position {pos} (n.args={n.args}) "
                f"in {''.join(map(_str, pos_users))}\n"
                f"-----\nseen_nodes={seen_nodes}"
            )
            assert all(isinstance(_i, (tuple, int, torch.fx.Node)) for _i in n.args[1:]), (
                f"Unexpected arguments for target {n.target} "
                f"args={string_type(n.args)} - {n.args}\n--graph--\n{graph}"
            )
            assert not is_getitem or not set_item_args, (
                f"set_item_args should be empty for getitem at position {pos} "
                f"in {''.join(map(_str, pos_users))}\n"
                f"-----\nseen_nodes={seen_nodes}"
            )

        def _macro_get_axis_(n, set_item_args, exc=exc):
            axis = n.args[1]
            if axis in set_item_args:
                assert not exc, (
                    f"duplicated axis {n.args[1]} (n={n!r}, n.args={n.args}, "
                    f"set_item_args={set_item_args}) in \n"
                    f"\nnode.meta=\n{node.meta}\n---\n"
                    f"{''.join(map(_str, pos_users))}\n---\n"
                    f"{graph}"
                )
                return LEAVE_INPLACE
            return axis

        def _macro_new_node_(
            n, current_remove, set_item_args, inplace_functions, verbose=verbose
        ):
            # We need to replace all the nodes in seen_nodes
            # with a set_item and make clone is now replaced by this new one
            assert len(n.args) and n.args[0] in seen_nodes, (
                f"Unexpected node {n} at position {pos} "
                f"in {''.join(map(_str, pos_users))}\n"
                f"-----\nseen_nodes={seen_nodes}\n----{err_graph}"
            )
            seen_nodes.add(n)
            current_remove.append(n)
            max_axis = max(set_item_args)
            args_slices = [None for i in range(max_axis + 1)]
            for k, v in set_item_args.items():
                args_slices[k] = v
            new_args = [clone, tuple(args_slices), *n.args[1:]]

            nodes_to_leave = {_n[1] for _n in existing_nodes[: pos + 1]}

            # We use operator.setitem in both cases.
            # torch.ops.aten.fill.Tensor cannot work in this case.
            if not inplace_functions:
                function_op = operator.setitem
            else:
                function_op = setitem_with_transformation
                new_args.append(tuple(inplace_functions))

            # We can replace with expand then.
            with graph.inserting_after(n):
                if verbose >= 5:
                    print(
                        f"[CustomTracer._modify_graph_clone_index_copy_] "
                        f"insert after {n}<-{n.args}"
                    )
                # We assume the first argument is the one modified inplace.
                new_node = graph.call_function(function_op, args=tuple(new_args))
                nodes_to_leave.add(new_node)
                # let's replace
                if verbose >= 5:
                    print(
                        f"[CustomTracer._modify_graph_clone_index_copy_] "
                        f"replace with {new_node}<-{n.args}"
                    )
                changed = clone.replace_all_uses_with(
                    new_node,
                    delete_user_cb=(lambda n, leave=nodes_to_leave: delete_user_cb(n, leave)),
                )
                assert changed, (
                    f"No change applied, for new_node={new_node} and "
                    f"args={new_node.args} in {''.join(map(_str, pos_users))}"
                    f"\n-- new graph\n{graph}"
                    f"\n-- old graph\n{err_graph}"
                )
                return new_node

        node_pos = {b: a for a, b in existing_nodes}
        pos_users = [(node_pos[n], n) for n in users]
        pos_users.sort()

        to_remove = []
        seen_nodes = {clone}
        set_item_args = {}
        current_remove = []
        inplace_functions = []
        for pos, n in pos_users:
            if n.target == operator.getitem:
                _macro_assert_index_(True)
                set_item_args = dict(enumerate(n.args[1]))
                seen_nodes.add(n)
                current_remove.append(n)
            elif hasattr(n.target, "name"):
                aten_name = n.target.name()
                if aten_name == "aten::slice.Tensor":
                    _macro_assert_index_()
                    axis = _macro_get_axis_(n, set_item_args)
                    if axis is LEAVE_INPLACE:
                        return -1
                    set_item_args[axis] = slice(*n.args[2:])
                    seen_nodes.add(n)
                    current_remove.append(n)
                elif aten_name == "aten::select.int":
                    if not n.args or n.args[0] not in seen_nodes:
                        # Cannot handle this.
                        return -1
                    _macro_assert_index_()
                    axis = _macro_get_axis_(n, set_item_args)
                    if axis is LEAVE_INPLACE:
                        return -1
                    set_item_args[axis] = n.args[2]
                    seen_nodes.add(n)
                    current_remove.append(n)
                elif aten_name[-1] != "_" and "_." not in aten_name:
                    # This is no inplace modification so all stored
                    # slice operator are cleaned.
                    set_item_args = {}
                    current_remove = []
                    seen_nodes = {clone}
                    inplace_functions = []
                elif aten_name in {"aten::copy_", "aten::fill_.Tensor"}:
                    new_node = _macro_new_node_(
                        n, current_remove, set_item_args, inplace_functions
                    )
                    # next root to use
                    clone = new_node
                    # reset
                    to_remove.extend(current_remove)
                    seen_nodes = {new_node}
                    set_item_args = {}
                    current_remove = []
                    inplace_functions = []
                elif aten_name == "aten::masked_fill_.Scalar":
                    return -1
                else:
                    raise NotImplementedError(
                        f"Unable to handle target {aten_name!r} with args={n.args} "
                        f"in\n{''.join(map(_str, pos_users))}\n----\n{err_graph}"
                    )
            elif n.target in {torch.exp_, torch.sigmoid_} or (
                isinstance(n.target, str) and n.target.endswith("_")
            ):
                # One inplace modification.
                # We assume the inplace modification takes place instead of a copy.
                assert (
                    n.args
                    and isinstance(n.args[0], torch.fx.Node)
                    and all(not isinstance(_, torch.fx.Node) for _ in n.args[1:])
                ), (
                    f"Unexpected type in argument of node {n} at position "
                    f"{pos}(args={string_type(n.args)})"
                )
                function_name = {torch.exp_: "exp", torch.sigmoid_: "sigmoid"}[n.target]
                inplace_functions.append((function_name, n.args[1:]))
                # do the same as before
                new_node = _macro_new_node_(n, current_remove, set_item_args, inplace_functions)
                # next root to use
                clone = new_node
                # reset
                to_remove.extend(current_remove)
                seen_nodes = {new_node}
                set_item_args = {}
                current_remove = []
                inplace_functions = []
            else:
                # Here this node should already been handled.
                # We skip it.
                assert pos == pos_users[-1][0], (
                    f"Unexpected node (1) at pos={pos}, node={n}, target={n.target!r} "
                    f"in {''.join(map(_str, pos_users))}"
                )
        else:
            # Here again this node should already been handled.
            # We skip it.
            assert pos == pos_users[-1][0], (
                f"Unexpected node (2) at pos={pos}, node={n}, target={n.target} "
                f"in {''.join(map(_str, pos_users))}"
            )

        # Let's replace the replaced nodes.
        for n in reversed(to_remove):
            graph.erase_node(n)

        # The last
        assert len(to_remove) == len(pos_users) - 1, (
            "Some nodes were not properly handled "
            f"len(to_remove)={len(to_remove)} and "
            f"len(pos_users)={len(pos_users)},\n-- nodes\n"
            f"{''.join(map(_str, pos_users))}\n"
            f"-- new graph\n{graph}"
            f"\n-- old graph\n{err_graph}"
        )
        return len(to_remove)

    @classmethod
    def get_node_target_name(cls, node, exc: bool = True):
        res = (
            node.target.name()
            if hasattr(node.target, "name")
            else (
                node.target
                if isinstance(node.target, str)
                else (
                    f"{node.target.__module__}.{node.target.__name__}"
                    if callable(node.target)
                    else None
                )
            )
        )
        assert res is not None, (
            f"Unable to guess the target node from type {type(node.target)}, "
            f"node.target={node.target}, name={node.name!r}, node.args={node.args} "
        )
        if res.startswith("_operator."):
            return res[1:]
        return res

    @classmethod
    def _remove_inplace(
        cls,
        exported_program,
        graph,
        MAX_ITER=100,
        verbose: int = 0,
        exc: bool = True,
        err_graph: str = "",
    ) -> int:
        inplace = cls._inplace_nodes(graph)
        n_inplace = len(inplace)
        if n_inplace == 0:
            return 0

        def delete_user_cb(n, nodes_to_leave):
            return n not in nodes_to_leave

        if not err_graph:
            err_graph = str(graph)

        max_iter = MAX_ITER
        if verbose:
            print(
                f"[CustomTracer.remove_inplace] S2: {len(inplace)} inplace nodes "
                f"and {max_iter} iterations"
            )
            if verbose > 1:
                for _, __ in zip(range(3), inplace):
                    print(f"   {__[0]}: {__[1]}: {__[1].target}({__[1].args})")
        while inplace and max_iter > 0:
            if verbose > 1:
                print(
                    f"[CustomTracer.remove_inplace] loop {max_iter} "
                    f"iterations left with {len(graph.nodes)} nodes and "
                    f"{len(inplace)} inplace nodes"
                )
            existing_nodes = list(enumerate(graph.nodes))
            for pos, node in reversed(inplace):

                if verbose > 5:
                    print(
                        f"[CustomTracer.remove_inplace] handle inplace node "
                        f"{pos}/{len(graph.nodes)}: {node} with args={node.args} "
                        f"and target={node.target}"
                    )

                # if the target has a name
                node_target_name = cls.get_node_target_name(node, False)
                assert node_target_name is not None, (
                    f"Unable to guess the target node from type {type(node.target)}, "
                    f"node.target={node.target}, name={node.name!r}, node.args={node.args} "
                    f"at position {pos}/{len(graph.nodes)}"
                    f"\n--original graph--\n{err_graph}"
                    f"\n--graph\n{exported_program or graph}"
                )

                if node_target_name in {
                    "add_",
                    "div_",
                    "mul_",
                    "mod_",
                    "sub_",
                    "torch.exp_",
                    "torch.sigmoid_",
                    "operator.setitem",
                }:

                    # We still need to check the predecessor.
                    predecessor_name = cls.get_node_target_name(node.args[0])
                    if predecessor_name in {
                        "aten::slice.Tensor",
                        "aten::select.int",
                        "operator.getitem",
                    }:
                        # We face a schema such as
                        # K_33[2:-2, 2:-2, :-1] = sumx[None, 2:-2, None]
                        do_break = cls._modify_graph_clone_index_copy_(
                            graph,
                            node,
                            existing_nodes,
                            pos,
                            exported_program,
                            err_graph,
                            verbose=verbose,
                            exc=exc,
                        )
                        if do_break == -1:
                            if verbose:
                                print(
                                    f"[CustomTracer.remove_inplace] "
                                    f"unable to remove (3) {node.target}"
                                )
                            return -1
                        if do_break:
                            break
                        continue

                    # We assume the first argument is the one modified inplace.
                    new_name = node
                    old_name = node.args[0]

                    if verbose > 2:
                        print(
                            f"[CustomTracer.remove_inplace] D.process {pos}: "
                            f"{node.target}({node.args}) -> {node}"
                        )

                    # class Node can be used as a key
                    # We also assume a user is placed after this node.
                    nodes_to_leave = {n[1] for n in existing_nodes[: pos + 1]}

                    # let's replace
                    changed = old_name.replace_all_uses_with(
                        new_name,
                        delete_user_cb=(lambda n, leave=nodes_to_leave: delete_user_cb(n, leave)),
                    )

                    assert changed, (
                        f"No change applied, the inplace node [{node}] at position {pos} "
                        f"does not replace [{old_name}] in \n{graph}\n-- node to keep --"
                        f"\n{nodes_to_leave}"
                    )
                    continue

                if node_target_name in {
                    "aten::view",
                    "aten::detach_",
                    "aten::add.Tensor",
                    "aten::div.Tensor",
                    "aten::mul.Tensor",
                    "aten::sub.Tensor",
                    "aten::zeros",
                    "expand",
                    "aten::__and__.Tensor",
                } or not (node_target_name.endswith(("_", "_.Tensor"))):
                    # This node cannot be one inplace modification.
                    # The node is just not used.
                    cls.graph_erase_node(graph, node)
                    del existing_nodes[pos]
                    if verbose > 2:
                        print(
                            f"[CustomTracer.remove_inplace] B1.remove "
                            f"{pos}: {node.target}({node.args}) -> {node}"
                        )
                    continue

                if (
                    node_target_name
                    in {
                        "aten::add_.Tensor",
                        "aten::mul_.Tensor",
                        "aten::div_.Tensor",
                        "aten::sub_.Tensor",
                    }
                    and isinstance(node.args[0], torch.fx.Node)
                    and not isinstance(node.args[1], torch.fx.Node)
                    and len(node.args[0].users) <= 1
                ):
                    cls.graph_erase_node(graph, node)
                    del existing_nodes[pos]
                    if verbose > 2:
                        print(
                            f"[CustomTracer.remove_inplace] B2.remove "
                            f"{pos}: {node.target}({node.args}) -> {node}"
                        )
                    continue

                if len(node.args) == 1:
                    predecessor = node.args[0]
                    if len(predecessor.users) == 0:
                        cls.graph_erase_node(graph, node)
                        del existing_nodes[pos]
                        if verbose > 2:
                            print(
                                f"[CustomTracer.remove_inplace] B3.remove "
                                f"{pos}: {node.target}({node.args}) -> {node}"
                            )
                        continue

                if not exc and node_target_name in {"aten::index_put_"}:
                    if verbose:
                        print(
                            f"[CustomTracer.remove_inplace] "
                            f"unable to remove (9) {node_target_name!r}"
                        )
                    return -1
                assert (
                    node_target_name in {"aten::copy_", "aten::fill_.Tensor"}
                    and len(node.args) == 2
                ) or node_target_name in {"aten::sigmoid_"}, (
                    f"(inplace) Unsupported target {node.target!r}, target_name="
                    f"{node_target_name!r}, name={node.name!r}, node.args={node.args} "
                    f"at position {pos}/{len(graph.nodes)}"
                    f"\n--original graph--\n{err_graph}"
                    f"\n--graph\n{exported_program or graph}"
                )

                # We check the predecessor if the node is a node copy_.
                predecessor = node.args[0]
                predecessor_name = cls.get_node_target_name(predecessor)
                if predecessor_name in {"aten::slice.Tensor", "aten::select.int"}:
                    if verbose > 5:
                        print(
                            f"[CustomTracer.remove_inplace] _modify_graph_clone_index_copy_ "
                            f"{pos}/{len(graph.nodes)}: {node} with args={node.args} "
                            f"and target={node.target}"
                        )
                    do_break = cls._modify_graph_clone_index_copy_(
                        graph,
                        node,
                        existing_nodes,
                        pos,
                        exported_program,
                        err_graph,
                        verbose=verbose,
                        exc=exc,
                    )
                    if do_break == -1:
                        if verbose:
                            print(
                                f"[CustomTracer.remove_inplace] "
                                f"unable to remove (1) {node_target_name!r}"
                            )
                        return -1
                    if do_break:
                        break
                    continue

                if verbose > 5:
                    print(
                        f"[CustomTracer.remove_inplace] _modify_graph_clone_copy_ "
                        f"{pos}/{len(graph.nodes)}: {node} with args={node.args} "
                        f"and target={node.target}"
                    )

                do_break = cls._modify_graph_clone_copy_(
                    graph,
                    node,
                    existing_nodes,
                    pos,
                    exported_program,
                    err_graph,
                    verbose=verbose,
                    exc=exc,
                )
                if do_break == -1:
                    if verbose:
                        print(
                            f"[CustomTracer.remove_inplace] "
                            f"unable to remove (2) {node_target_name!r}"
                        )
                    return -1
                if do_break:
                    break

            if verbose > 5:
                print(f"[CustomTracer.remove_inplace] continue with {node_target_name!r}")
            inplace = cls._inplace_nodes(graph)
            if len(inplace) == 0:
                break
            max_iter -= 1

        if verbose:
            print(
                f"[CustomTracer.remove_inplace] end with {max_iter} "
                f"iterations and {len(graph.nodes)} nodes (n_inplace={n_inplace})"
            )
        return n_inplace

    @classmethod
    def _replace_inplace_call_methods(cls, graph: torch.fx.Graph, verbose: int = 0) -> int:
        """
        Replaces ``call_method`` nodes with inplace targets (e.g. ``add_``, ``mul_``,
        ``sub_``, ``div_``) with their non-inplace ``call_function`` equivalents.

        For each such node, the first argument is the tensor modified in-place.
        After conversion, all uses of that first argument that appear *after*
        the current node are replaced with the new non-inplace result, preserving
        the original in-place semantics.

        :param graph: graph to modify
        :param verbose: verbosity level
        :return: number of nodes replaced
        """
        _method_to_aten = {
            "add_": torch.ops.aten.add.Tensor,
            "mul_": torch.ops.aten.mul.Tensor,
            "sub_": torch.ops.aten.sub.Tensor,
            "div_": torch.ops.aten.div.Tensor,
        }
        n = 0
        # Build a position map for O(1) position lookups inside the loop.
        position_map = {node: pos for pos, node in enumerate(graph.nodes)}
        existing_nodes = list(enumerate(graph.nodes))
        for pos, node in existing_nodes:
            if (
                node.op != "call_method"
                or not isinstance(node.target, str)
                or node.target not in _method_to_aten
            ):
                continue
            old_first_arg = (
                node.args[0] if node.args and isinstance(node.args[0], torch.fx.Node) else None
            )
            old_target = node.target
            node.op = "call_function"
            node.target = _method_to_aten[old_target]
            n += 1
            if old_first_arg is not None:
                # Only replace uses of old_first_arg that appear *after* the inplace node.
                # Users at or before pos (including node itself) are left untouched so that
                # they continue to reference the pre-modification value, which preserves
                # correct in-place semantics.
                if any(position_map.get(user, -1) > pos for user in old_first_arg.users):
                    old_first_arg.replace_all_uses_with(
                        node, delete_user_cb=lambda user, p=pos: position_map.get(user, -1) > p
                    )
            if verbose > 1:
                print(
                    f"[CustomTracer._replace_inplace_call_methods] replaced "
                    f"call_method {old_target!r} -> aten at {node.name!r}"
                )
        return n

    @classmethod
    def remove_inplace(
        cls,
        graph: torch.fx.Graph,
        exported_program: Optional[torch.export.ExportedProgram] = None,
        verbose: int = 0,
        exc: bool = True,
        recursive: bool = False,
    ) -> int:
        """
        Removes inplace operations.

        :param graph: graph to modify
        :param exported_program: if available, it is used in the error message
            to make it easier to trace the code source
        :param verbose: verbosity
        :param exc: raise an exception if not possible, other return -1
        :param recursive: remove node inside submodules
        :return: number of inplace nodes removed, a negative number means
            there are still inplace nodes to be removed but this
            function is unable to do that, only decompositions
            may help in that case
        """
        n_inplace_submobules = 0
        if recursive:
            for node in graph.nodes:
                if node.op == "get_attr":
                    init = getattr(node.graph.owning_module, node.target)
                    if not hasattr(init, "graph"):
                        continue
                    if verbose:
                        print(f"[CustomTracer.remove_inplace] submodule {node.name!r} ...")
                    inpl = cls.remove_inplace(
                        init.graph, verbose=verbose, exc=exc, recursive=recursive
                    )
                    if verbose:
                        print(f"[CustomTracer.remove_inplace] submodule {node.name!r} done")
                    if inpl < 0:
                        return inpl
                    n_inplace_submobules += inpl

        # Convert call_method inplace nodes (add_, mul_, etc.) to call_function equivalents.
        cls._replace_inplace_call_methods(graph, verbose=verbose)

        # Remove obvious unused nodes.
        rem = []
        for node in graph.nodes:
            if (
                not node.users
                and hasattr(node, "target")
                and node.target
                in {
                    torch._C._set_grad_enabled,
                    torch._C._log_api_usage_once,
                    torch.autograd.function.FunctionCtx,
                    torch.amp.autocast_mode._enter_autocast,
                    torch.amp.autocast_mode._exit_autocast,
                    torch.ops.aten._assert_scalar.default,
                    torch.ops.aten._assert_tensor_metadata.default,
                }
            ):
                rem.append(node)
        for node in reversed(rem):
            graph.erase_node(node)
            if verbose > 2:
                print(f"[CustomTracer.remove_inplace] 0.remove {node.target}")
            continue

        # True inplace nodes.
        inplace = cls._inplace_nodes(graph)
        if len(inplace) == 0:
            return n_inplace_submobules

        def delete_user_cb(n, nodes_to_leave):
            return n not in nodes_to_leave

        if verbose:
            print(
                f"[CustomTracer.remove_inplace] starts with {len(graph.nodes)} "
                f"nodes (n_inplace_submobules={n_inplace_submobules})"
            )

        n_inplace = len(inplace)
        if verbose:
            print(f"[CustomTracer.remove_inplace] S1: {len(inplace)} inplace nodes")
            if verbose > 1:
                for _, __ in zip(range(3), inplace):
                    print(f"   {__[0]}: {__[1]}: {__[1].target}({__[1].args})")
        changed = cls._replace_getattr(graph)
        changed |= cls._replace_meth_setitem(graph)

        err_graph = str(graph)
        if changed:
            cls._inplace_nodes(graph)

        if len(inplace) == 0:
            return n_inplace_submobules

        # First step: we remove every unused node.
        while True:
            removed = 0
            for pos, node in reversed(inplace):
                if node.target in {
                    operator.add,
                    operator.floordiv,
                    operator.mul,
                    operator.mod,
                    operator.sub,
                    operator.le,
                    operator.ge,
                    operator.eq,
                    torch._C._set_grad_enabled,
                    torch._C._log_api_usage_once,
                    torch.autograd.function.FunctionCtx,
                    torch.amp.autocast_mode._enter_autocast,
                    torch.amp.autocast_mode._exit_autocast,
                    torch.ops.aten._assert_scalar.default,
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.item.default,
                    torch.ops.aten.sym_constrain_range_for_size,
                }:
                    graph.erase_node(node)
                    removed += 1
                    if verbose > 2:
                        print(
                            f"[CustomTracer.remove_inplace] A.remove "
                            f"{pos}: {node.target}({node.args}) -> {node}"
                        )
                    continue
            if removed:
                inplace = cls._inplace_nodes(graph)
            else:
                break

        if len(inplace) == 0:
            return n_inplace_submobules

        # Then the difficult ones, we first operate on a copy to avoid
        # breaking the consistency of the graph.
        graph_copy = graph.__class__(
            tracer_cls=graph._tracer_cls, tracer_extras=graph._tracer_extras
        )
        _vmap = {}
        out = graph_copy.graph_copy(graph, _vmap)
        graph_copy.output(out)
        assert len(graph_copy.nodes) == len(
            graph.nodes
        ), f"Graph copy did not work: {len(graph_copy.nodes)} != {len(graph.nodes)}"
        result = cls._remove_inplace(
            exported_program, graph_copy, verbose=verbose, exc=exc, err_graph=err_graph
        )
        if result < 0:
            assert not exc, f"Unable to remove all inline nodes in\n{err_graph}"
            return result
        if result == 0:
            return result + n_inplace_submobules
        inplace = cls._inplace_nodes(graph_copy)
        if len(inplace) > 0 and not exc:
            return -len(inplace)
        assert len(inplace) == 0, (
            f"Inplace nodes remain at positions {sorted(inplace)}"
            f"/{len(graph.nodes)} in\n{graph}\n--original graph--\n{err_graph}"
        )
        # It worked, we put the modified nodes back into the original graph.
        _vmap = {}
        graph.__init__(
            owning_module=graph.owning_module,
            tracer_cls=graph._tracer_cls,
            tracer_extras=graph._tracer_extras,
        )
        out = graph.graph_copy(graph_copy, _vmap)
        graph.output(out)
        return n_inplace + n_inplace_submobules
