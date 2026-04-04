from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.fx
import torch.utils._pytree as pytree
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
        tracer: Optional["GraphTracer"] = None,
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
        else:
            assert isinstance(size, tuple), f"Unexpected type {type(size)} for size"
            assert not size or all(
                isinstance(i, (int, str)) for i in size
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
        tracer: Optional["GraphTracer"] = None,
    ):
        self._tracer = tracer
        self._node: Optional[torch.fx.Node] = None
        self._tracing_shape = TracingShape(size)
        self._node: Optional[torch.fx.Graph] = None

    @property
    def shape(self) -> TracingShape:
        """Returns the shape as a TracingShape."""
        return self._tracing_shape

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
        dynamic_shapes: Optional[Dict[int, str]] = None,
        tracer: Optional["GraphTracer"] = None,
    ) -> "TracingTensor":
        """Creates a tracing tensor."""
        if not dynamic_shapes:
            return TracingTensor(tuple(int(i) for i in t.shape), t.dtype, t.device, tracer=tracer)
        shape = [int(i) for i in t.shape]
        for d, name in dynamic_shapes.items():
            shape[d] = name
        return TracingTensor(tuple(shape), t.dtype, t.device, tracer=tracer)

    def make_empty_instance(
        self, dyanmic_shape_values: Optional[Dict[str, int]] = None
    ) -> torch.Tensor:
        """
        Returns an instance of this tracing tensor
        with the same type, shape, device.
        """
        if not dyanmic_shape_values:
            return torch.empty(tuple(self.shape), dtype=self.dtype, device=self.device)
        new_shape = []
        for s in self.shape:
            if isinstance(s, int):
                new_shape.append(s)
            elif isinstance(s, str):
                assert s in dyanmic_shape_values, f"Missing value for dynamic dimension {s!r}"
                new_shape.append(dyanmic_shape_values[s])
            else:
                raise NotImplementedError(f"Unexected type {type(s)}")
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
        kwargs = kwargs or {}

        # infinite loop here, we create FakeTensor and we call the op.
        res = tracer._dispatch(func, args, kwargs)
        assert (
            func not in {torch.ops.aten.split.Tensor} or res is not None
        ), f"res is None but func is {func}, this is not possible, args={args}, kwargs={kwargs}"
        return res
