from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.fx
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
        tracer: Optional["GraphTracer"] = None,  # type: ignore # noqa: F821
    ):
        self._tracer = tracer
        self._node: Optional[torch.fx.Node] = None
        self._tracing_shape = TracingShape(size)  # type: ignore

    @property
    def shape(self) -> TracingShape:  # type: ignore
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
            return torch.empty(tuple(self.shape), dtype=self.dtype, device=self.device)
        new_shape = []
        for s in self.shape:
            if isinstance(s, int):
                new_shape.append(s)
            elif isinstance(s, str):
                assert s in dyanmic_shape_values, f"Missing value for dynamic dimension {s!r}"
                new_shape.append(dyanmic_shape_values[s])
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
        tracer = next(a._tracer for a in args if isinstance(a, TracingTensor))
        assert tracer, f"Missing tracer for func={func}, types={types}, {args}, {kwargs=}"
        return tracer.dispatch(func, args, kwargs)
