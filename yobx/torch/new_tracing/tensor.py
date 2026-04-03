"""
Tensor-related classes for dispatch-level tracing.

Defines :class:`TracingTensor`, a :class:`torch.Tensor` subclass that records
all dispatch-level operations into a :class:`torch.fx.Graph`.
"""

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
    * ``_tracing_shape``: The symbolic :class:`TracingShape` when any
      dimension is dynamic, or ``None`` for fully concrete tensors.
      The concrete PyTorch ``shape`` attribute uses ``1`` as a stand-in for
      symbolic dimensions, so downstream code that needs the symbolic form
      should read ``_tracing_shape`` instead.  The tensor ``dtype`` is always
      concrete and is accessible directly via ``self.dtype``.
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
        self._tracer: Optional[Any] = None
        self._node: Optional[torch.fx.Node] = None
        # Symbolic shape (TracingShape) if any dimension is dynamic; None otherwise.
        self._tracing_shape: Optional["TracingShape"] = None

    def __repr__(self) -> str:  # type: ignore
        node_name = self._node.name if self._node is not None else "<unregistered>"
        shape_repr = (
            repr(self._tracing_shape)
            if self._tracing_shape is not None
            else repr(tuple(self.shape))
        )
        return (
            f"TracingTensor(node={node_name!r}, shape={shape_repr}, "
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

        # Locate the DispatchTracer from the first TracingTensor leaf.
        tracer: Optional[Any] = None
        all_inputs = (*args, *kwargs.values())
        for leaf in pytree.tree_leaves(all_inputs):
            if isinstance(leaf, TracingTensor) and leaf._tracer is not None:
                tracer = leaf._tracer
                break

        if tracer is None:
            # No tracer found — fall back to the default behaviour.
            return NotImplemented

        return tracer._dispatch(func, args, kwargs)
