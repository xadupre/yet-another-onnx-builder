"""
Public API for :mod:`yobx.torch.new_tracing`.

New dispatch-level tracing for PyTorch models.  Produces
:class:`torch.fx.Graph` by intercepting tensor operations via
``__torch_dispatch__``, similar to how
:class:`torch._subclasses.FakeTensor` works.

The implementation is split across three submodules:

- :mod:`._shape` — shape/dimension types
  (:class:`TracingBool`, :class:`TracingInt`, :data:`TracingDimension`,
  :class:`TracingShape`).
- :mod:`._tensor` — :class:`TracingTensor`.
- :mod:`._dispatcher` — :class:`DispatchTracer` and :func:`trace_model`.
"""

from ._dispatcher import DispatchTracer, trace_model
from ._shape import TracingBool, TracingDimension, TracingInt, TracingShape
from ._tensor import TracingTensor

__all__ = [
    "DispatchTracer",
    "TracingBool",
    "TracingDimension",
    "TracingInt",
    "TracingShape",
    "TracingTensor",
    "trace_model",
]
