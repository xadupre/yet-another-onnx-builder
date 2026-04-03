"""
New tracing mechanism for PyTorch models using ``__torch_dispatch__``.

This module re-exports all public names from the three implementation
submodules for backward compatibility:

- :mod:`.shape` — :class:`TracingBool`, :class:`TracingInt`,
  :data:`TracingDimension`, :class:`TracingShape`.
- :mod:`.tensor` — :class:`TracingTensor`.
- :mod:`.dispatcher` — :class:`DispatchTracer`, :func:`trace_model`.

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

from .dispatcher import DispatchTracer, trace_model
from .shape import TracingBool, TracingDimension, TracingInt, TracingShape
from .tensor import TracingTensor

__all__ = [
    "DispatchTracer",
    "TracingBool",
    "TracingDimension",
    "TracingInt",
    "TracingShape",
    "TracingTensor",
    "trace_model",
]
