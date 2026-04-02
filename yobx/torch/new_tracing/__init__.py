"""
Public API for :mod:`yobx.torch.new_tracing`.

New dispatch-level tracing for PyTorch models.  Produces
:class:`torch.fx.Graph` by intercepting tensor operations via
``__torch_dispatch__``, similar to how
:class:`torch._subclasses.FakeTensor` works.

Exported names
--------------

.. autosummary::

    TracingInt
    TracingBool
    TracingShape
    TracingTensor
    DispatchTracer
    trace_model
"""

from .tracing import (
    DispatchTracer,
    TracingBool,
    TracingDimension,
    TracingInt,
    TracingShape,
    TracingTensor,
    trace_model,
)

__all__ = [
    "DispatchTracer",
    "TracingBool",
    "TracingDimension",
    "TracingInt",
    "TracingShape",
    "TracingTensor",
    "trace_model",
]
