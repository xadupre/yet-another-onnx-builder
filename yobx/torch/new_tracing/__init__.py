"""
Public API for :mod:`yobx.torch.new_tracing`.

New dispatch-level tracing for PyTorch models.  Produces
:class:`torch.fx.Graph` by intercepting tensor operations via
``__torch_dispatch__``, similar to how
:class:`torch._subclasses.FakeTensor` works.

Exported names
--------------

.. autosummary::

    TracingDimension
    TracingShape
    TracingTensor
    DispatchTracer
    trace_model
"""

from .tracing import (
    DispatchTracer,
    TracingDimension,
    TracingShape,
    TracingTensor,
    trace_model,
)

__all__ = [
    "DispatchTracer",
    "TracingDimension",
    "TracingShape",
    "TracingTensor",
    "trace_model",
]
