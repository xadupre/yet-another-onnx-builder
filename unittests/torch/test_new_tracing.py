"""Tests for yobx.torch.new_tracing.

This file is kept for backward compatibility. Tests have been split into:
  - test_new_tracing_shapes.py  – TracingInt, TracingBool, TracingShape
  - test_new_tracing_tensor.py  – TracingTensor
  - test_new_tracing_tracing.py – DispatchTracer and trace_model
"""

from unittests.torch.test_new_tracing_shapes import TestNewTracingShapes
from unittests.torch.test_new_tracing_tensor import TestNewTracingTensor
from unittests.torch.test_new_tracing_tracing import TestNewTracingTracing

__all__ = ["TestNewTracingShapes", "TestNewTracingTensor", "TestNewTracingTracing"]

import unittest

if __name__ == "__main__":
    unittest.main(verbosity=2)
