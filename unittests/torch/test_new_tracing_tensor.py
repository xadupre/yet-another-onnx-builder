"""Tests for yobx.torch.new_tracing – TracingTensor."""

import unittest
import torch
from yobx.ext_test_case import ExtTestCase, requires_torch


@requires_torch("2.0")
class TestNewTracingTensor(ExtTestCase):
    def test_tracing_tensor_creation(self):
        from yobx.torch.new_tracing.tensor import TracingTensor

        t = TracingTensor.__new__(TracingTensor, (3, 4), dtype=torch.float32)
        t.__init__((3, 4), dtype=torch.float32)
        self.assertEqual(t.shape, torch.Size([3, 4]))
        self.assertEqual(t.dtype, torch.float32)

    def test_tracing_tensor_repr(self):
        from yobx.torch.new_tracing.dispatcher import DispatchTracer
        from yobx.torch.new_tracing.shape import TracingShape

        tracer = DispatchTracer()
        x = torch.randn(2, 3)
        t = tracer.placeholder(
            "x", TracingShape(tuple(int(i) for i in x.shape)), x.dtype, x.device
        )
        self.assertIn("TracingTensor", repr(t))
        self.assertIn("x", repr(t))


if __name__ == "__main__":
    unittest.main(verbosity=2)
