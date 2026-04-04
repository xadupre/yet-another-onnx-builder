"""Tests for yobx.torch.new_tracing – TracingTensor."""

import unittest
import torch
from yobx.ext_test_case import ExtTestCase, requires_torch
from yobx.torch.new_tracing.shape import TracingShape
from yobx.torch.new_tracing.tensor import TracingTensor
from yobx.torch.new_tracing.tracer import GraphTracer


@requires_torch("2.0")
class TestNewTracingTensor(ExtTestCase):
    def test_tracing_tensor_creation(self):
        t = TracingTensor.__new__(TracingTensor, (3, 4), dtype=torch.float32)
        t.__init__((3, 4), dtype=torch.float32)
        self.assertEqual(t.shape, torch.Size([3, 4]))
        self.assertEqual(t.dtype, torch.float32)

    def test_tracing_tensor_repr(self):
        tracer = GraphTracer()
        x = torch.randn(2, 3)
        t = tracer.placeholder(
            "x", TracingShape(tuple(int(i) for i in x.shape)), x.dtype, x.device
        )
        self.assertIn("TracingTensor", repr(t))
        self.assertIn("x", repr(t))

    def test_tracing_tensor_from_tensor(self):
        t = torch.rand((3, 4), dtype=torch.float16)
        tt = TracingTensor.from_tensor(t)
        self.assertIsInstance(tt.shape, TracingShape)
        self.assertEqual(tt.shape, (3, 4))
        self.assertEqual(tt.dtype, torch.float16)
        t2 = tt.make_empty_instance()
        self.assertEqual(t2.dtype, t.dtype)
        self.assertEqual(t2.shape, t.shape)

    def test_tracing_tensor_from_tensor_dynamic(self):
        t = torch.rand((3, 4), dtype=torch.float16)
        tt = TracingTensor.from_tensor(t, dynamic_shapes={0: "batch"})
        self.assertIsInstance(tt.shape, TracingShape)
        self.assertEqual(tt.shape, ("batch", 4))
        self.assertEqual(tt.dtype, torch.float16)
        t2 = tt.make_empty_instance({"batch": 2})
        self.assertEqual(t2.dtype, t.dtype)
        self.assertEqual(t2.shape, (2, 4))

    def test_tracing_tensor_add(self):
        tracer = GraphTracer()
        t = torch.rand((3, 4), dtype=torch.float16)
        tt1 = TracingTensor.from_tensor(t, dynamic_shapes={0: "batch"}, tracer=tracer)
        tt2 = TracingTensor((1, 4), dtype=torch.float16, tracer=tracer)
        tt = tt1 + tt2
        self.assertEqual(tt.dtype, torch.float16)
        self.assertEqual(tt.shape, ("batch", 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
