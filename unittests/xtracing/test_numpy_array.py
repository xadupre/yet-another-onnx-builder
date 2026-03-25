"""
Unit tests for yobx.xtracing tracing primitives (NumpyArray, trace_numpy_function).
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase
from yobx.reference import ExtendedReferenceEvaluator


class TestNumpyArray(ExtTestCase):
    """Tests for the NumpyArray proxy and trace_numpy_function."""

    # ------------------------------------------------------------------
    # trace_numpy_function (converter-API)
    # ------------------------------------------------------------------

    def test_trace_numpy_function_basic(self):
        """trace_numpy_function should work with an existing GraphBuilder."""
        from onnx import TensorProto
        from yobx.xbuilder import GraphBuilder
        from yobx.xtracing import trace_numpy_function

        def f(X):
            return np.sqrt(np.abs(X) + np.float32(1))

        g = GraphBuilder({"": 21, "ai.onnx.ml": 1})
        g.make_tensor_input("X", TensorProto.FLOAT, ("batch", 3))
        trace_numpy_function(g, {}, ["output_0"], f, ["X"])
        g.make_tensor_output("output_0", indexed=False, allow_untyped_output=True)
        onx = g.to_onnx(return_optimize_report=True)

        X = np.random.randn(4, 3).astype(np.float32)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"X": X})[0]
        expected = f(X)
        self.assertEqualArray(expected, got, atol=1e-5)

    def test_trace_numpy_function_multiple_outputs(self):
        """trace_numpy_function raises when output count mismatches."""
        from onnx import TensorProto
        from yobx.xbuilder import GraphBuilder
        from yobx.xtracing import trace_numpy_function

        def f(X):
            return X + np.float32(1)

        g = GraphBuilder({"": 21, "ai.onnx.ml": 1})
        g.make_tensor_input("X", TensorProto.FLOAT, ("batch", 3))
        # Providing 2 output names for a single-output function should raise.
        with self.assertRaises(ValueError):
            trace_numpy_function(g, {}, ["out0", "out1"], f, ["X"])

    # ------------------------------------------------------------------
    # NumpyArray proxy
    # ------------------------------------------------------------------

    def test_expand_dims_method(self):
        """NumpyArray.expand_dims adds a size-1 dimension."""
        from onnx import TensorProto
        from yobx.xbuilder import GraphBuilder
        from yobx.xtracing.numpy_array import NumpyArray

        g = GraphBuilder({"": 21, "ai.onnx.ml": 1})
        g.make_tensor_input("X", TensorProto.FLOAT, ("batch", 3))
        proxy = NumpyArray("X", g, dtype=np.float32)
        out = proxy.expand_dims(axis=0)
        self.assertIsInstance(out, NumpyArray)

    # ------------------------------------------------------------------
    # Array functions
    # ------------------------------------------------------------------

    def _run(self, func, *inputs, input_names=None, rtol=1e-5, atol=1e-5):
        """Helper: trace *func*, run it, and compare with numpy reference."""
        from yobx.sql import trace_numpy_to_onnx

        onx = trace_numpy_to_onnx(func, *inputs, input_names=input_names)
        feeds = {
            (input_names[i] if input_names else ("X" if len(inputs) == 1 else f"X{i}")): inp
            for i, inp in enumerate(inputs)
        }
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, feeds)[0]
        expected = func(*inputs)
        self.assertEqualArray(expected, got, rtol=rtol, atol=atol)
        return onx

    def test_concatenate(self):
        def f(A, B):
            return np.concatenate([A, B], axis=0)

        A = np.random.randn(4, 3).astype(np.float32)
        B = np.random.randn(2, 3).astype(np.float32)
        self._run(f, A, B, input_names=["A", "B"])

    def test_stack(self):
        def f(A, B):
            return np.stack([A, B], axis=0)

        A = np.random.randn(4, 3).astype(np.float32)
        B = np.random.randn(4, 3).astype(np.float32)
        self._run(f, A, B, input_names=["A", "B"])

    def test_expand_dims_numpy(self):
        def f(X):
            return np.expand_dims(X, axis=0)

        X = np.random.randn(4, 3).astype(np.float32)
        self._run(f, X)

    def test_squeeze_numpy(self):
        def f(X):
            return np.squeeze(X, axis=0)

        X = np.random.randn(1, 4, 3).astype(np.float32)
        self._run(f, X)

    def test_dot(self):
        W = np.random.randn(3, 5).astype(np.float32)

        def f(X):
            return np.dot(X, W)

        X = np.random.randn(4, 3).astype(np.float32)
        self._run(f, X)

    # ------------------------------------------------------------------
    # Multiple outputs
    # ------------------------------------------------------------------

    def test_trace_numpy_to_onnx_two_outputs_auto(self):
        """trace_numpy_to_onnx auto-detects 2 outputs when output_names is None."""
        from yobx.sql import trace_numpy_to_onnx

        def f(X):
            return X + np.float32(1), X * np.float32(2)

        X = np.random.randn(4, 3).astype(np.float32)
        onx = trace_numpy_to_onnx(f, X)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"X": X})
        self.assertEqual(len(got), 2)
        self.assertEqualArray(X + np.float32(1), got[0], atol=1e-5)
        self.assertEqualArray(X * np.float32(2), got[1], atol=1e-5)

    def test_trace_numpy_to_onnx_two_outputs_explicit(self):
        """trace_numpy_to_onnx respects explicit output_names for 2 outputs."""
        from yobx.sql import trace_numpy_to_onnx

        def f(X):
            return X + np.float32(1), X * np.float32(2)

        X = np.random.randn(4, 3).astype(np.float32)
        onx = trace_numpy_to_onnx(f, X, output_names=["out_a", "out_b"])
        outputs = onx.proto.graph.output
        names = [o.name for o in outputs]
        self.assertIn("out_a", names)
        self.assertIn("out_b", names)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"X": X})
        self.assertEqual(len(got), 2)
        self.assertEqualArray(X + np.float32(1), got[0], atol=1e-5)
        self.assertEqualArray(X * np.float32(2), got[1], atol=1e-5)

    def test_trace_numpy_function_two_outputs(self):
        """trace_numpy_function should correctly handle functions returning tuples."""
        from onnx import TensorProto
        from yobx.xbuilder import GraphBuilder
        from yobx.xtracing import trace_numpy_function

        def f(X):
            return X + np.float32(1), X * np.float32(2)

        g = GraphBuilder({"": 21, "ai.onnx.ml": 1})
        g.make_tensor_input("X", TensorProto.FLOAT, ("batch", 3))
        trace_numpy_function(g, {}, ["out_a", "out_b"], f, ["X"])
        g.make_tensor_output("out_a", indexed=False, allow_untyped_output=True)
        g.make_tensor_output("out_b", indexed=False, allow_untyped_output=True)
        onx = g.to_onnx(return_optimize_report=True)

        X = np.random.randn(4, 3).astype(np.float32)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"X": X})
        self.assertEqual(len(got), 2)
        self.assertEqualArray(X + np.float32(1), got[0], atol=1e-5)
        self.assertEqualArray(X * np.float32(2), got[1], atol=1e-5)

    def test_trace_numpy_to_onnx_three_outputs_auto(self):
        """trace_numpy_to_onnx auto-detects 3 outputs returned as a tuple."""
        from yobx.sql import trace_numpy_to_onnx

        def f(X):
            return X + np.float32(1), X * np.float32(2), np.abs(X)

        X = np.random.randn(4, 3).astype(np.float32)
        onx = trace_numpy_to_onnx(f, X)
        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"X": X})
        self.assertEqual(len(got), 3)
        self.assertEqualArray(X + np.float32(1), got[0], atol=1e-5)
        self.assertEqualArray(X * np.float32(2), got[1], atol=1e-5)
        self.assertEqualArray(np.abs(X), got[2], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
