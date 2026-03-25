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


if __name__ == "__main__":
    unittest.main(verbosity=2)
