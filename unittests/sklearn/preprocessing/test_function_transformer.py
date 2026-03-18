"""
Unit tests for the FunctionTransformer → ONNX converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestFunctionTransformer(ExtTestCase):
    """Tests for yobx.sklearn.preprocessing.function_transformer."""

    # ------------------------------------------------------------------
    # Identity (func=None)
    # ------------------------------------------------------------------

    def test_identity(self):
        """func=None should emit a single Identity node."""
        from sklearn.preprocessing import FunctionTransformer
        from yobx.sklearn import to_onnx

        X = np.random.randn(6, 4).astype(np.float32)
        transformer = FunctionTransformer(func=None)
        transformer.fit(X)

        onx = to_onnx(transformer, (X,))
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Identity", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"X": X})[0]
        self.assertEqualArray(X, got)

        sess = self.check_ort(onx)
        self.assertEqualArray(X, sess.run(None, {"X": X})[0])

    # ------------------------------------------------------------------
    # Simple element-wise functions
    # ------------------------------------------------------------------

    def test_sqrt_abs(self):
        """sqrt(|X| + 1) traced to Sqrt, Abs, Add, Identity."""
        from sklearn.preprocessing import FunctionTransformer
        from yobx.sklearn import to_onnx

        def my_func(X):
            return np.sqrt(np.abs(X) + np.float32(1))

        X = np.random.randn(6, 4).astype(np.float32)
        transformer = FunctionTransformer(func=my_func)
        transformer.fit(X)

        onx = to_onnx(transformer, (X,))
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sqrt", op_types)
        self.assertIn("Abs", op_types)
        self.assertIn("Add", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"X": X})[0]
        expected = my_func(X)
        self.assertEqualArray(expected, got, atol=1e-5)

        sess = self.check_ort(onx)
        self.assertEqualArray(expected, sess.run(None, {"X": X})[0], atol=1e-5)

    def test_log1p(self):
        """log1p(X) should be emitted as Add + Log."""
        from sklearn.preprocessing import FunctionTransformer
        from yobx.sklearn import to_onnx

        def my_func(X):
            return np.log1p(X)

        X = np.abs(np.random.randn(6, 4).astype(np.float32))
        transformer = FunctionTransformer(func=my_func)
        transformer.fit(X)

        onx = to_onnx(transformer, (X,))
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Log", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"X": X})[0]
        expected = my_func(X)
        self.assertEqualArray(expected, got, atol=1e-4)

        sess = self.check_ort(onx)
        self.assertEqualArray(expected, sess.run(None, {"X": X})[0], atol=1e-4)

    def test_scale_shift(self):
        """2 * X + 1."""
        from sklearn.preprocessing import FunctionTransformer
        from yobx.sklearn import to_onnx

        def my_func(X):
            return np.float32(2) * X + np.float32(1)

        X = np.random.randn(6, 4).astype(np.float32)
        transformer = FunctionTransformer(func=my_func)
        transformer.fit(X)

        onx = to_onnx(transformer, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"X": X})[0]
        expected = my_func(X)
        self.assertEqualArray(expected, got, atol=1e-5)

        sess = self.check_ort(onx)
        self.assertEqualArray(expected, sess.run(None, {"X": X})[0], atol=1e-5)

    # ------------------------------------------------------------------
    # kw_args
    # ------------------------------------------------------------------

    def test_kw_args(self):
        """kw_args are forwarded to func."""
        from sklearn.preprocessing import FunctionTransformer
        from yobx.sklearn import to_onnx

        def my_func(X, scale=np.float32(1)):
            return X * scale

        X = np.random.randn(6, 4).astype(np.float32)
        transformer = FunctionTransformer(func=my_func, kw_args={"scale": np.float32(3)})
        transformer.fit(X)

        onx = to_onnx(transformer, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"X": X})[0]
        expected = my_func(X, scale=np.float32(3))
        self.assertEqualArray(expected, got, atol=1e-5)

    # ------------------------------------------------------------------
    # Pipeline integration
    # ------------------------------------------------------------------

    def test_pipeline(self):
        """FunctionTransformer inside a Pipeline."""
        from sklearn.preprocessing import FunctionTransformer, StandardScaler
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        def my_func(X):
            return np.sqrt(np.abs(X) + np.float32(1))

        X = np.random.randn(20, 5).astype(np.float32)
        pipe = Pipeline(
            [
                ("fn", FunctionTransformer(func=my_func)),
                ("scaler", StandardScaler()),
            ]
        )
        pipe.fit(X)

        onx = to_onnx(pipe, (X,))
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sqrt", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"X": X})[0]
        expected = pipe.transform(X).astype(np.float32)
        self.assertEqualArray(expected, got, atol=1e-5)

        sess = self.check_ort(onx)
        self.assertEqualArray(expected, sess.run(None, {"X": X})[0], atol=1e-5)

    # ------------------------------------------------------------------
    # float64
    # ------------------------------------------------------------------

    def test_float64(self):
        """Float64 inputs should be handled correctly."""
        from sklearn.preprocessing import FunctionTransformer
        from yobx.sklearn import to_onnx

        def my_func(X):
            return np.tanh(X)

        X = np.random.randn(6, 4)  # float64
        transformer = FunctionTransformer(func=my_func)
        transformer.fit(X)

        onx = to_onnx(transformer, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        got = ref.run(None, {"X": X})[0]
        expected = my_func(X)
        self.assertEqualArray(expected, got, atol=1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
