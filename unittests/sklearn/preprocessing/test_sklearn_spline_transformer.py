"""
Unit tests for yobx.sklearn.preprocessing.SplineTransformer converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestSplineTransformer(ExtTestCase):
    def test_spline_transformer_default(self):
        from sklearn.preprocessing import SplineTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 2)).astype(np.float32)
        st = SplineTransformer(n_knots=4, degree=3)
        st.fit(X)

        onx = to_onnx(st, (X,))

        # Check graph structure includes polynomial evaluation ops.
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Pow", op_types)
        self.assertIn("MatMul", op_types)
        self.assertIn("Gather", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = st.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_spline_transformer_include_bias_false(self):
        from sklearn.preprocessing import SplineTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((30, 3)).astype(np.float32)
        st = SplineTransformer(n_knots=4, degree=3, include_bias=False)
        st.fit(X)

        onx = to_onnx(st, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = st.transform(X).astype(np.float32)
        self.assertEqual(expected.shape, result.shape)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_spline_transformer_degree1(self):
        from sklearn.preprocessing import SplineTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((20, 2)).astype(np.float32)
        st = SplineTransformer(n_knots=5, degree=1)
        st.fit(X)

        onx = to_onnx(st, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = st.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_spline_transformer_degree2(self):
        from sklearn.preprocessing import SplineTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X = rng.standard_normal((25, 2)).astype(np.float32)
        st = SplineTransformer(n_knots=4, degree=2)
        st.fit(X)

        onx = to_onnx(st, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = st.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_spline_transformer_constant_extrapolation_out_of_range(self):
        from sklearn.preprocessing import SplineTransformer
        from yobx.sklearn import to_onnx

        # Fit on [0, 1], test with values outside this range.
        X_fit = np.linspace(0, 1, 30).reshape(-1, 1).astype(np.float32)
        X_test = np.linspace(-0.5, 1.5, 20).reshape(-1, 1).astype(np.float32)
        st = SplineTransformer(n_knots=4, degree=3, extrapolation="constant")
        st.fit(X_fit)

        onx = to_onnx(st, (X_fit,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_test})[0]
        expected = st.transform(X_test).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_spline_transformer_continue_extrapolation(self):
        from sklearn.preprocessing import SplineTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(4)
        X = rng.standard_normal((30, 2)).astype(np.float32)
        st = SplineTransformer(n_knots=4, degree=3, extrapolation="continue")
        st.fit(X)

        onx = to_onnx(st, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = st.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_spline_transformer_single_feature(self):
        from sklearn.preprocessing import SplineTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(5)
        X = rng.standard_normal((40, 1)).astype(np.float32)
        st = SplineTransformer(n_knots=5, degree=3)
        st.fit(X)

        onx = to_onnx(st, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = st.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_spline_transformer_float64(self):
        from sklearn.preprocessing import SplineTransformer
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(6)
        X = rng.standard_normal((25, 2))  # float64
        st = SplineTransformer(n_knots=4, degree=3)
        st.fit(X)

        onx = to_onnx(st, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = st.transform(X)
        self.assertEqualArray(expected, result, atol=1e-10)

    @requires_sklearn("1.8")  # discrepancies issues
    def test_spline_transformer_pipeline_with_linear_regression(self):
        from sklearn.preprocessing import SplineTransformer
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(7)
        X = rng.standard_normal((50, 3)).astype(np.float32)
        y = rng.standard_normal(50).astype(np.float32)

        pipe = Pipeline(
            [("spline", SplineTransformer(n_knots=5, degree=3)), ("reg", LinearRegression())]
        )
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = pipe.predict(X).astype(np.float32)
        self.assertEqualArray(expected, result.ravel(), atol=1e-4)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result.ravel(), atol=1e-4)

    def test_spline_transformer_unsupported_extrapolation_raises(self):
        from sklearn.preprocessing import SplineTransformer
        from yobx.sklearn import to_onnx

        X = np.linspace(0, 1, 20).reshape(-1, 1).astype(np.float32)
        st = SplineTransformer(n_knots=4, degree=3, extrapolation="linear")
        st.fit(X)

        with self.assertRaises(NotImplementedError):
            to_onnx(st, (X,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
