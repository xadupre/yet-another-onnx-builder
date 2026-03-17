"""
Unit tests for yobx.sklearn.covariance.GraphicalLasso and GraphicalLassoCV converters.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestGraphicalLasso(ExtTestCase):
    def _make_data(self, seed=0, n=60, n_features=4):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n, n_features)).astype(np.float32)

    def _check(self, estimator, X, atol=1e-5):
        from yobx.sklearn import to_onnx

        estimator.fit(X)
        onx = to_onnx(estimator, (X,))

        output_names = [o.name for o in onx.graph.output]
        self.assertEqual(len(output_names), 1, f"Expected 1 output, got {output_names}")

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        mahal_onnx = results[0]

        expected = estimator.mahalanobis(X).astype(np.float32)
        self.assertEqualArray(expected, mahal_onnx, atol=atol)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=atol)

    def test_graphical_lasso_default(self):
        from sklearn.covariance import GraphicalLasso

        X = self._make_data(seed=0)
        self._check(GraphicalLasso(), X)

    def test_graphical_lasso_assume_centered(self):
        from sklearn.covariance import GraphicalLasso
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((60, 4)).astype(np.float64)
        gl = GraphicalLasso(assume_centered=True)
        gl.fit(X)
        onx = to_onnx(gl, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        mahal_onnx = results[0]

        expected = gl.mahalanobis(X)
        self.assertEqualArray(expected, mahal_onnx, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_graphical_lasso_float64(self):
        from sklearn.covariance import GraphicalLasso
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((60, 4)).astype(np.float64)
        gl = GraphicalLasso()
        gl.fit(X)
        onx = to_onnx(gl, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        mahal_onnx = results[0]

        expected = gl.mahalanobis(X)
        self.assertEqualArray(expected, mahal_onnx, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_graphical_lasso_op_types(self):
        from sklearn.covariance import GraphicalLasso
        from yobx.sklearn import to_onnx

        X = self._make_data(seed=3)
        gl = GraphicalLasso()
        gl.fit(X)
        onx = to_onnx(gl, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("MatMul", op_types)
        self.assertIn("Mul", op_types)
        self.assertIn("ReduceSum", op_types)

    def test_graphical_lasso_pipeline(self):
        from sklearn.covariance import GraphicalLasso
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(4)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        pipe = Pipeline([("scaler", StandardScaler()), ("gl", GraphicalLasso())])
        pipe.fit(X)
        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        mahal_onnx = results[0]

        gl = pipe.named_steps["gl"]
        scaler = pipe.named_steps["scaler"]
        X_scaled = scaler.transform(X).astype(np.float32)
        expected = gl.mahalanobis(X_scaled).astype(np.float32)
        self.assertEqualArray(expected, mahal_onnx, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)


@requires_sklearn("1.4")
class TestGraphicalLassoCV(ExtTestCase):
    def _make_data(self, seed=0, n=60, n_features=4):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n, n_features)).astype(np.float32)

    def _check(self, estimator, X, atol=1e-5):
        from yobx.sklearn import to_onnx

        estimator.fit(X)
        onx = to_onnx(estimator, (X,))

        output_names = [o.name for o in onx.graph.output]
        self.assertEqual(len(output_names), 1, f"Expected 1 output, got {output_names}")

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        mahal_onnx = results[0]

        expected = estimator.mahalanobis(X).astype(np.float32)
        self.assertEqualArray(expected, mahal_onnx, atol=atol)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=atol)

    def test_graphical_lasso_cv_default(self):
        from sklearn.covariance import GraphicalLassoCV

        X = self._make_data(seed=0)
        self._check(GraphicalLassoCV(), X)

    def test_graphical_lasso_cv_float64(self):
        from sklearn.covariance import GraphicalLassoCV
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X = rng.standard_normal((60, 4)).astype(np.float64)
        glcv = GraphicalLassoCV()
        glcv.fit(X)
        onx = to_onnx(glcv, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        mahal_onnx = results[0]

        expected = glcv.mahalanobis(X)
        self.assertEqualArray(expected, mahal_onnx, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_graphical_lasso_cv_op_types(self):
        from sklearn.covariance import GraphicalLassoCV
        from yobx.sklearn import to_onnx

        X = self._make_data(seed=2)
        glcv = GraphicalLassoCV()
        glcv.fit(X)
        onx = to_onnx(glcv, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("MatMul", op_types)
        self.assertIn("Mul", op_types)
        self.assertIn("ReduceSum", op_types)


if __name__ == "__main__":
    unittest.main(verbosity=2)
