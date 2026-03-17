"""
Unit tests for yobx.sklearn.covariance converters for EmpiricalCovariance
and all its subclasses that implement the mahalanobis method.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestEmpiricalCovarianceConverters(ExtTestCase):
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

    # ── EmpiricalCovariance ────────────────────────────────────────────────

    def test_empirical_covariance_default(self):
        from sklearn.covariance import EmpiricalCovariance

        X = self._make_data(seed=0)
        self._check(EmpiricalCovariance(), X)

    def test_empirical_covariance_assume_centered(self):
        from sklearn.covariance import EmpiricalCovariance

        X = self._make_data(seed=1)
        self._check(EmpiricalCovariance(assume_centered=True), X)

    def test_empirical_covariance_float64(self):
        from sklearn.covariance import EmpiricalCovariance
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 3)).astype(np.float64)
        est = EmpiricalCovariance()
        est.fit(X)
        onx = to_onnx(est, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        mahal_onnx = ref.run(None, {"X": X})[0]
        expected = est.mahalanobis(X)
        self.assertEqualArray(expected, mahal_onnx, atol=1e-5)

    def test_empirical_covariance_op_types(self):
        from sklearn.covariance import EmpiricalCovariance
        from yobx.sklearn import to_onnx

        X = self._make_data(seed=3)
        est = EmpiricalCovariance()
        est.fit(X)
        onx = to_onnx(est, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("MatMul", op_types)
        self.assertIn("Mul", op_types)
        self.assertIn("ReduceSum", op_types)

    # ── ShrunkCovariance ───────────────────────────────────────────────────

    def test_shrunk_covariance_default(self):
        from sklearn.covariance import ShrunkCovariance

        X = self._make_data(seed=4)
        self._check(ShrunkCovariance(), X)

    def test_shrunk_covariance_custom_shrinkage(self):
        from sklearn.covariance import ShrunkCovariance

        X = self._make_data(seed=5)
        self._check(ShrunkCovariance(shrinkage=0.2), X)

    def test_shrunk_covariance_float64(self):
        from sklearn.covariance import ShrunkCovariance
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(6)
        X = rng.standard_normal((40, 3)).astype(np.float64)
        est = ShrunkCovariance()
        est.fit(X)
        onx = to_onnx(est, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        mahal_onnx = ref.run(None, {"X": X})[0]
        expected = est.mahalanobis(X)
        self.assertEqualArray(expected, mahal_onnx, atol=1e-5)

    # ── OAS ───────────────────────────────────────────────────────────────

    def test_oas_default(self):
        from sklearn.covariance import OAS

        X = self._make_data(seed=7)
        self._check(OAS(), X)

    def test_oas_assume_centered(self):
        from sklearn.covariance import OAS

        X = self._make_data(seed=8)
        self._check(OAS(assume_centered=True), X)

    def test_oas_float64(self):
        from sklearn.covariance import OAS
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(9)
        X = rng.standard_normal((40, 3)).astype(np.float64)
        est = OAS()
        est.fit(X)
        onx = to_onnx(est, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        mahal_onnx = ref.run(None, {"X": X})[0]
        expected = est.mahalanobis(X)
        self.assertEqualArray(expected, mahal_onnx, atol=1e-5)

    # ── MinCovDet ─────────────────────────────────────────────────────────

    def test_min_cov_det_default(self):
        from sklearn.covariance import MinCovDet

        X = self._make_data(seed=10, n=100)
        self._check(MinCovDet(random_state=0), X)

    def test_min_cov_det_float64(self):
        from sklearn.covariance import MinCovDet
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(11)
        X = rng.standard_normal((100, 3)).astype(np.float64)
        est = MinCovDet(random_state=0)
        est.fit(X)
        onx = to_onnx(est, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        mahal_onnx = ref.run(None, {"X": X})[0]
        expected = est.mahalanobis(X)
        self.assertEqualArray(expected, mahal_onnx, atol=1e-5)

    # ── GraphicalLasso ────────────────────────────────────────────────────

    def test_graphical_lasso_default(self):
        from sklearn.covariance import GraphicalLasso

        X = self._make_data(seed=12)
        self._check(GraphicalLasso(), X)

    def test_graphical_lasso_float64(self):
        from sklearn.covariance import GraphicalLasso
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(13)
        X = rng.standard_normal((60, 4)).astype(np.float64)
        est = GraphicalLasso()
        est.fit(X)
        onx = to_onnx(est, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        mahal_onnx = ref.run(None, {"X": X})[0]
        expected = est.mahalanobis(X)
        self.assertEqualArray(expected, mahal_onnx, atol=1e-5)

    # ── GraphicalLassoCV ──────────────────────────────────────────────────

    def test_graphical_lasso_cv_default(self):
        from sklearn.covariance import GraphicalLassoCV

        X = self._make_data(seed=14, n=80)
        self._check(GraphicalLassoCV(), X, atol=1e-4)

    def test_graphical_lasso_cv_float64(self):
        from sklearn.covariance import GraphicalLassoCV
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(15)
        X = rng.standard_normal((80, 4)).astype(np.float64)
        est = GraphicalLassoCV()
        est.fit(X)
        onx = to_onnx(est, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        mahal_onnx = ref.run(None, {"X": X})[0]
        expected = est.mahalanobis(X)
        self.assertEqualArray(expected, mahal_onnx, atol=1e-5)

    # ── Pipeline tests ────────────────────────────────────────────────────

    def test_empirical_covariance_pipeline(self):
        from sklearn.covariance import EmpiricalCovariance
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(16)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        pipe = Pipeline([("scaler", StandardScaler()), ("cov", EmpiricalCovariance())])
        pipe.fit(X)
        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        mahal_onnx = ref.run(None, {"X": X})[0]

        cov = pipe.named_steps["cov"]
        scaler = pipe.named_steps["scaler"]
        X_scaled = scaler.transform(X).astype(np.float32)
        expected = cov.mahalanobis(X_scaled).astype(np.float32)
        self.assertEqualArray(expected, mahal_onnx, atol=1e-5)

    def test_oas_pipeline(self):
        from sklearn.covariance import OAS
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(17)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        pipe = Pipeline([("scaler", StandardScaler()), ("cov", OAS())])
        pipe.fit(X)
        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        mahal_onnx = ref.run(None, {"X": X})[0]

        cov = pipe.named_steps["cov"]
        scaler = pipe.named_steps["scaler"]
        X_scaled = scaler.transform(X).astype(np.float32)
        expected = cov.mahalanobis(X_scaled).astype(np.float32)
        self.assertEqualArray(expected, mahal_onnx, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
