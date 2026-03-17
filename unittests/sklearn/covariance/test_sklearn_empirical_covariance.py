"""
Unit tests for yobx.sklearn.covariance.EmpiricalCovariance converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestEmpiricalCovariance(ExtTestCase):
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

    def test_empirical_covariance_default(self):
        from sklearn.covariance import EmpiricalCovariance

        X = self._make_data(seed=0)
        self._check(EmpiricalCovariance(), X)

    def test_empirical_covariance_assume_centered(self):
        from sklearn.covariance import EmpiricalCovariance

        X = self._make_data(seed=1)
        self._check(EmpiricalCovariance(assume_centered=True), X)

    def test_empirical_covariance_float32(self):
        from sklearn.covariance import EmpiricalCovariance
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(2)
        X = rng.standard_normal((40, 3)).astype(np.float32)
        ec = EmpiricalCovariance()
        ec.fit(X)
        onx = to_onnx(ec, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        mahal_onnx = results[0]

        expected = ec.mahalanobis(X).astype(np.float32)
        self.assertEqualArray(expected, mahal_onnx, atol=1e-4)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_empirical_covariance_float64(self):
        from sklearn.covariance import EmpiricalCovariance
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(3)
        X = rng.standard_normal((40, 3)).astype(np.float64)
        ec = EmpiricalCovariance()
        ec.fit(X)
        onx = to_onnx(ec, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        mahal_onnx = results[0]

        expected = ec.mahalanobis(X)
        self.assertEqualArray(expected, mahal_onnx, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_empirical_covariance_op_types(self):
        from sklearn.covariance import EmpiricalCovariance
        from yobx.sklearn import to_onnx

        X = self._make_data(seed=3)
        ec = EmpiricalCovariance()
        ec.fit(X)
        onx = to_onnx(ec, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("MatMul", op_types)
        self.assertIn("Mul", op_types)
        self.assertIn("ReduceSum", op_types)

    def test_empirical_covariance_pipeline(self):
        from sklearn.covariance import EmpiricalCovariance
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(4)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        pipe = Pipeline([("scaler", StandardScaler()), ("ec", EmpiricalCovariance())])
        pipe.fit(X)
        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        mahal_onnx = results[0]

        ec = pipe.named_steps["ec"]
        scaler = pipe.named_steps["scaler"]
        X_scaled = scaler.transform(X).astype(np.float32)
        expected = ec.mahalanobis(X_scaled).astype(np.float32)
        self.assertEqualArray(expected, mahal_onnx, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
