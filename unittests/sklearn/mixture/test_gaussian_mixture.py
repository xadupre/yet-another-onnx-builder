"""
Unit tests for yobx.sklearn.mixture.GaussianMixture converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestGaussianMixture(ExtTestCase):
    def _check_gaussian_mixture(self, gm, X, atol=1e-4):
        """Helper: convert, run reference + ORT, compare to sklearn."""
        from yobx.sklearn import to_onnx

        onx = to_onnx(gm, (X,))

        expected_labels = gm.predict(X).astype(np.int64)
        expected_proba = gm.predict_proba(X).astype(np.float32)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        self.assertEqualArray(expected_labels, results[0])
        self.assertEqualArray(expected_proba, results[1], atol=atol)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=atol)

    def test_gaussian_mixture_full(self):
        from sklearn.mixture import GaussianMixture

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        gm = GaussianMixture(n_components=3, covariance_type="full", random_state=0)
        gm.fit(X)
        self._check_gaussian_mixture(gm, X)

    def test_gaussian_mixture_tied(self):
        from sklearn.mixture import GaussianMixture

        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        gm = GaussianMixture(n_components=3, covariance_type="tied", random_state=0)
        gm.fit(X)
        self._check_gaussian_mixture(gm, X)

    def test_gaussian_mixture_diag(self):
        from sklearn.mixture import GaussianMixture

        rng = np.random.default_rng(2)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        gm = GaussianMixture(n_components=3, covariance_type="diag", random_state=0)
        gm.fit(X)
        self._check_gaussian_mixture(gm, X)

    def test_gaussian_mixture_spherical(self):
        from sklearn.mixture import GaussianMixture

        rng = np.random.default_rng(3)
        X = rng.standard_normal((50, 4)).astype(np.float32)
        gm = GaussianMixture(n_components=3, covariance_type="spherical", random_state=0)
        gm.fit(X)
        self._check_gaussian_mixture(gm, X)

    def test_gaussian_mixture_two_components(self):
        from sklearn.mixture import GaussianMixture

        rng = np.random.default_rng(4)
        X = rng.standard_normal((20, 2)).astype(np.float32)
        gm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
        gm.fit(X)
        self._check_gaussian_mixture(gm, X)

    def test_gaussian_mixture_pipeline(self):
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(5)
        X = rng.standard_normal((40, 3)).astype(np.float32)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("gm", GaussianMixture(n_components=3, covariance_type="diag", random_state=0)),
            ]
        )
        pipe.fit(X)

        onx = to_onnx(pipe, (X,))

        expected_labels = pipe.predict(X).astype(np.int64)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        self.assertEqualArray(expected_labels, results[0])

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_labels, ort_results[0])

    def test_gaussian_mixture_op_types(self):
        from sklearn.mixture import GaussianMixture
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(6)
        X = rng.standard_normal((30, 3)).astype(np.float32)
        gm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
        gm.fit(X)

        onx = to_onnx(gm, (X,))
        op_types = {n.op_type for n in onx.proto.graph.node}
        self.assertIn("MatMul", op_types)
        self.assertIn("Softmax", op_types)
        self.assertIn("ArgMax", op_types)


if __name__ == "__main__":
    unittest.main(verbosity=2)
