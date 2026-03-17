"""
Unit tests for yobx.sklearn.kernel_approximation.rbf_sampler converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestRBFSampler(ExtTestCase):
    def _make_data(self, seed=0, n_samples=30, n_features=5):
        return (
            np.random.default_rng(seed)
            .standard_normal((n_samples, n_features))
            .astype(np.float32)
        )

    def _run(self, X, **rbf_kwargs):
        from sklearn.kernel_approximation import RBFSampler
        from yobx.sklearn import to_onnx

        est = RBFSampler(random_state=0, **rbf_kwargs)
        est.fit(X)

        onx = to_onnx(est, (X,))

        expected = est.transform(X).astype(np.float32)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_rbf_sampler_default(self):
        self._run(self._make_data())

    def test_rbf_sampler_custom_gamma(self):
        self._run(self._make_data(), gamma=0.5)

    def test_rbf_sampler_n_components(self):
        self._run(self._make_data(), n_components=16)

    def test_rbf_sampler_float64(self):
        # ORT does not implement Cos for float64, so we only verify
        # correctness via the reference evaluator, not via ORT.
        from sklearn.kernel_approximation import RBFSampler
        from yobx.sklearn import to_onnx

        X = self._make_data().astype(np.float64)
        est = RBFSampler(gamma=0.1, n_components=10, random_state=0)
        est.fit(X)

        onx = to_onnx(est, (X,))

        expected = est.transform(X)
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(expected, result, atol=1e-10)

    def test_rbf_sampler_out_of_sample(self):
        """Verify the converter works on data not seen during fitting."""
        from sklearn.kernel_approximation import RBFSampler
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((20, 4)).astype(np.float32)
        X_test = rng.standard_normal((8, 4)).astype(np.float32)

        est = RBFSampler(gamma=0.5, n_components=12, random_state=0)
        est.fit(X_train)

        onx = to_onnx(est, (X_train,))

        expected = est.transform(X_test).astype(np.float32)
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
