"""
Unit tests for yobx.sklearn.kernel_approximation.nystroem converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestNystroem(ExtTestCase):
    def _make_data(self, seed=0, n_samples=30, n_features=5):
        return (
            np.random.default_rng(seed)
            .standard_normal((n_samples, n_features))
            .astype(np.float32)
        )

    def _run_kernel(self, kernel, n_components=8, **nystroem_kwargs):
        from sklearn.kernel_approximation import Nystroem
        from yobx.sklearn import to_onnx

        X = self._make_data()
        est = Nystroem(
            n_components=n_components, kernel=kernel, random_state=0, **nystroem_kwargs
        )
        est.fit(X)

        onx = to_onnx(est, (X,))

        expected = est.transform(X).astype(np.float32)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(expected, result, atol=2e-3)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=2e-3)

    def test_nystroem_linear(self):
        self._run_kernel("linear")

    def test_nystroem_rbf(self):
        self._run_kernel("rbf", gamma=0.1)

    def test_nystroem_rbf_default_gamma(self):
        # When gamma is None sklearn uses 1/n_features.
        self._run_kernel("rbf")

    def test_nystroem_poly(self):
        self._run_kernel("poly", gamma=0.2, degree=3, coef0=1.0)

    def test_nystroem_poly_defaults(self):
        # Test that the converter correctly resolves None gamma/coef0/degree.
        self._run_kernel("poly")

    def test_nystroem_sigmoid(self):
        self._run_kernel("sigmoid", gamma=0.05, coef0=1.0)

    def test_nystroem_cosine(self):
        self._run_kernel("cosine")

    def test_nystroem_float64(self):
        from sklearn.kernel_approximation import Nystroem
        from yobx.sklearn import to_onnx

        X = self._make_data().astype(np.float64)
        est = Nystroem(n_components=8, kernel="rbf", gamma=0.1, random_state=0)
        est.fit(X)

        onx = to_onnx(est, (X,))

        expected = est.transform(X)
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_nystroem_out_of_sample(self):
        """Verify the converter works on data *not* seen during fitting."""
        from sklearn.kernel_approximation import Nystroem
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((20, 4)).astype(np.float32)
        X_test = rng.standard_normal((8, 4)).astype(np.float32)

        est = Nystroem(n_components=10, kernel="rbf", gamma=0.5, random_state=0)
        est.fit(X_train)

        onx = to_onnx(est, (X_train,))

        expected = est.transform(X_test).astype(np.float32)
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, result, atol=1e-4)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-4)

    def test_nystroem_unsupported_callable(self):
        from sklearn.kernel_approximation import Nystroem
        from yobx.sklearn import to_onnx

        X = self._make_data()
        est = Nystroem(n_components=5, kernel=lambda x, y: x @ y.T, random_state=0)
        est.fit(X)

        with self.assertRaises(NotImplementedError):
            to_onnx(est, (X,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
