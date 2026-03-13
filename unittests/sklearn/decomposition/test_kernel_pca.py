"""
Unit tests for yobx.sklearn.decomposition.KernelPCA converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestKernelPCA(ExtTestCase):
    def _make_data(self, seed=0, n_samples=30, n_features=5):
        return np.random.default_rng(seed).standard_normal(
            (n_samples, n_features)
        ).astype(np.float32)

    def _run_kernel(self, kernel, n_components=3, **kpca_kwargs):
        from sklearn.decomposition import KernelPCA
        from yobx.sklearn import to_onnx

        X = self._make_data()
        kpca = KernelPCA(n_components=n_components, kernel=kernel, **kpca_kwargs)
        kpca.fit(X)

        onx = to_onnx(kpca, (X,))

        expected = kpca.transform(X).astype(np.float32)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(expected, result, atol=1e-4)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-4)

    def test_kernel_pca_linear(self):
        self._run_kernel("linear")

    def test_kernel_pca_rbf(self):
        self._run_kernel("rbf", gamma=0.1)

    def test_kernel_pca_rbf_default_gamma(self):
        # When gamma is None sklearn uses 1/n_features.
        self._run_kernel("rbf")

    def test_kernel_pca_poly(self):
        self._run_kernel("poly", gamma=0.2, degree=3, coef0=1.0)

    def test_kernel_pca_sigmoid(self):
        self._run_kernel("sigmoid", gamma=0.05, coef0=1.0)

    def test_kernel_pca_cosine(self):
        self._run_kernel("cosine")

    def test_kernel_pca_float64(self):
        from sklearn.decomposition import KernelPCA
        from yobx.sklearn import to_onnx

        X = self._make_data().astype(np.float64)
        kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.1)
        kpca.fit(X)

        onx = to_onnx(kpca, (X,))

        expected = kpca.transform(X)
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_kernel_pca_unsupported_precomputed(self):
        from sklearn.decomposition import KernelPCA
        from yobx.sklearn import to_onnx

        # For precomputed we need to pass a kernel matrix during fit.
        X = self._make_data()
        K = X @ X.T  # linear kernel matrix as precomputed input
        kpca = KernelPCA(n_components=2, kernel="precomputed")
        kpca.fit(K)

        with self.assertRaises(NotImplementedError):
            to_onnx(kpca, (K,))

    def test_kernel_pca_unsupported_callable(self):
        from sklearn.decomposition import KernelPCA
        from yobx.sklearn import to_onnx

        X = self._make_data()
        kpca = KernelPCA(n_components=2, kernel=lambda x, y: x @ y.T)
        kpca.fit(X)

        with self.assertRaises(NotImplementedError):
            to_onnx(kpca, (X,))

    def test_kernel_pca_out_of_sample(self):
        """Verify that the converter works on data *not* seen during fitting."""
        from sklearn.decomposition import KernelPCA
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((20, 4)).astype(np.float32)
        X_test = rng.standard_normal((8, 4)).astype(np.float32)

        kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.5)
        kpca.fit(X_train)

        onx = to_onnx(kpca, (X_train,))

        expected = kpca.transform(X_test).astype(np.float32)
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, result, atol=1e-4)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
