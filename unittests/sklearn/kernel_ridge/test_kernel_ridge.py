"""
Unit tests for sklearn KernelRidge converter.
"""

import unittest
import warnings

import numpy as np
from sklearn.kernel_ridge import KernelRidge

from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.0")
class TestSklearnKernelRidge(ExtTestCase):
    """Tests for KernelRidge ONNX converter."""

    # ── shared fixtures ────────────────────────────────────────────────────────

    _X = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float64
    )
    # positive values required by chi2 kernel
    _X_pos = np.abs(_X) + 0.1
    _y = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5])
    _y_multi = np.column_stack([_y, _y * 0.5 + 0.3])

    # ── helper ─────────────────────────────────────────────────────────────────

    def _check(self, estimator, X, y, atol=1e-4):
        """Fit, convert to ONNX, compare predictions for float32 and float64."""
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X, y)
            onx = to_onnx(estimator, (Xd,))

            output_names = [o.name for o in onx.graph.output]
            self.assertEqual(len(output_names), 1, f"Expected 1 output, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            expected = estimator.predict(X).astype(dtype)
            self.assertEqualArray(expected, results[0], atol=atol)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected, ort_results[0], atol=atol)

    # ── kernel tests ───────────────────────────────────────────────────────────

    def test_linear_kernel(self):
        """Linear kernel: K(x, y) = x · y."""
        self._check(KernelRidge(kernel="linear", alpha=1.0), self._X, self._y)

    def test_rbf_kernel_default_gamma(self):
        """RBF kernel with default gamma (1 / n_features)."""
        self._check(KernelRidge(kernel="rbf", alpha=0.5), self._X, self._y)

    def test_rbf_kernel_explicit_gamma(self):
        """RBF kernel with an explicit gamma."""
        self._check(KernelRidge(kernel="rbf", gamma=0.1, alpha=0.5), self._X, self._y)

    def test_poly_kernel(self):
        """Polynomial kernel."""
        self._check(
            KernelRidge(kernel="poly", gamma=0.5, degree=2, coef0=1.0, alpha=0.1),
            self._X,
            self._y,
        )

    def test_polynomial_kernel_alias(self):
        """'polynomial' is an alias for 'poly'."""
        self._check(
            KernelRidge(kernel="polynomial", gamma=0.5, degree=2, coef0=1.0, alpha=0.1),
            self._X,
            self._y,
        )

    def test_sigmoid_kernel(self):
        """Sigmoid kernel: K(x, y) = tanh(gamma * x·y + coef0)."""
        self._check(
            KernelRidge(kernel="sigmoid", gamma=0.01, coef0=0.0, alpha=0.1), self._X, self._y
        )

    def test_cosine_kernel(self):
        """Cosine similarity kernel."""
        self._check(KernelRidge(kernel="cosine", alpha=0.5), self._X, self._y)

    def test_laplacian_kernel(self):
        """Laplacian kernel: K(x, y) = exp(-gamma * ||x-y||_1)."""
        self._check(KernelRidge(kernel="laplacian", gamma=0.1, alpha=0.5), self._X, self._y)

    def test_chi2_kernel(self):
        """Chi² kernel (requires non-negative features)."""
        self._check(KernelRidge(kernel="chi2", gamma=0.5, alpha=0.5), self._X_pos, self._y)

    def test_multi_target(self):
        """Multi-target regression: dual_coef_ is (M, n_targets)."""
        self._check(KernelRidge(kernel="rbf", gamma=0.1, alpha=0.5), self._X, self._y_multi)

    def test_rbf_float32_fit(self):
        """Ensure float32 conversion is numerically stable."""
        X32 = self._X.astype(np.float32)
        est = KernelRidge(kernel="rbf", gamma=0.1, alpha=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X32, self._y.astype(np.float32))
        onx = to_onnx(est, (X32,))
        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X32})
        expected = est.predict(X32).astype(np.float32)
        self.assertEqualArray(expected, results[0], atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
