"""
Unit tests for sklearn GaussianProcessRegressor and GaussianProcessClassifier converters.
"""

import unittest
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, Matern, RBF, WhiteKernel

from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.8")
class TestSklearnGaussianProcess(ExtTestCase):
    """Tests for GaussianProcessRegressor and GaussianProcessClassifier ONNX converters."""

    # ── shared fixtures ────────────────────────────────────────────────────────

    _X = np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float64
    )
    _y_reg = np.array([1.0, 2.0, 3.0, 4.0, 1.5, 2.5])
    _y_bin = np.array([0, 0, 1, 1, 0, 1])
    _y_multi = np.array([0, 1, 2, 1, 0, 2])

    # ── helpers ────────────────────────────────────────────────────────────────

    def _check_regressor(self, estimator, X, y, atol=1e-4):
        """Fit, convert to ONNX, compare predictions against sklearn for float32/64."""
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X, y)  # always fit in float64
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

    def _check_classifier(self, estimator, X, y, atol=5e-4):
        """Fit, convert to ONNX, compare outputs against sklearn for float32/64."""
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X, y)  # always fit in float64
            onx = to_onnx(estimator, (Xd,))

            output_names = [o.name for o in onx.graph.output]
            self.assertEqual(len(output_names), 2, f"Expected 2 outputs, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            label, proba = results[0], results[1]

            expected_label = estimator.predict(X)
            expected_proba = estimator.predict_proba(X).astype(dtype)

            self.assertEqualArray(expected_label, label)
            self.assertEqualArray(expected_proba, proba, atol=atol)

            # OnnxRuntime does not support Erf for float64 inputs; skip ORT
            # validation when dtype=float64.
            if dtype == np.float32:
                sess = self.check_ort(onx)
                ort_results = sess.run(None, {"X": Xd})
                self.assertEqualArray(expected_label, ort_results[0])
                self.assertEqualArray(expected_proba, ort_results[1], atol=atol)

    # ── GaussianProcessRegressor ───────────────────────────────────────────────

    def test_gpr_default_kernel(self):
        """Default kernel: ConstantKernel * RBF."""
        gpr = GaussianProcessRegressor(random_state=0)
        self._check_regressor(gpr, self._X, self._y_reg)

    def test_gpr_rbf_kernel(self):
        """Pure RBF kernel with fixed hyperparameters."""
        gpr = GaussianProcessRegressor(kernel=RBF(1.0), optimizer=None)
        self._check_regressor(gpr, self._X, self._y_reg)

    def test_gpr_rbf_ard_kernel(self):
        """ARD-RBF kernel (per-feature length scale)."""
        gpr = GaussianProcessRegressor(kernel=RBF(length_scale=[1.0, 2.0]), optimizer=None)
        self._check_regressor(gpr, self._X, self._y_reg)

    def test_gpr_matern_nu05(self):
        """Matern kernel with nu=0.5 (absolute exponential)."""
        gpr = GaussianProcessRegressor(kernel=Matern(nu=0.5), optimizer=None)
        self._check_regressor(gpr, self._X, self._y_reg)

    def test_gpr_matern_nu15(self):
        """Matern kernel with nu=1.5 (once-differentiable)."""
        gpr = GaussianProcessRegressor(kernel=Matern(nu=1.5), optimizer=None)
        self._check_regressor(gpr, self._X, self._y_reg)

    def test_gpr_matern_nu25(self):
        """Matern kernel with nu=2.5 (twice-differentiable)."""
        gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), optimizer=None)
        self._check_regressor(gpr, self._X, self._y_reg)

    def test_gpr_dot_product_kernel(self):
        """DotProduct kernel."""
        gpr = GaussianProcessRegressor(kernel=DotProduct(sigma_0=1.0), optimizer=None)
        self._check_regressor(gpr, self._X, self._y_reg)

    def test_gpr_product_kernel(self):
        """ConstantKernel * RBF product kernel with fixed hyperparameters."""
        kernel = ConstantKernel(2.0) * RBF(1.5)
        gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None)
        self._check_regressor(gpr, self._X, self._y_reg)

    def test_gpr_sum_kernel(self):
        """RBF + WhiteKernel sum kernel."""
        kernel = RBF(1.0) + WhiteKernel(noise_level=0.1)
        gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None)
        self._check_regressor(gpr, self._X, self._y_reg)

    def test_gpr_constant_times_matern(self):
        """ConstantKernel * Matern product kernel."""
        kernel = ConstantKernel(2.0) * Matern(nu=1.5)
        gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None)
        self._check_regressor(gpr, self._X, self._y_reg)

    def test_gpr_normalize_y(self):
        """GPR with normalize_y=True (non-unit y_train_std)."""
        gpr = GaussianProcessRegressor(kernel=RBF(1.0), optimizer=None, normalize_y=True)
        self._check_regressor(gpr, self._X, self._y_reg)

    # ── GaussianProcessClassifier — binary ─────────────────────────────────────

    def test_gpc_binary_default_kernel(self):
        """Binary GPC with default kernel."""
        gpc = GaussianProcessClassifier(random_state=0)
        self._check_classifier(gpc, self._X, self._y_bin)

    def test_gpc_binary_rbf_kernel(self):
        """Binary GPC with pure RBF kernel."""
        gpc = GaussianProcessClassifier(kernel=RBF(1.0), random_state=0)
        self._check_classifier(gpc, self._X, self._y_bin)

    def test_gpc_binary_matern_kernel(self):
        """Binary GPC with Matern nu=1.5 kernel."""
        gpc = GaussianProcessClassifier(kernel=Matern(nu=1.5), random_state=0)
        self._check_classifier(gpc, self._X, self._y_bin)

    def test_gpc_binary_product_kernel(self):
        """Binary GPC with ConstantKernel * RBF kernel."""
        kernel = ConstantKernel(1.0) * RBF(1.0)
        gpc = GaussianProcessClassifier(kernel=kernel, random_state=0)
        self._check_classifier(gpc, self._X, self._y_bin)

    # ── GaussianProcessClassifier — multiclass (OvR) ──────────────────────────

    def test_gpc_multiclass_default_kernel(self):
        """Multiclass GPC (one_vs_rest) with default kernel."""
        gpc = GaussianProcessClassifier(random_state=0)
        self._check_classifier(gpc, self._X, self._y_multi)

    def test_gpc_multiclass_rbf_kernel(self):
        """Multiclass GPC (one_vs_rest) with RBF kernel."""
        gpc = GaussianProcessClassifier(kernel=RBF(1.0), random_state=0)
        self._check_classifier(gpc, self._X, self._y_multi)

    def test_gpc_multiclass_matern_kernel(self):
        """Multiclass GPC (one_vs_rest) with Matern nu=1.5 kernel."""
        gpc = GaussianProcessClassifier(kernel=Matern(nu=1.5), random_state=0)
        self._check_classifier(gpc, self._X, self._y_multi)

    # ── unsupported cases ──────────────────────────────────────────────────────

    def test_gpc_one_vs_one_raises(self):
        """GPC with multi_class='one_vs_one' and >2 classes should raise."""
        gpc = GaussianProcessClassifier(multi_class="one_vs_one", random_state=0)
        gpc.fit(self._X, self._y_multi)
        with self.assertRaises(NotImplementedError):
            to_onnx(gpc, (self._X.astype(np.float32),))

    def test_gpr_unsupported_kernel_raises(self):
        """GPR with an unsupported kernel type should raise NotImplementedError."""
        from sklearn.gaussian_process.kernels import ExpSineSquared

        gpr = GaussianProcessRegressor(kernel=ExpSineSquared(), optimizer=None)
        gpr.fit(self._X, self._y_reg)
        with self.assertRaises(NotImplementedError):
            to_onnx(gpr, (self._X.astype(np.float32),))

    # ── com.microsoft CDist path ───────────────────────────────────────────────

    def test_gpr_com_microsoft_cdist(self):
        """GPR uses com.microsoft.CDist when the domain is available."""
        X = self._X.astype(np.float32)
        gpr = GaussianProcessRegressor(kernel=RBF(1.0), optimizer=None)
        gpr.fit(self._X, self._y_reg)

        onx = to_onnx(gpr, (X,), target_opset={"": 20, "ai.onnx.ml": 3, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        expected = gpr.predict(self._X).astype(np.float32)
        self.assertEqualArray(expected, ort_results[0], atol=1e-4)

    def test_gpc_com_microsoft_cdist(self):
        """GPC uses com.microsoft.CDist when the domain is available."""
        X = self._X.astype(np.float32)
        gpc = GaussianProcessClassifier(random_state=0)
        gpc.fit(self._X, self._y_bin)

        onx = to_onnx(gpc, (X,), target_opset={"": 20, "ai.onnx.ml": 3, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        expected_proba = gpc.predict_proba(self._X).astype(np.float32)
        self.assertEqualArray(expected_proba, ort_results[1], atol=5e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
