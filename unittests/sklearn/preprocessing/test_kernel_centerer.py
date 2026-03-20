"""
Unit tests for yobx.sklearn.preprocessing.KernelCenterer converter.
"""

import unittest
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestKernelCenterer(ExtTestCase):
    # ── shared fixtures ────────────────────────────────────────────────────
    _X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    _X_test = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], dtype=np.float64)

    # ── helper ─────────────────────────────────────────────────────────────

    def _check(self, K_train, K_test, estimator, dtype, atol=1e-5):
        """Fit centerer, convert to ONNX, compare outputs."""
        from yobx.sklearn import to_onnx

        K_test_d = K_test.astype(dtype)
        onx = to_onnx(estimator, (K_test_d,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": K_test_d})[0]
        expected = estimator.transform(K_test.copy()).astype(dtype)
        self.assertEqualArray(expected, result, atol=atol)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": K_test_d})[0]
        self.assertEqualArray(expected, ort_result, atol=atol)

    # ── tests ──────────────────────────────────────────────────────────────

    def test_kernel_centerer_float32(self):
        """KernelCenterer on RBF kernel matrix, float32 input."""
        from sklearn.preprocessing import KernelCenterer

        K_train = rbf_kernel(self._X_train, gamma=0.1)
        K_test = rbf_kernel(self._X_test, self._X_train, gamma=0.1)

        kc = KernelCenterer()
        kc.fit(K_train)

        self._check(K_train, K_test, kc, np.float32)

    def test_kernel_centerer_float64(self):
        """KernelCenterer on RBF kernel matrix, float64 input."""
        from sklearn.preprocessing import KernelCenterer

        K_train = rbf_kernel(self._X_train, gamma=0.1)
        K_test = rbf_kernel(self._X_test, self._X_train, gamma=0.1)

        kc = KernelCenterer()
        kc.fit(K_train)

        self._check(K_train, K_test, kc, np.float64)

    def test_kernel_centerer_linear_kernel(self):
        """KernelCenterer on a linear (dot-product) kernel matrix."""
        from sklearn.preprocessing import KernelCenterer

        K_train = self._X_train @ self._X_train.T
        K_test = self._X_test @ self._X_train.T

        kc = KernelCenterer()
        kc.fit(K_train)

        for dtype in (np.float32, np.float64):
            self._check(K_train, K_test, kc, dtype, atol=1e-4)

    def test_kernel_centerer_graph_ops(self):
        """Graph must contain Sub, Div, and Add nodes."""
        from sklearn.preprocessing import KernelCenterer
        from yobx.sklearn import to_onnx

        K_train = rbf_kernel(self._X_train, gamma=0.1)
        K_test = rbf_kernel(self._X_test, self._X_train, gamma=0.1).astype(np.float32)

        kc = KernelCenterer()
        kc.fit(K_train)

        onx = to_onnx(kc, (K_test,))
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Div", op_types)
        self.assertIn("Add", op_types)
        self.assertIn("ReduceSum", op_types)

    def test_kernel_centerer_transform_training_set(self):
        """Centering the training kernel matrix yields zero-mean columns."""
        from sklearn.preprocessing import KernelCenterer
        from yobx.sklearn import to_onnx

        K_train = rbf_kernel(self._X_train, gamma=0.5)
        K_train_f32 = K_train.astype(np.float32)

        kc = KernelCenterer()
        kc.fit(K_train)

        onx = to_onnx(kc, (K_train_f32,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": K_train_f32})[0]
        expected = kc.transform(K_train.copy()).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": K_train_f32})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
