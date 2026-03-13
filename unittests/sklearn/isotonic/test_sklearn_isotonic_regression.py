"""
Unit tests for yobx.sklearn.isotonic.isotonic_regression converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
class TestIsotonicRegressionConverter(ExtTestCase):

    def _check(self, estimator, X, atol=1e-5):
        """Fit *estimator*, convert to ONNX, and compare predictions."""
        from yobx.sklearn import to_onnx

        y = np.arange(len(X), dtype=np.float64)
        X_1d = X.ravel()

        for dtype in (np.float32, np.float64):
            Xd = X_1d.astype(dtype)
            estimator.fit(Xd, y)

            # Test with 1D input
            onx = to_onnx(estimator, (Xd,))
            ref = ExtendedReferenceEvaluator(onx)
            result = ref.run(None, {"X": Xd})[0]
            expected = estimator.predict(Xd).astype(dtype)
            self.assertEqualArray(expected, result, atol=atol)

            sess = self.check_ort(onx)
            ort_result = sess.run(None, {"X": Xd})[0]
            self.assertEqualArray(expected, ort_result, atol=atol)

            # Test with 2D input (N, 1)
            Xd_2d = Xd.reshape(-1, 1)
            onx_2d = to_onnx(estimator, (Xd_2d,))
            ref_2d = ExtendedReferenceEvaluator(onx_2d)
            result_2d = ref_2d.run(None, {"X": Xd_2d})[0]
            self.assertEqualArray(expected, result_2d, atol=atol)

            sess_2d = self.check_ort(onx_2d)
            ort_result_2d = sess_2d.run(None, {"X": Xd_2d})[0]
            self.assertEqualArray(expected, ort_result_2d, atol=atol)

    def test_increasing(self):
        from sklearn.isotonic import IsotonicRegression

        X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
        self._check(IsotonicRegression(out_of_bounds="clip"), X)

    def test_decreasing(self):
        from sklearn.isotonic import IsotonicRegression

        X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
        self._check(IsotonicRegression(increasing=False, out_of_bounds="clip"), X)

    def test_out_of_bounds_clip(self):
        """Values outside the training range should be clamped."""
        from sklearn.isotonic import IsotonicRegression
        from yobx.sklearn import to_onnx

        X_train = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
        y_train = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(X_train, y_train)

        X_test = np.array([0.0, 1.0, 5.0, 9.0, 10.0], dtype=np.float32)
        onx = to_onnx(ir, (X_test,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_test})[0]
        expected = ir.predict(X_test).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_flat_plateaus(self):
        """Isotonic regression may produce flat plateaus (repeated y values)."""
        from sklearn.isotonic import IsotonicRegression
        from yobx.sklearn import to_onnx

        X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
        y_train = np.array([1, 2, 2, 3, 5, 5, 6, 7, 8, 9], dtype=np.float32)
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(X_train, y_train)

        X_test = np.array([1.5, 3.0, 5.5, 8.0], dtype=np.float32)
        onx = to_onnx(ir, (X_test,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_test})[0]
        expected = ir.predict(X_test).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_constant_model(self):
        """When all X values are identical K=1 — constant prediction."""
        from sklearn.isotonic import IsotonicRegression
        from yobx.sklearn import to_onnx

        X_train = np.array([3.0, 3.0, 3.0], dtype=np.float32)
        y_train = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(X_train, y_train)

        # K==1 (single threshold)
        self.assertEqual(len(ir.X_thresholds_), 1)

        X_test = np.array([1.0, 3.0, 5.0], dtype=np.float32)
        onx = to_onnx(ir, (X_test,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_test})[0]
        expected = ir.predict(X_test).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_float64(self):
        """Converter should preserve float64 precision."""
        from sklearn.isotonic import IsotonicRegression
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(42)
        X_train = np.sort(rng.standard_normal(20)).astype(np.float64)
        y_train = rng.standard_normal(20)
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(X_train, y_train)

        X_test = np.sort(rng.standard_normal(10)).astype(np.float64)
        onx = to_onnx(ir, (X_test,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_test})[0]
        expected = ir.predict(X_test)
        self.assertEqualArray(expected, result, atol=1e-10)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_test})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
