"""
Unit tests for the QuadraticDiscriminantAnalysis converter.
"""

import unittest
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnQuadraticDiscriminantAnalysis(ExtTestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(42)
        # Binary: 2 well-separated classes with enough samples to avoid singular covariance
        cls._X_bin = rng.randn(30, 4).astype(np.float32)
        cls._X_bin[:15] += 2.0
        cls._y_bin = np.array([0] * 15 + [1] * 15)

        # Multiclass: 3 classes
        cls._X_multi = rng.randn(30, 4).astype(np.float32)
        cls._X_multi[:10] += np.array([3, 0, 0, 0], dtype=np.float32)
        cls._X_multi[10:20] += np.array([0, 3, 0, 0], dtype=np.float32)
        cls._y_multi = np.array([0] * 10 + [1] * 10 + [2] * 10)

    def _check_qda(self, X, y, atol=1e-4):
        """Run the converter for both float32 and float64, check label and proba."""
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            qda = QuadraticDiscriminantAnalysis()
            qda.fit(Xd, y)
            onx = to_onnx(qda, (Xd,))

            # Two outputs: label and probabilities
            output_names = [o.name for o in onx.graph.output]
            self.assertEqual(len(output_names), 2, f"Expected 2 outputs, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            label, proba = results[0], results[1]

            expected_label = qda.predict(Xd)
            expected_proba = qda.predict_proba(Xd).astype(dtype)

            self.assertEqualArray(expected_label, label)
            self.assertEqualArray(expected_proba, proba, atol=atol)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_label, ort_results[0])
            self.assertEqualArray(expected_proba, ort_results[1], atol=atol)

    def test_qda_binary_float32(self):
        """Binary QDA, float32 input."""
        self._check_qda(self._X_bin, self._y_bin)

    def test_qda_binary_float64(self):
        """Binary QDA, float64 input."""
        self._check_qda(self._X_bin.astype(np.float64), self._y_bin)

    def test_qda_multiclass_float32(self):
        """Multiclass QDA (3 classes), float32 input."""
        self._check_qda(self._X_multi, self._y_multi)

    def test_qda_multiclass_float64(self):
        """Multiclass QDA (3 classes), float64 input."""
        self._check_qda(self._X_multi.astype(np.float64), self._y_multi)

    def test_qda_with_reg_param(self):
        """QDA with regularization (reg_param > 0)."""
        rng = np.random.RandomState(7)
        X = rng.randn(20, 3).astype(np.float32)
        y = np.array([0] * 10 + [1] * 10)
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
            qda.fit(Xd, y)
            onx = to_onnx(qda, (Xd,))
            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(qda.predict(Xd), ort_results[0])

    def test_qda_in_pipeline(self):
        """QDA at the end of a Pipeline."""
        for dtype in (np.float32, np.float64):
            Xd = self._X_multi.astype(dtype)
            y = self._y_multi
            pipe = Pipeline(
                [("scaler", StandardScaler()), ("clf", QuadraticDiscriminantAnalysis())]
            )
            pipe.fit(Xd, y)
            onx = to_onnx(pipe, (Xd,))

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            expected_label = pipe.predict(Xd)
            self.assertEqualArray(expected_label, results[0])

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_label, ort_results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
