"""
Unit tests for the LinearDiscriminantAnalysis converter.
"""

import unittest
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnLinearDiscriminantAnalysis(ExtTestCase):

    _X_bin = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [0, 1], [9, 10]], dtype=np.float32)
    _y_bin = np.array([0, 0, 1, 1, 0, 1])

    _X_multi = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
    _y_multi = np.array([0, 0, 1, 1, 2, 2])

    def _check_lda(self, X, y, atol=1e-5):
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            lda = LinearDiscriminantAnalysis()
            lda.fit(Xd, y)
            onx = to_onnx(lda, (Xd,))

            # Two outputs: label and probabilities
            output_names = [o.name for o in onx.proto.graph.output]
            self.assertEqual(len(output_names), 2, f"Expected 2 outputs, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            label, proba = results[0], results[1]

            expected_label = lda.predict(Xd)
            expected_proba = lda.predict_proba(Xd).astype(dtype)

            self.assertEqualArray(expected_label, label)
            self.assertEqualArray(expected_proba, proba, atol=atol)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_label, ort_results[0])
            self.assertEqualArray(expected_proba, ort_results[1], atol=atol)

    def test_lda_binary(self):
        self._check_lda(self._X_bin, self._y_bin)

    def test_lda_multiclass(self):
        self._check_lda(self._X_multi, self._y_multi)

    def test_lda_in_pipeline(self):
        """LDA at the end of a Pipeline."""
        for dtype in (np.float32, np.float64):
            Xd = self._X_multi.astype(dtype)
            y = self._y_multi
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", LinearDiscriminantAnalysis())])
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
