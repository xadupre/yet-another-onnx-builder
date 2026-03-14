"""
Unit tests for the SGDOneClassSVM ONNX converter.

Covers:
- float32 and float64 inputs
- label and scores outputs
- Pipeline usage
"""

import unittest
import numpy as np
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnSGDOneClassSVM(ExtTestCase):

    _X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [0.5, 1.5, 2.5],
            [3.5, 4.5, 5.5],
            [-1.0, -2.0, -3.0],
            [10.0, 11.0, 12.0],
            [2.0, 3.0, 4.0],
        ],
        dtype=np.float32,
    )

    def _check_sgd_one_class_svm(self, dtype, atol=1e-5):
        X = self._X.astype(dtype)
        clf = SGDOneClassSVM(nu=0.1, random_state=42, max_iter=1000)
        clf.fit(X)
        onx = to_onnx(clf, (X,))

        output_names = [o.name for o in onx.graph.output]
        self.assertEqual(len(output_names), 2, f"Expected 2 outputs (label, scores), got {output_names}")

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})

        # Check labels (sklearn returns int32 but ONNX outputs int64)
        expected_labels = clf.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, ort_results[0])

        # Check scores (decision_function values)
        expected_scores = clf.decision_function(X).astype(dtype)
        self.assertEqualArray(expected_scores, ort_results[1], atol=atol)

    def test_sgd_one_class_svm_float32(self):
        self._check_sgd_one_class_svm(np.float32)

    def test_sgd_one_class_svm_float64(self):
        self._check_sgd_one_class_svm(np.float64)

    def test_sgd_one_class_svm_in_pipeline(self):
        X = self._X.astype(np.float32)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SGDOneClassSVM(nu=0.1, random_state=42, max_iter=1000)),
            ]
        )
        pipe.fit(X)
        onx = to_onnx(pipe, (X,))

        output_names = [o.name for o in onx.graph.output]
        self.assertEqual(len(output_names), 2, f"Expected 2 outputs, got {output_names}")

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})

        expected_labels = pipe.predict(X).astype(np.int64)
        self.assertEqualArray(expected_labels, ort_results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
