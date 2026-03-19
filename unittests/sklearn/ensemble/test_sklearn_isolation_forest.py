"""
Unit tests for the IsolationForest converter.
"""

import unittest
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnIsolationForest(ExtTestCase):
    _X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [2.0, 3.0],
            [4.0, 5.0],
            [0.0, 1.0],
            [9.0, 10.0],
            [100.0, 200.0],  # clear outlier
        ],
        dtype=np.float64,
    )

    def _check(self, X, atol=1e-5, **kwargs):
        for dtype in (np.float32, np.float64):
            Xd = X.astype(dtype)
            clf = IsolationForest(n_estimators=10, random_state=0, **kwargs)
            clf.fit(Xd)
            onx = to_onnx(clf, (Xd,))

            output_names = [o.name for o in onx.proto.graph.output]
            self.assertEqual(len(output_names), 2, f"Expected 2 outputs, got {output_names}")

            ref = ExtendedReferenceEvaluator(onx)
            results = ref.run(None, {"X": Xd})
            label, scores = results[0], results[1]

            expected_label = clf.predict(Xd)
            expected_scores = clf.decision_function(Xd).astype(dtype)

            self.assertEqualArray(expected_label, label)
            self.assertEqualArray(expected_scores, scores.ravel(), atol=atol)

            sess = self.check_ort(onx)
            ort_results = sess.run(None, {"X": Xd})
            self.assertEqualArray(expected_label, ort_results[0])
            self.assertEqualArray(expected_scores, ort_results[1].ravel(), atol=atol)

    def test_default(self):
        """IsolationForest with default parameters."""
        self._check(self._X)

    def test_max_samples(self):
        """IsolationForest with explicit max_samples."""
        self._check(self._X, max_samples=5)

    def test_max_features(self):
        """IsolationForest with feature subsampling (max_features < n_features)."""
        X = np.array(
            [
                [1.0, 2.0, 3.0],
                [3.0, 4.0, 5.0],
                [5.0, 6.0, 7.0],
                [7.0, 8.0, 9.0],
                [2.0, 3.0, 4.0],
                [4.0, 5.0, 6.0],
                [0.0, 1.0, 2.0],
                [9.0, 10.0, 11.0],
                [100.0, 200.0, 300.0],
            ],
            dtype=np.float64,
        )
        self._check(X, max_features=2)

    def test_contamination(self):
        """IsolationForest with explicit contamination fraction."""
        self._check(self._X, contamination=0.1)

    def test_in_pipeline(self):
        """IsolationForest as last step in a Pipeline."""
        Xd = self._X.astype(np.float32)
        clf = IsolationForest(n_estimators=10, random_state=0, max_samples=5)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(Xd)

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
