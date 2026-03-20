"""
Unit tests for the LocalOutlierFactor converter.
"""

import unittest
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnLocalOutlierFactor(ExtTestCase):
    # Training data with one clear outlier.
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
    # Test data distinct from training data to avoid self-distance effects.
    _X_test = np.array([[1.5, 2.5], [3.5, 4.5], [50.0, 100.0]], dtype=np.float64)  # clear outlier

    def _check(self, X, X_test, dtype, atol=1e-5, n_neighbors=3, **kwargs):
        Xd = X.astype(dtype)
        Xtest = X_test.astype(dtype)
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, **kwargs)
        clf.fit(Xd)
        onx = to_onnx(clf, (Xd,))

        output_names = [o.name for o in onx.proto.graph.output]
        self.assertEqual(len(output_names), 2, f"Expected 2 outputs, got {output_names}")

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": Xtest})
        label, scores = results[0], results[1]

        expected_label = clf.predict(Xtest)
        expected_scores = clf.decision_function(Xtest).astype(dtype)

        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_scores, scores.ravel(), atol=atol)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": Xtest})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_scores, ort_results[1].ravel(), atol=atol)

    # ── float32 / float64 basic tests ────────────────────────────────────────

    def test_default_float32(self):
        """LocalOutlierFactor with default parameters (float32)."""
        self._check(self._X, self._X_test, np.float32)

    def test_default_float64(self):
        """LocalOutlierFactor with default parameters (float64)."""
        self._check(self._X, self._X_test, np.float64)

    # ── n_neighbors ──────────────────────────────────────────────────────────

    def test_n_neighbors_float32(self):
        """LocalOutlierFactor with n_neighbors=5 (float32)."""
        self._check(self._X, self._X_test, np.float32, n_neighbors=5)

    def test_n_neighbors_float64(self):
        """LocalOutlierFactor with n_neighbors=5 (float64)."""
        self._check(self._X, self._X_test, np.float64, n_neighbors=5)

    # ── contamination ────────────────────────────────────────────────────────

    def test_contamination_float32(self):
        """LocalOutlierFactor with explicit contamination fraction (float32)."""
        self._check(self._X, self._X_test, np.float32, contamination=0.1)

    def test_contamination_float64(self):
        """LocalOutlierFactor with explicit contamination fraction (float64)."""
        self._check(self._X, self._X_test, np.float64, contamination=0.1)

    # ── novelty=False guard ──────────────────────────────────────────────────

    def test_novelty_false_raises(self):
        """Converter must raise if novelty=False."""
        Xd = self._X.astype(np.float32)
        clf = LocalOutlierFactor(n_neighbors=3, novelty=False)
        clf.fit(Xd)
        with self.assertRaises(ValueError):
            to_onnx(clf, (Xd,))

    # ── pipeline ─────────────────────────────────────────────────────────────

    def test_in_pipeline_float32(self):
        """LocalOutlierFactor as last step in a Pipeline (float32)."""
        Xd = self._X.astype(np.float32)
        Xtest = self._X_test.astype(np.float32)
        clf = LocalOutlierFactor(n_neighbors=3, novelty=True)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(Xd)

        onx = to_onnx(pipe, (Xd,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": Xtest})
        expected_label = pipe.predict(Xtest)
        self.assertEqualArray(expected_label, results[0])

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": Xtest})
        self.assertEqualArray(expected_label, ort_results[0])

    def test_in_pipeline_float64(self):
        """LocalOutlierFactor as last step in a Pipeline (float64)."""
        Xd = self._X.astype(np.float64)
        Xtest = self._X_test.astype(np.float64)
        clf = LocalOutlierFactor(n_neighbors=3, novelty=True)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(Xd)

        onx = to_onnx(pipe, (Xd,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": Xtest})
        expected_label = pipe.predict(Xtest)
        self.assertEqualArray(expected_label, results[0])

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": Xtest})
        self.assertEqualArray(expected_label, ort_results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
