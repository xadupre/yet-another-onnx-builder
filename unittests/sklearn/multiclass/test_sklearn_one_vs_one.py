"""
Unit tests for the OneVsOneClassifier ONNX converter.
"""

import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnOneVsOneClassifier(ExtTestCase):
    """Tests for the OneVsOneClassifier converter."""

    _X_bin = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    _y_bin = np.array([0, 0, 1, 1])

    _X_multi = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32
    )
    _y_multi = np.array([0, 0, 1, 1, 2, 2])

    def test_binary_logistic_regression_float32(self):
        """OVO wrapping binary LogisticRegression (float32 input)."""
        clf = OneVsOneClassifier(LogisticRegression(max_iter=200))
        clf.fit(self._X_bin, self._y_bin)

        onx = to_onnx(clf, (self._X_bin,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X_bin})
        label = results[0]

        expected_label = clf.predict(self._X_bin)
        self.assertEqualArray(expected_label, label)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": self._X_bin})
        self.assertEqualArray(expected_label, ort_results[0])

    def test_binary_logistic_regression_float64(self):
        """OVO wrapping binary LogisticRegression (float64 input)."""
        X64 = self._X_bin.astype(np.float64)
        clf = OneVsOneClassifier(LogisticRegression(max_iter=200))
        clf.fit(X64, self._y_bin)

        onx = to_onnx(clf, (X64,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X64})
        label = results[0]

        expected_label = clf.predict(X64)
        self.assertEqualArray(expected_label, label)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X64})
        self.assertEqualArray(expected_label, ort_results[0])

    def test_multiclass_logistic_regression_float32(self):
        """OVO wrapping multi-class LogisticRegression (3 classes, float32)."""
        clf = OneVsOneClassifier(LogisticRegression(max_iter=200))
        clf.fit(self._X_multi, self._y_multi)

        onx = to_onnx(clf, (self._X_multi,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X_multi})
        label = results[0]

        expected_label = clf.predict(self._X_multi)
        self.assertEqualArray(expected_label, label)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": self._X_multi})
        self.assertEqualArray(expected_label, ort_results[0])

    def test_multiclass_logistic_regression_float64(self):
        """OVO wrapping multi-class LogisticRegression (3 classes, float64)."""
        X64 = self._X_multi.astype(np.float64)
        clf = OneVsOneClassifier(LogisticRegression(max_iter=200))
        clf.fit(X64, self._y_multi)

        onx = to_onnx(clf, (X64,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X64})
        label = results[0]

        expected_label = clf.predict(X64)
        self.assertEqualArray(expected_label, label)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X64})
        self.assertEqualArray(expected_label, ort_results[0])

    def test_pipeline_with_one_vs_one(self):
        """OVO at the end of a Pipeline."""
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", OneVsOneClassifier(LogisticRegression(max_iter=200))),
            ]
        )
        pipe.fit(self._X_multi, self._y_multi)

        onx = to_onnx(pipe, (self._X_multi,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X_multi})
        label = results[0]

        expected_label = pipe.predict(self._X_multi)
        self.assertEqualArray(expected_label, label)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": self._X_multi})
        self.assertEqualArray(expected_label, ort_results[0])

    def test_no_predict_proba_raises(self):
        """OVO with a sub-estimator that has no predict_proba raises NotImplementedError."""
        from sklearn.svm import LinearSVC

        clf = OneVsOneClassifier(LinearSVC(max_iter=1000))
        clf.fit(self._X_multi, self._y_multi)

        self.assertRaises(NotImplementedError, to_onnx, clf, (self._X_multi,))

    def test_multiclass_decision_tree_float32(self):
        """OVO wrapping DecisionTreeClassifier (3 classes, float32)."""
        clf = OneVsOneClassifier(DecisionTreeClassifier(max_depth=3, random_state=0))
        clf.fit(self._X_multi, self._y_multi)

        onx = to_onnx(clf, (self._X_multi,), target_opset=18)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X_multi})
        label = results[0]

        expected_label = clf.predict(self._X_multi)
        self.assertEqualArray(expected_label, label)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": self._X_multi})
        self.assertEqualArray(expected_label, ort_results[0])

    def test_graph_structure_multiclass(self):
        """Check ONNX graph contains expected op types for multiclass OVO."""
        clf = OneVsOneClassifier(LogisticRegression(max_iter=200))
        clf.fit(self._X_multi, self._y_multi)

        onx = to_onnx(clf, (self._X_multi,))

        op_types = [n.op_type for n in onx.graph.node]
        # 3 classes → 3 pairs → 3 Gemm nodes (one per binary LogisticRegression)
        self.assertEqual(op_types.count("Gemm"), 3)
        # One Slice per pair to extract positive-class prob
        self.assertEqual(op_types.count("Slice"), 3)
        # Tie-breaking path
        self.assertIn("Abs", op_types)
        self.assertIn("Div", op_types)
        # Label path
        self.assertIn("ArgMax", op_types)
        self.assertIn("Gather", op_types)


if __name__ == "__main__":
    unittest.main(verbosity=2)
