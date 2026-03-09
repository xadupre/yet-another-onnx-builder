"""
Unit tests for the OneVsRestClassifier ONNX converter.
"""

import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnOneVsRestClassifier(ExtTestCase):
    """Tests for the OneVsRestClassifier converter."""

    _X_bin = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    _y_bin = np.array([0, 0, 1, 1])

    _X_multi = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
    _y_multi = np.array([0, 0, 1, 1, 2, 2])

    def test_binary_logistic_regression(self):
        """OVR wrapping binary LogisticRegression."""
        clf = OneVsRestClassifier(LogisticRegression(max_iter=200))
        clf.fit(self._X_bin, self._y_bin)

        onx = to_onnx(clf, (self._X_bin,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X_bin})
        label, proba = results[0], results[1]

        expected_label = clf.predict(self._X_bin)
        expected_proba = clf.predict_proba(self._X_bin).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": self._X_bin})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_multiclass_logistic_regression(self):
        """OVR wrapping multi-class LogisticRegression (3 classes)."""
        clf = OneVsRestClassifier(LogisticRegression(max_iter=200))
        clf.fit(self._X_multi, self._y_multi)

        onx = to_onnx(clf, (self._X_multi,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X_multi})
        label, proba = results[0], results[1]

        expected_label = clf.predict(self._X_multi)
        expected_proba = clf.predict_proba(self._X_multi).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": self._X_multi})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_float64_input(self):
        """OVR converter works with float64 input."""
        X64 = self._X_multi.astype(np.float64)
        clf = OneVsRestClassifier(LogisticRegression(max_iter=200))
        clf.fit(X64, self._y_multi)

        onx = to_onnx(clf, (X64,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X64})
        label, proba = results[0], results[1]

        expected_label = clf.predict(X64)
        expected_proba = clf.predict_proba(X64).astype(np.float64)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X64})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_pipeline_with_one_vs_rest(self):
        """OVR at the end of a Pipeline."""
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", OneVsRestClassifier(LogisticRegression(max_iter=200))),
            ]
        )
        pipe.fit(self._X_multi, self._y_multi)

        onx = to_onnx(pipe, (self._X_multi,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X_multi})
        label, proba = results[0], results[1]

        expected_label = pipe.predict(self._X_multi)
        expected_proba = pipe.predict_proba(self._X_multi).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": self._X_multi})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_multilabel_raises(self):
        """OVR with multilabel_ = True should raise NotImplementedError."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        # Multi-label target: each sample can belong to multiple classes
        y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        clf = OneVsRestClassifier(LogisticRegression(max_iter=200))
        clf.fit(X, y)

        self.assertTrue(clf.multilabel_)
        self.assertRaises(NotImplementedError, to_onnx, clf, (X,))

    def test_multiclass_decision_tree_18(self):
        """OVR wrapping DecisionTreeClassifier (3 classes)."""
        clf = OneVsRestClassifier(DecisionTreeClassifier(max_depth=3, random_state=0))
        clf.fit(self._X_multi, self._y_multi)

        onx = to_onnx(clf, (self._X_multi,), target_opset=18)

        # Each sub-estimator should produce a TreeEnsembleClassifier node.
        op_types = [n.op_type for n in onx.graph.node]
        self.assertEqual(op_types.count("TreeEnsembleClassifier"), 3)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X_multi})
        label, proba = results[0], results[1]

        expected_label = clf.predict(self._X_multi)
        expected_proba = clf.predict_proba(self._X_multi).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": self._X_multi})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_multiclass_decision_tree_21(self):
        """OVR wrapping DecisionTreeClassifier (3 classes)."""
        clf = OneVsRestClassifier(DecisionTreeClassifier(max_depth=3, random_state=0))
        clf.fit(self._X_multi, self._y_multi)

        onx = to_onnx(clf, (self._X_multi,), target_opset=21)

        # Each sub-estimator should produce a TreeEnsembleClassifier node.
        op_types = [n.op_type for n in onx.graph.node]
        self.assertEqual(op_types.count("TreeEnsemble"), 3)

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X_multi})
        label, proba = results[0], results[1]

        expected_label = clf.predict(self._X_multi)
        expected_proba = clf.predict_proba(self._X_multi).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": self._X_multi})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)

    def test_graph_structure_multiclass(self):
        """Check ONNX graph contains expected op types for multiclass OVR."""
        clf = OneVsRestClassifier(LogisticRegression(max_iter=200))
        clf.fit(self._X_multi, self._y_multi)

        onx = to_onnx(clf, (self._X_multi,))

        op_types = [n.op_type for n in onx.graph.node]
        # One Gemm per sub-estimator (3 for 3 classes)
        self.assertEqual(op_types.count("Gemm"), 3)
        # One Slice per sub-estimator
        self.assertEqual(op_types.count("Slice"), 3)
        # One Concat to stack scores
        self.assertIn("Concat", op_types)
        # Normalisation
        self.assertIn("ReduceSum", op_types)
        self.assertIn("Div", op_types)
        # Label path
        self.assertIn("ArgMax", op_types)
        self.assertIn("Gather", op_types)


if __name__ == "__main__":
    unittest.main(verbosity=2)
