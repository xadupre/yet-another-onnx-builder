"""
Unit tests for the OutputCodeClassifier ONNX converter.
"""

import unittest
import numpy as np
from sklearn.multiclass import OutputCodeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnOutputCodeClassifier(ExtTestCase):
    """Tests for the OutputCodeClassifier converter.

    The converter uses ``predict_proba[:, 1]`` for all binary sub-estimators,
    which matches sklearn exactly for classifiers that do not expose
    ``decision_function`` (e.g. :class:`~sklearn.tree.DecisionTreeClassifier`).
    """

    _X_bin = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    _y_bin = np.array([0, 0, 1, 1])

    _X_multi = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
    _y_multi = np.array([0, 0, 1, 1, 2, 2])

    def test_binary_decision_tree_float32(self):
        """OutputCodeClassifier wrapping binary DecisionTreeClassifier (float32)."""
        clf = OutputCodeClassifier(
            DecisionTreeClassifier(max_depth=3, random_state=0), code_size=2, random_state=0
        )
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

    def test_multiclass_decision_tree_float32(self):
        clf = OutputCodeClassifier(
            DecisionTreeClassifier(max_depth=3, random_state=0), code_size=2, random_state=0
        )
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

    def test_multiclass_decision_tree_float64(self):
        """OutputCodeClassifier wrapping DecisionTreeClassifier with float64 input."""
        X64 = self._X_multi.astype(np.float64)
        clf = OutputCodeClassifier(
            DecisionTreeClassifier(max_depth=3, random_state=0), code_size=2, random_state=0
        )
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

    def test_pipeline_with_output_code(self):
        """OutputCodeClassifier at the end of a Pipeline."""
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    OutputCodeClassifier(
                        DecisionTreeClassifier(max_depth=3, random_state=0),
                        code_size=2,
                        random_state=0,
                    ),
                ),
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
        """OutputCodeClassifier with a sub-estimator without predict_proba raises."""
        from sklearn.svm import LinearSVC

        clf = OutputCodeClassifier(LinearSVC(max_iter=1000), code_size=2, random_state=0)
        clf.fit(self._X_multi, self._y_multi)

        self.assertRaises(NotImplementedError, to_onnx, clf, (self._X_multi,))

    def test_graph_structure_multiclass(self):
        """Check ONNX graph contains expected op types for multiclass OutputCodeClassifier."""
        clf = OutputCodeClassifier(
            DecisionTreeClassifier(max_depth=3, random_state=0), code_size=2, random_state=0
        )
        clf.fit(self._X_multi, self._y_multi)

        onx = to_onnx(clf, (self._X_multi,), target_opset=18)

        op_types = [n.op_type for n in onx.proto.graph.node]
        # One Slice per sub-estimator
        self.assertIn("Slice", op_types)
        # Distance computation (standard path, no com.microsoft)
        self.assertIn("MatMul", op_types)
        self.assertIn("ReduceSum", op_types)
        # Argmin and label gathering
        self.assertIn("ArgMin", op_types)
        self.assertIn("Gather", op_types)

    # ------------------------------------------------------------------
    # CDist path (com.microsoft opset)
    # ------------------------------------------------------------------

    def test_cdist_float32(self):
        """Distance computed via com.microsoft.CDist (float32)."""
        clf = OutputCodeClassifier(
            DecisionTreeClassifier(max_depth=3, random_state=0), code_size=2, random_state=0
        )
        clf.fit(self._X_multi, self._y_multi)

        onx = to_onnx(clf, (self._X_multi,), target_opset={"": 18, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.proto.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types)

        domains = {oi.domain for oi in onx.proto.opset_import}
        self.assertIn("com.microsoft", domains)

        expected_label = clf.predict(self._X_multi)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": self._X_multi})
        self.assertEqualArray(expected_label, ort_results[0])

    def test_cdist_float64(self):
        """Distance computed via com.microsoft.CDist (float64)."""
        X64 = self._X_multi.astype(np.float64)
        clf = OutputCodeClassifier(
            DecisionTreeClassifier(max_depth=3, random_state=0), code_size=2, random_state=0
        )
        clf.fit(X64, self._y_multi)

        onx = to_onnx(clf, (X64,), target_opset={"": 18, "com.microsoft": 1})

        op_types = [(n.op_type, n.domain) for n in onx.proto.graph.node]
        self.assertIn(("CDist", "com.microsoft"), op_types)

        expected_label = clf.predict(X64)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X64})
        self.assertEqualArray(expected_label, ort_results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
