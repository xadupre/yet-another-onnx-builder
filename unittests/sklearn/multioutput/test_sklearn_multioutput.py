"""
Unit tests for the MultiOutputClassifier and MultiOutputRegressor ONNX converters.
"""

import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnMultiOutputRegressor(ExtTestCase):
    """Tests for the MultiOutputRegressor converter."""

    _X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    _y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    def test_ridge_two_targets(self):
        """MultiOutputRegressor wrapping Ridge with 2 targets."""
        reg = MultiOutputRegressor(Ridge())
        reg.fit(self._X, self._y)

        onx = to_onnx(reg, (self._X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X})
        predictions = results[0]

        expected = reg.predict(self._X).astype(np.float32)
        self.assertEqualArray(expected, predictions, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": self._X})
        self.assertEqualArray(expected, ort_results[0], atol=1e-5)

    def test_three_targets(self):
        """MultiOutputRegressor with 3 targets."""
        y3 = np.column_stack([self._y, self._y[:, 0] * 2])
        reg = MultiOutputRegressor(Ridge())
        reg.fit(self._X, y3)

        onx = to_onnx(reg, (self._X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X})
        predictions = results[0]

        expected = reg.predict(self._X).astype(np.float32)
        self.assertEqualArray(expected, predictions, atol=1e-5)
        self.assertEqual(predictions.shape, (4, 3))

    def test_float64_input(self):
        """MultiOutputRegressor works with float64 input."""
        X64 = self._X.astype(np.float64)
        y64 = self._y.astype(np.float64)
        reg = MultiOutputRegressor(Ridge())
        reg.fit(X64, y64)

        onx = to_onnx(reg, (X64,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X64})
        predictions = results[0]

        expected = reg.predict(X64)
        self.assertEqualArray(expected, predictions, atol=1e-5)

    def test_decision_tree_regressor(self):
        """MultiOutputRegressor wrapping DecisionTreeRegressor."""
        reg = MultiOutputRegressor(DecisionTreeRegressor(max_depth=2, random_state=0))
        reg.fit(self._X, self._y)

        onx = to_onnx(reg, (self._X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X})
        predictions = results[0]

        expected = reg.predict(self._X).astype(np.float32)
        self.assertEqualArray(expected, predictions, atol=1e-5)

    def test_pipeline_with_multi_output_regressor(self):
        """MultiOutputRegressor at the end of a Pipeline."""
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", MultiOutputRegressor(Ridge()))])
        pipe.fit(self._X, self._y)

        onx = to_onnx(pipe, (self._X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X})
        predictions = results[0]

        expected = pipe.predict(self._X).astype(np.float32)
        self.assertEqualArray(expected, predictions, atol=1e-5)

    def test_graph_structure(self):
        """Check ONNX graph contains expected op types for MultiOutputRegressor."""
        reg = MultiOutputRegressor(Ridge())
        reg.fit(self._X, self._y)

        onx = to_onnx(reg, (self._X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        # One Gemm per sub-estimator (Ridge → Gemm)
        self.assertEqual(op_types.count("Gemm"), 2)
        # Two Reshape ops (one per target)
        self.assertEqual(op_types.count("Reshape"), 2)
        # One Concat to combine targets
        self.assertIn("Concat", op_types)


@requires_sklearn("1.4")
class TestSklearnMultiOutputClassifier(ExtTestCase):
    """Tests for the MultiOutputClassifier converter."""

    _X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
    _y = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 0], [1, 1]])

    def test_logistic_regression_two_targets(self):
        """MultiOutputClassifier wrapping LogisticRegression with 2 binary targets."""
        clf = MultiOutputClassifier(LogisticRegression(max_iter=200))
        clf.fit(self._X, self._y)

        onx = to_onnx(clf, (self._X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X})
        labels, probabilities = results[0], results[1]

        expected_labels = clf.predict(self._X)
        expected_probas = clf.predict_proba(self._X)

        self.assertEqualArray(expected_labels, labels)
        self.assertEqual(labels.shape, (6, 2))

        # probabilities shape: (N, n_targets, n_classes)
        self.assertEqual(probabilities.shape, (6, 2, 2))
        for i, ep in enumerate(expected_probas):
            self.assertEqualArray(ep.astype(np.float32), probabilities[:, i, :], atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": self._X})
        self.assertEqualArray(expected_labels, ort_results[0])
        for i, ep in enumerate(expected_probas):
            self.assertEqualArray(ep.astype(np.float32), ort_results[1][:, i, :], atol=1e-5)

    def test_float64_input(self):
        """MultiOutputClassifier works with float64 input."""
        X64 = self._X.astype(np.float64)
        clf = MultiOutputClassifier(LogisticRegression(max_iter=200))
        clf.fit(X64, self._y)

        onx = to_onnx(clf, (X64,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X64})
        labels, probabilities = results[0], results[1]

        expected_labels = clf.predict(X64)
        expected_probas = clf.predict_proba(X64)

        self.assertEqualArray(expected_labels, labels)
        for i, ep in enumerate(expected_probas):
            self.assertEqualArray(ep, probabilities[:, i, :], atol=1e-5)

    def test_decision_tree_classifier(self):
        """MultiOutputClassifier wrapping DecisionTreeClassifier."""
        clf = MultiOutputClassifier(DecisionTreeClassifier(max_depth=2, random_state=0))
        clf.fit(self._X, self._y)

        onx = to_onnx(clf, (self._X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X})
        labels = results[0]

        expected_labels = clf.predict(self._X)
        self.assertEqualArray(expected_labels, labels)

    def test_pipeline_with_multi_output_classifier(self):
        """MultiOutputClassifier at the end of a Pipeline."""
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", MultiOutputClassifier(LogisticRegression(max_iter=200))),
            ]
        )
        pipe.fit(self._X, self._y)

        onx = to_onnx(pipe, (self._X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X})
        labels, probabilities = results[0], results[1]

        expected_labels = pipe.predict(self._X)
        expected_probas = pipe.predict_proba(self._X)

        self.assertEqualArray(expected_labels, labels)
        for i, ep in enumerate(expected_probas):
            self.assertEqualArray(ep.astype(np.float32), probabilities[:, i, :], atol=1e-5)

    def test_different_n_classes_raises(self):
        """MultiOutputClassifier with different n_classes per target should raise."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5]], dtype=np.float32)
        # Target 0: 2 classes, Target 1: 3 classes
        y = np.array([[0, 0], [1, 1], [0, 2], [1, 0], [0, 1], [1, 2]])
        clf = MultiOutputClassifier(LogisticRegression(max_iter=200))
        clf.fit(X, y)

        self.assertRaises(NotImplementedError, to_onnx, clf, (X,))

    def test_graph_structure(self):
        """Check ONNX graph contains expected op types for MultiOutputClassifier."""
        clf = MultiOutputClassifier(LogisticRegression(max_iter=200))
        clf.fit(self._X, self._y)

        onx = to_onnx(clf, (self._X,))

        op_types = [n.op_type for n in onx.proto.graph.node]
        # One Gemm per sub-estimator (LogisticRegression → Gemm)
        self.assertEqual(op_types.count("Gemm"), 2)
        # ArgMax to derive label indices (one per sub-estimator)
        self.assertEqual(op_types.count("ArgMax"), 2)
        # Gather to map indices to class values (one per sub-estimator)
        self.assertEqual(op_types.count("Gather"), 2)
        # At least two Concat nodes (one for labels, one for probabilities)
        self.assertGreaterEqual(op_types.count("Concat"), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
