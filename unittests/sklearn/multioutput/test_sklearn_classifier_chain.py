"""
Unit tests for the ClassifierChain ONNX converter.
Tests cover float32 and float64 inputs, identity and non-identity chain orders,
single/multiple targets, and optional probability outputs.
"""

import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx


@requires_sklearn("1.4")
class TestSklearnClassifierChain(ExtTestCase):
    """Tests for the ClassifierChain converter."""

    _X = np.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [2, 3], [4, 5], [1, 3], [5, 2]], dtype=np.float32
    )
    _y = np.array(
        [[0, 1, 1], [1, 0, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0], [0, 1, 0], [1, 0, 1]]
    )

    def test_float32_identity_order(self):
        """ClassifierChain with float32 input and default (identity) order."""
        cc = ClassifierChain(LogisticRegression(max_iter=200))
        cc.fit(self._X, self._y)

        onx = to_onnx(cc, (self._X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X})
        labels, probabilities = results[0], results[1]

        expected_labels = cc.predict(self._X).astype(np.float32)
        expected_probas = cc.predict_proba(self._X).astype(np.float32)

        self.assertEqualArray(expected_labels, labels, atol=1e-5)
        self.assertEqualArray(expected_probas, probabilities, atol=1e-5)
        self.assertEqual(labels.shape, (8, 3))
        self.assertEqual(probabilities.shape, (8, 3))

    def test_float64_identity_order(self):
        """ClassifierChain with float64 input and default (identity) order."""
        X64 = self._X.astype(np.float64)
        cc = ClassifierChain(LogisticRegression(max_iter=200))
        cc.fit(X64, self._y)

        onx = to_onnx(cc, (X64,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X64})
        labels, probabilities = results[0], results[1]

        expected_labels = cc.predict(X64)
        expected_probas = cc.predict_proba(X64)

        self.assertEqualArray(expected_labels, labels, atol=1e-5)
        self.assertEqualArray(expected_probas, probabilities, atol=1e-5)

    def test_float32_custom_order(self):
        """ClassifierChain with float32 input and non-identity order [2, 0, 1]."""
        cc = ClassifierChain(LogisticRegression(max_iter=200), order=[2, 0, 1])
        cc.fit(self._X, self._y)

        onx = to_onnx(cc, (self._X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X})
        labels, probabilities = results[0], results[1]

        expected_labels = cc.predict(self._X).astype(np.float32)
        expected_probas = cc.predict_proba(self._X).astype(np.float32)

        self.assertEqualArray(expected_labels, labels, atol=1e-5)
        self.assertEqualArray(expected_probas, probabilities, atol=1e-5)

    def test_float64_custom_order(self):
        """ClassifierChain with float64 input and non-identity order [2, 0, 1]."""
        X64 = self._X.astype(np.float64)
        cc = ClassifierChain(LogisticRegression(max_iter=200), order=[2, 0, 1])
        cc.fit(X64, self._y)

        onx = to_onnx(cc, (X64,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X64})
        labels, probabilities = results[0], results[1]

        expected_labels = cc.predict(X64)
        expected_probas = cc.predict_proba(X64)

        self.assertEqualArray(expected_labels, labels, atol=1e-5)
        self.assertEqualArray(expected_probas, probabilities, atol=1e-5)

    def test_random_order(self):
        """ClassifierChain with float32 input and random chain order."""
        cc = ClassifierChain(LogisticRegression(max_iter=200), order="random", random_state=42)
        cc.fit(self._X, self._y)

        onx = to_onnx(cc, (self._X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X})
        labels, probabilities = results[0], results[1]

        expected_labels = cc.predict(self._X).astype(np.float32)
        expected_probas = cc.predict_proba(self._X).astype(np.float32)

        self.assertEqualArray(expected_labels, labels, atol=1e-5)
        self.assertEqualArray(expected_probas, probabilities, atol=1e-5)

    def test_decision_tree_classifier(self):
        """ClassifierChain wrapping DecisionTreeClassifier."""
        cc = ClassifierChain(DecisionTreeClassifier(max_depth=2, random_state=0))
        cc.fit(self._X, self._y)

        onx = to_onnx(cc, (self._X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X})
        labels = results[0]

        expected_labels = cc.predict(self._X).astype(np.float32)
        self.assertEqualArray(expected_labels, labels, atol=1e-5)

    def test_pipeline_with_classifier_chain(self):
        """ClassifierChain at the end of a Pipeline."""
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("cc", ClassifierChain(LogisticRegression(max_iter=200))),
            ]
        )
        pipe.fit(self._X, self._y)

        onx = to_onnx(pipe, (self._X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": self._X})
        labels, probabilities = results[0], results[1]

        expected_labels = pipe.predict(self._X).astype(np.float32)
        expected_probas = pipe.predict_proba(self._X).astype(np.float32)

        self.assertEqualArray(expected_labels, labels, atol=1e-5)
        self.assertEqualArray(expected_probas, probabilities, atol=1e-5)

    def test_ort_float32(self):
        """Validate ONNX Runtime output for float32."""
        cc = ClassifierChain(LogisticRegression(max_iter=200))
        cc.fit(self._X, self._y)

        onx = to_onnx(cc, (self._X,))

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": self._X})
        labels, probabilities = ort_results[0], ort_results[1]

        expected_labels = cc.predict(self._X).astype(np.float32)
        expected_probas = cc.predict_proba(self._X).astype(np.float32)

        self.assertEqualArray(expected_labels, labels, atol=1e-5)
        self.assertEqualArray(expected_probas, probabilities, atol=1e-5)

    def test_graph_structure(self):
        """Check ONNX graph contains expected op types for ClassifierChain."""
        cc = ClassifierChain(LogisticRegression(max_iter=200))
        cc.fit(self._X, self._y)

        onx = to_onnx(cc, (self._X,))

        op_types = [n.op_type for n in onx.graph.node]
        # One Gemm per sub-estimator (LogisticRegression → Gemm)
        self.assertEqual(op_types.count("Gemm"), 3)
        # Concat nodes to augment input features for subsequent chain steps
        self.assertGreaterEqual(op_types.count("Concat"), 2)
        # Cast nodes to convert int64 labels to float for chain augmentation
        self.assertGreaterEqual(op_types.count("Cast"), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
