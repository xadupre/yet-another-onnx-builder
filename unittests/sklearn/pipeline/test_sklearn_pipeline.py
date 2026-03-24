"""
Unit tests for yobx.sklearn Pipeline converter.
"""

import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx, ConvertOptions


@requires_sklearn("1.4")
class TestSklearnPipeline(ExtTestCase):
    def test_pipeline_step_names_are_prefixed(self):
        """Tensor names produced during each step conversion should carry the step prefix."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        # Every intermediate tensor name produced during "scaler" step conversion
        # should include the step prefix.
        all_names = [n.name for n in onx.proto.graph.node]
        all_names += [i.name for i in onx.proto.graph.initializer]
        all_names += [v.name for v in onx.proto.graph.value_info]
        scaler_names = [n for n in all_names if "scaler" in n]
        self.assertTrue(
            len(scaler_names) > 0,
            msg="Expected at least one name containing 'scaler' but found none",
        )

    def test_pipeline_produces_correct_predictions(self):
        """Pipeline converter should produce numerically correct ONNX output."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
        y = np.array([0, 1, 0, 1])
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        expected = pipe.predict(X)
        self.assertEqualArray(expected, results[0])


@requires_sklearn("1.4")
class TestSklearnPipelineConvertOptions(ExtTestCase):
    """Tests that ConvertOptions (decision_leaf, decision_path) work inside a Pipeline."""

    def setUp(self):
        X, y = make_classification(n_samples=100, n_features=4, random_state=0)
        X = X.astype(np.float32)
        self.X_train, self.X_test, y_train, _ = train_test_split(
            X, y, test_size=0.4, random_state=0
        )
        self.pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", DecisionTreeClassifier(max_depth=3, random_state=0)),
            ]
        )
        self.pipe.fit(self.X_train, y_train)

    def test_pipeline_convert_options_decision_leaf(self):
        """ConvertOptions(decision_leaf=True) should add a leaf-index output for a Pipeline
        ending with a DecisionTreeClassifier."""
        onx = to_onnx(
            self.pipe, (self.X_train,), convert_options=ConvertOptions(decision_leaf=True)
        )

        # Three outputs: label, probabilities, decision_leaf
        self.assertEqual(len(onx.graph.output), 3)

        feeds = {onx.graph.input[0].name: self.X_test}
        sess = self.check_ort(onx)
        ort_out = sess.run(None, feeds)
        ref_out = ExtendedReferenceEvaluator(onx).run(None, feeds)

        # Label and probabilities should match sklearn predictions
        expected_labels = self.pipe.predict(self.X_test)
        expected_proba = self.pipe.predict_proba(self.X_test).astype(np.float32)
        np.testing.assert_array_equal(ort_out[0], expected_labels)
        np.testing.assert_allclose(ort_out[1], expected_proba, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(ref_out[0], expected_labels)

        # ORT and reference evaluator should agree on the decision_leaf output
        np.testing.assert_array_equal(ort_out[2], ref_out[2])

        # decision_leaf shape is (n_samples, 1), dtype int64
        self.assertEqual(ort_out[2].ndim, 2)
        self.assertEqual(ort_out[2].shape[0], self.X_test.shape[0])
        self.assertEqual(ort_out[2].shape[1], 1)

        # Values should match sklearn's apply() on the pre-processed features
        X_test_scaled = self.pipe.named_steps["scaler"].transform(self.X_test)
        expected_leaves = self.pipe.named_steps["clf"].apply(X_test_scaled).reshape(-1, 1)
        np.testing.assert_array_equal(ort_out[2], expected_leaves)

    def test_pipeline_convert_options_decision_path(self):
        """ConvertOptions(decision_path=True) should add a binary
        path-string output for a Pipeline ending with a DecisionTreeClassifier."""
        onx = to_onnx(
            self.pipe, (self.X_train,), convert_options=ConvertOptions(decision_path=True)
        )

        # Three outputs: label, probabilities, decision_path
        self.assertEqual(len(onx.graph.output), 3)

        feeds = {onx.graph.input[0].name: self.X_test}
        sess = self.check_ort(onx)
        ort_out = sess.run(None, feeds)
        ref_out = ExtendedReferenceEvaluator(onx).run(None, feeds)

        # Label and probabilities should match sklearn predictions
        expected_labels = self.pipe.predict(self.X_test)
        expected_proba = self.pipe.predict_proba(self.X_test).astype(np.float32)
        np.testing.assert_array_equal(ort_out[0], expected_labels)
        np.testing.assert_allclose(ort_out[1], expected_proba, rtol=1e-5, atol=1e-5)

        # ORT and reference evaluator should agree on the decision_path output
        np.testing.assert_array_equal(ort_out[2], ref_out[2])

        # decision_path shape is (n_samples, 1), dtype string
        self.assertEqual(ort_out[2].ndim, 2)
        self.assertEqual(ort_out[2].shape[0], self.X_test.shape[0])
        self.assertEqual(ort_out[2].shape[1], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
