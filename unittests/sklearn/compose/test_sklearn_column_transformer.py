"""
Unit tests for yobx.sklearn ColumnTransformer converter.
"""

import unittest
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx, ConvertOptions


@requires_sklearn("1.4")
class TestSklearnColumnTransformer(ExtTestCase):
    def test_column_transformer_scaler_and_passthrough(self):
        X = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32
        )
        ct = ColumnTransformer(
            [("scaler", StandardScaler(), [0, 1]), ("passthrough", "passthrough", [2, 3])]
        )
        ct.fit(X)

        onx = to_onnx(ct, (X,))

        # Gather + Sub + Div (scaler) + Concat nodes expected
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Gather", op_types)
        self.assertIn("Concat", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = ct.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_column_transformer_two_scalers(self):
        X = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32
        )
        ct = ColumnTransformer(
            [("std", StandardScaler(), [0, 1]), ("mm", MinMaxScaler(), [2, 3])]
        )
        ct.fit(X)

        onx = to_onnx(ct, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = ct.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_column_transformer_drop(self):
        X = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32
        )
        ct = ColumnTransformer(
            [("scaler", StandardScaler(), [0, 1]), ("drop_cols", "drop", [2, 3])]
        )
        ct.fit(X)

        onx = to_onnx(ct, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = ct.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)
        self.assertEqual(result.shape[1], 2)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_column_transformer_remainder_passthrough(self):
        X = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32
        )
        ct = ColumnTransformer([("scaler", StandardScaler(), [0, 1])], remainder="passthrough")
        ct.fit(X)

        onx = to_onnx(ct, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = ct.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)
        # scaler outputs 2 cols, remainder passes through 2 more
        self.assertEqual(result.shape[1], 4)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_column_transformer_slice_columns(self):
        X = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32
        )
        ct = ColumnTransformer(
            [("std", StandardScaler(), slice(0, 2)), ("mm", MinMaxScaler(), slice(2, 4))]
        )
        ct.fit(X)

        onx = to_onnx(ct, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = ct.transform(X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_pipeline_with_column_transformer(self):
        X = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [2, 3, 4, 5],
                [6, 7, 8, 9],
            ],
            dtype=np.float32,
        )
        y = np.array([0, 0, 1, 1, 0, 1])
        ct = ColumnTransformer([("scaler", StandardScaler(), [0, 1, 2, 3])])
        pipe = Pipeline([("preprocessor", ct), ("clf", LogisticRegression())])
        pipe.fit(X, y)

        onx = to_onnx(pipe, (X,))

        ref = ExtendedReferenceEvaluator(onx)
        results = ref.run(None, {"X": X})
        label, proba = results[0], results[1]

        expected_label = pipe.predict(X)
        expected_proba = pipe.predict_proba(X).astype(np.float32)
        self.assertEqualArray(expected_label, label)
        self.assertEqualArray(expected_proba, proba, atol=1e-5)

        sess = self.check_ort(onx)
        ort_results = sess.run(None, {"X": X})
        self.assertEqualArray(expected_label, ort_results[0])
        self.assertEqualArray(expected_proba, ort_results[1], atol=1e-5)


@requires_sklearn("1.4")
class TestSklearnColumnTransformerConvertOptions(ExtTestCase):
    """Tests that ConvertOptions (decision_leaf, decision_path) work inside a Pipeline
    that contains a ColumnTransformer preprocessor."""

    def setUp(self):
        X, y = make_classification(n_samples=100, n_features=4, random_state=0)
        X = X.astype(np.float32)
        self.X_train, self.X_test, y_train, _ = train_test_split(
            X, y, test_size=0.4, random_state=0
        )
        ct = ColumnTransformer(
            [("std", StandardScaler(), [0, 1]), ("mm", MinMaxScaler(), [2, 3])]
        )
        self.pipe = Pipeline(
            [("preprocessor", ct), ("clf", DecisionTreeClassifier(max_depth=3, random_state=0))]
        )
        self.pipe.fit(self.X_train, y_train)

    def test_pipeline_with_column_transformer_decision_leaf(self):
        """ConvertOptions(decision_leaf=True) should add a leaf-index output when a
        Pipeline combines a ColumnTransformer preprocessor with a DecisionTreeClassifier."""
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

        # Values should match sklearn's apply() on the preprocessed features
        X_test_preprocessed = self.pipe.named_steps["preprocessor"].transform(self.X_test)
        expected_leaves = self.pipe.named_steps["clf"].apply(X_test_preprocessed).reshape(-1, 1)
        np.testing.assert_array_equal(ort_out[2], expected_leaves)

    def test_pipeline_with_column_transformer_decision_path(self):
        """ConvertOptions(decision_path=True) should add a path-string output when a
        Pipeline combines a ColumnTransformer preprocessor with a DecisionTreeClassifier."""
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
