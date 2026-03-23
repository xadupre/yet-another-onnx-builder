"""
Unit tests for yobx.sklearn ColumnTransformer converter.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from yobx.sklearn import to_onnx


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

    def test_column_transformer_string_columns_dataframe(self):
        """ColumnTransformer fitted with string column names on a DataFrame."""
        import pandas as pd

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
        df = pd.DataFrame(X, columns=["age", "income", "score", "balance"])
        ct = ColumnTransformer(
            [("std", StandardScaler(), ["age", "income"]), ("mm", MinMaxScaler(), ["score", "balance"])]
        )
        ct.fit(df)

        onx = to_onnx(ct, (df,))

        graph_inputs = {inp.name: inp for inp in onx.proto.graph.input}
        for col in ["age", "income", "score", "balance"]:
            self.assertIn(col, graph_inputs)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {col: df[col].to_numpy() for col in df.columns})[0]
        expected = ct.transform(df).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {col: df[col].to_numpy() for col in df.columns})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_pipeline_with_column_transformer_dataframe(self):
        """Pipeline with ColumnTransformer (string column names) and DataFrame input."""
        import pandas as pd

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
        df = pd.DataFrame(X, columns=["age", "income", "score", "balance"])
        ct = ColumnTransformer(
            [("std", StandardScaler(), ["age", "income"]), ("mm", MinMaxScaler(), ["score", "balance"])]
        )
        pipe = Pipeline([("preprocessor", ct), ("clf", LogisticRegression())])
        pipe.fit(df, y)

        onx = to_onnx(pipe, (df,))

        graph_inputs = {inp.name: inp for inp in onx.proto.graph.input}
        for col in ["age", "income", "score", "balance"]:
            self.assertIn(col, graph_inputs)

        ref = ExtendedReferenceEvaluator(onx)
        label, _ = ref.run(None, {col: df[col].to_numpy() for col in df.columns})
        expected_label = pipe.predict(df)
        self.assertEqualArray(expected_label, label)

        sess = self.check_ort(onx)
        ort_label, _ = sess.run(None, {col: df[col].to_numpy() for col in df.columns})
        self.assertEqualArray(expected_label, ort_label)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
