"""
Unit tests for yobx.sklearn.category_encoders.OrdinalEncoder converter.
"""

import unittest
import numpy as np
import pandas as pd
from yobx.ext_test_case import ExtTestCase, requires_sklearn, requires_category_encoders
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
@requires_category_encoders("2.6")
class TestCEOrdinalEncoder(ExtTestCase):
    def _run_test(self, enc, X_df, X_np, atol=1e-6):
        from yobx.sklearn import to_onnx

        onx = to_onnx(enc, (X_np,))

        expected = enc.transform(X_df).values.astype(X_np.dtype)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_np})[0]
        self.assertEqualArray(expected, result, atol=atol)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_np})[0]
        self.assertEqualArray(expected, ort_result, atol=atol)

    def test_basic_float32(self):
        """Two categorical columns, one numeric, float32."""
        from category_encoders import OrdinalEncoder

        X_df = pd.DataFrame(
            {
                "cat1": [0.0, 1.0, 2.0, 0.0, 1.0],
                "cat2": [0.0, 1.0, 0.0, 2.0, 1.0],
                "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        enc = OrdinalEncoder(cols=["cat1", "cat2"])
        enc.fit(X_df)
        X_np = X_df.values.astype(np.float32)
        self._run_test(enc, X_df, X_np)

    def test_basic_float64(self):
        """Two categorical columns, one numeric, float64."""
        from category_encoders import OrdinalEncoder

        X_df = pd.DataFrame(
            {
                "cat1": [0.0, 1.0, 2.0, 0.0, 1.0],
                "cat2": [0.0, 1.0, 0.0, 2.0, 1.0],
                "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        enc = OrdinalEncoder(cols=["cat1", "cat2"])
        enc.fit(X_df)
        X_np = X_df.values.astype(np.float64)
        self._run_test(enc, X_df, X_np)

    def test_single_categorical_col(self):
        """Single categorical column only."""
        from category_encoders import OrdinalEncoder

        X_df = pd.DataFrame({"cat": [0.0, 1.0, 2.0, 0.0, 2.0]})
        enc = OrdinalEncoder(cols=["cat"])
        enc.fit(X_df)
        X_np = X_df.values.astype(np.float32)
        self._run_test(enc, X_df, X_np)

    def test_no_categorical_cols(self):
        """All numeric - everything passes through unchanged."""
        from category_encoders import OrdinalEncoder

        X_df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        enc = OrdinalEncoder(cols=[])
        enc.fit(X_df)
        X_np = X_df.values.astype(np.float32)
        self._run_test(enc, X_df, X_np)

    def test_unknown_category_value(self):
        """Unknown categories produce -1 (handle_unknown='value')."""
        from category_encoders import OrdinalEncoder
        from yobx.sklearn import to_onnx

        X_df = pd.DataFrame({"cat": [0.0, 1.0, 2.0], "num": [1.0, 2.0, 3.0]})
        enc = OrdinalEncoder(cols=["cat"], handle_unknown="value")
        enc.fit(X_df)

        X_test_df = pd.DataFrame({"cat": [0.0, 1.0, 99.0], "num": [1.0, 2.0, 3.0]})
        X_np = X_test_df.values.astype(np.float32)

        onx = to_onnx(enc, (X_np,))
        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_np})[0]

        expected = enc.transform(X_test_df).values.astype(np.float32)
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_unknown_category_return_nan(self):
        """Unknown categories produce NaN when handle_unknown='return_nan'."""
        from category_encoders import OrdinalEncoder
        from yobx.sklearn import to_onnx

        X_df = pd.DataFrame({"cat": [0.0, 1.0, 2.0], "num": [1.0, 2.0, 3.0]})
        enc = OrdinalEncoder(cols=["cat"], handle_unknown="return_nan")
        enc.fit(X_df)

        X_test_df = pd.DataFrame({"cat": [0.0, 1.0, 99.0], "num": [1.0, 2.0, 3.0]})
        X_np = X_test_df.values.astype(np.float32)

        onx = to_onnx(enc, (X_np,))
        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_np})[0]

        expected = enc.transform(X_test_df).values.astype(np.float32)
        nan_mask = np.isnan(expected)
        self.assertEqualArray(nan_mask, np.isnan(ort_result))
        self.assertEqualArray(expected[~nan_mask], ort_result[~nan_mask], atol=1e-6)

    def test_missing_value_default(self):
        """NaN input produces -2 (handle_missing='value')."""
        from category_encoders import OrdinalEncoder
        from yobx.sklearn import to_onnx

        X_df = pd.DataFrame({"cat": [0.0, 1.0, 2.0], "num": [1.0, 2.0, 3.0]})
        enc = OrdinalEncoder(cols=["cat"], handle_missing="value")
        enc.fit(X_df)

        X_test_df = pd.DataFrame({"cat": [0.0, np.nan, 1.0], "num": [1.0, 2.0, 3.0]})
        X_np = X_test_df.values.astype(np.float32)

        onx = to_onnx(enc, (X_np,))
        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_np})[0]

        expected = enc.transform(X_test_df).values.astype(np.float32)
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_missing_value_return_nan(self):
        """NaN input produces NaN when handle_missing='return_nan'."""
        from category_encoders import OrdinalEncoder
        from yobx.sklearn import to_onnx

        X_df = pd.DataFrame({"cat": [0.0, 1.0, 2.0], "num": [1.0, 2.0, 3.0]})
        enc = OrdinalEncoder(cols=["cat"], handle_missing="return_nan")
        enc.fit(X_df)

        X_test_df = pd.DataFrame({"cat": [0.0, np.nan, 1.0], "num": [1.0, 2.0, 3.0]})
        X_np = X_test_df.values.astype(np.float32)

        onx = to_onnx(enc, (X_np,))
        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_np})[0]

        expected = enc.transform(X_test_df).values.astype(np.float32)
        nan_mask = np.isnan(expected)
        self.assertEqualArray(nan_mask, np.isnan(ort_result))
        self.assertEqualArray(expected[~nan_mask], ort_result[~nan_mask], atol=1e-6)

    def test_pipeline(self):
        """OrdinalEncoder (category_encoders) in a sklearn Pipeline."""
        from category_encoders import OrdinalEncoder
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X_df = pd.DataFrame(
            {
                "cat1": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
                "cat2": [0.0, 1.0, 0.0, 2.0, 1.0, 2.0],
                "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)

        pipe = Pipeline(
            [("enc", OrdinalEncoder(cols=["cat1", "cat2"])), ("reg", LinearRegression())]
        )
        pipe.fit(X_df, y)

        X_np = X_df.values.astype(np.float32)
        onx = to_onnx(pipe, (X_np,))
        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_np})[0]

        expected = pipe.predict(X_df).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, ort_result, atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
