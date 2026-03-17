"""
Unit tests for yobx.sklearn.category_encoders.BinaryEncoder converter.
"""

import unittest
import numpy as np
import pandas as pd
from yobx.ext_test_case import ExtTestCase, requires_sklearn, requires_category_encoders
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
@requires_category_encoders("2.6")
class TestCEBinaryEncoder(ExtTestCase):
    def _make_data(self, dtype=np.float32):
        X = pd.DataFrame(
            {
                "cat1": np.tile([0.0, 1.0, 2.0, 3.0, 0.0, 1.0], 5).astype(float),
                "cat2": np.tile([0.0, 1.0, 0.0, 2.0, 1.0, 0.0], 5).astype(float),
                "num": np.arange(30, dtype=float),
            }
        )
        y = np.tile([1, 0, 1, 0, 1, 0], 5)
        return X, y, dtype

    def _run_test(self, enc, X_df, dtype, atol=1e-6):
        from yobx.sklearn import to_onnx

        X_np = X_df.values.astype(dtype)
        X_df_typed = pd.DataFrame(X_np, columns=X_df.columns)

        onx = to_onnx(enc, (X_np,))

        expected = enc.transform(X_df_typed).values.astype(dtype)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_np})[0]
        self.assertEqualArray(expected, result, atol=atol)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_np})[0]
        self.assertEqualArray(expected, ort_result, atol=atol)

    def test_binary_encoder_float32(self):
        """Two categorical columns plus one numeric column, float32."""
        from category_encoders import BinaryEncoder

        X_df, y, dtype = self._make_data(np.float32)
        enc = BinaryEncoder(cols=["cat1", "cat2"])
        enc.fit(X_df, y)
        self._run_test(enc, X_df[:8], dtype)

    def test_binary_encoder_float64(self):
        """Two categorical columns plus one numeric column, float64."""
        from category_encoders import BinaryEncoder

        X_df, y, dtype = self._make_data(np.float64)
        enc = BinaryEncoder(cols=["cat1", "cat2"])
        enc.fit(X_df, y)
        self._run_test(enc, X_df[:8], dtype)

    def test_binary_encoder_single_col_float32(self):
        """Single categorical column, float32."""
        from category_encoders import BinaryEncoder

        X_df, y, dtype = self._make_data(np.float32)
        enc = BinaryEncoder(cols=["cat1"])
        enc.fit(X_df, y)
        self._run_test(enc, X_df[:8], dtype)

    def test_binary_encoder_single_col_float64(self):
        """Single categorical column, float64."""
        from category_encoders import BinaryEncoder

        X_df, y, dtype = self._make_data(np.float64)
        enc = BinaryEncoder(cols=["cat1"])
        enc.fit(X_df, y)
        self._run_test(enc, X_df[:8], dtype)

    def test_binary_encoder_all_numeric_passthrough(self):
        """When cols=[], all columns pass through unchanged."""
        from category_encoders import BinaryEncoder
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X_df = pd.DataFrame({"a": rng.standard_normal(20), "b": rng.standard_normal(20)})
        y = (X_df["a"] > 0).astype(int).values
        enc = BinaryEncoder(cols=[])
        enc.fit(X_df, y)

        X_np = X_df.values.astype(np.float32)
        X_df_typed = pd.DataFrame(X_np, columns=X_df.columns)
        onx = to_onnx(enc, (X_np,))
        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_np})[0]

        expected = enc.transform(X_df_typed).values.astype(np.float32)
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_binary_encoder_unknown_category_value(self):
        """Unknown categories produce all-zero binary block (handle_unknown='value')."""
        from category_encoders import BinaryEncoder
        from yobx.sklearn import to_onnx

        X_df = pd.DataFrame(
            {
                "cat": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0],
                "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        y = [0, 1, 0, 1, 0, 1]
        enc = BinaryEncoder(cols=["cat"], handle_unknown="value")
        enc.fit(X_df, y)

        X_test = pd.DataFrame({"cat": [0.0, 1.0, 99.0], "num": [1.0, 2.0, 3.0]})
        X_np = X_test.values.astype(np.float32)

        onx = to_onnx(enc, (X_np,))
        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_np})[0]

        expected = enc.transform(X_test).values.astype(np.float32)
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_binary_encoder_unknown_category_return_nan(self):
        """Unknown categories produce NaN when handle_unknown='return_nan'."""
        from category_encoders import BinaryEncoder
        from yobx.sklearn import to_onnx

        X_df = pd.DataFrame(
            {
                "cat": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0],
                "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        y = [0, 1, 0, 1, 0, 1]
        enc = BinaryEncoder(cols=["cat"], handle_unknown="return_nan")
        enc.fit(X_df, y)

        X_test = pd.DataFrame({"cat": [0.0, 99.0, 1.0], "num": [1.0, 2.0, 3.0]})
        X_np = X_test.values.astype(np.float32)

        onx = to_onnx(enc, (X_np,))
        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_np})[0]

        expected = enc.transform(X_test).values.astype(np.float32)
        nan_mask = np.isnan(expected)
        self.assertEqualArray(nan_mask, np.isnan(ort_result))
        self.assertEqualArray(expected[~nan_mask], ort_result[~nan_mask], atol=1e-6)

    def test_binary_encoder_missing_value(self):
        """NaN input produces all-zero binary block (handle_missing='value')."""
        from category_encoders import BinaryEncoder
        from yobx.sklearn import to_onnx

        X_df = pd.DataFrame(
            {
                "cat": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0],
                "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        y = [0, 1, 0, 1, 0, 1]
        enc = BinaryEncoder(cols=["cat"], handle_missing="value")
        enc.fit(X_df, y)

        X_test = pd.DataFrame({"cat": [0.0, np.nan, 2.0], "num": [1.0, 2.0, 3.0]})
        X_np = X_test.values.astype(np.float32)

        onx = to_onnx(enc, (X_np,))
        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_np})[0]

        expected = enc.transform(X_test).values.astype(np.float32)
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_binary_encoder_missing_return_nan(self):
        """NaN input produces NaN binary block when handle_missing='return_nan'."""
        from category_encoders import BinaryEncoder
        from yobx.sklearn import to_onnx

        X_df = pd.DataFrame(
            {
                "cat": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0],
                "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        y = [0, 1, 0, 1, 0, 1]
        enc = BinaryEncoder(cols=["cat"], handle_missing="return_nan")
        enc.fit(X_df, y)

        X_test = pd.DataFrame({"cat": [0.0, np.nan, 2.0], "num": [1.0, 2.0, 3.0]})
        X_np = X_test.values.astype(np.float32)

        onx = to_onnx(enc, (X_np,))
        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_np})[0]

        expected = enc.transform(X_test).values.astype(np.float32)
        nan_mask = np.isnan(expected)
        self.assertEqualArray(nan_mask, np.isnan(ort_result))
        self.assertEqualArray(expected[~nan_mask], ort_result[~nan_mask], atol=1e-6)

    def test_binary_encoder_pipeline(self):
        """BinaryEncoder in a sklearn Pipeline."""
        from category_encoders import BinaryEncoder
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X_df, y, _dtype = self._make_data(np.float32)
        y_f = y.astype(np.float32)
        pipe = Pipeline(
            [("enc", BinaryEncoder(cols=["cat1", "cat2"])), ("reg", LinearRegression())]
        )
        pipe.fit(X_df, y_f)

        X_np = X_df[:8].values.astype(np.float32)
        onx = to_onnx(pipe, (X_np,))
        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_np})[0]

        expected = pipe.predict(X_df[:8]).astype(np.float32).reshape(-1, 1)
        self.assertEqualArray(expected, ort_result, atol=1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
