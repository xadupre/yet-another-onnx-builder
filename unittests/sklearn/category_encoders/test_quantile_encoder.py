"""
Unit tests for yobx.sklearn.category_encoders.QuantileEncoder converter.
"""

import unittest
import numpy as np
import pandas as pd
from yobx.ext_test_case import ExtTestCase, requires_sklearn, requires_category_encoders
from yobx.reference import ExtendedReferenceEvaluator


@requires_sklearn("1.4")
@requires_category_encoders("2.6")
class TestQuantileEncoder(ExtTestCase):
    def _make_data(self, dtype=np.float32):
        _rng = np.random.default_rng(0)
        X = pd.DataFrame(
            {
                "cat1": np.tile([0, 1, 2, 0, 1, 0], 5).astype(float),
                "cat2": np.tile([0, 1, 0, 2, 1, 0], 5).astype(float),
                "num": np.arange(30, dtype=float),
            }
        )
        y = np.tile([1, 0, 1, 0, 1, 0], 5)
        return X, y, dtype

    def _run_test(self, enc, X_df, dtype):
        from yobx.sklearn import to_onnx

        X_np = X_df.values.astype(dtype)
        X_df_typed = pd.DataFrame(X_np, columns=X_df.columns)

        onx = to_onnx(enc, (X_np,))

        expected = enc.transform(X_df_typed).values.astype(dtype)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X_np})[0]
        self.assertEqualArray(expected, result, atol=1e-6)

        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_np})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_quantile_encoder_float32(self):
        """Basic two-column encoding with float32 inputs."""
        from category_encoders import QuantileEncoder

        X_df, y, dtype = self._make_data(np.float32)
        enc = QuantileEncoder(cols=["cat1", "cat2"])
        enc.fit(X_df, y)
        self._run_test(enc, X_df[:8], dtype)

    def test_quantile_encoder_float64(self):
        """Basic two-column encoding with float64 inputs."""
        from category_encoders import QuantileEncoder

        X_df, y, dtype = self._make_data(np.float64)
        enc = QuantileEncoder(cols=["cat1", "cat2"])
        enc.fit(X_df, y)
        self._run_test(enc, X_df[:8], dtype)

    def test_quantile_encoder_single_col_float32(self):
        """Single categorical column with float32."""
        from category_encoders import QuantileEncoder

        X_df, y, dtype = self._make_data(np.float32)
        enc = QuantileEncoder(cols=["cat1"])
        enc.fit(X_df, y)
        self._run_test(enc, X_df[:8], dtype)

    def test_quantile_encoder_single_col_float64(self):
        """Single categorical column with float64."""
        from category_encoders import QuantileEncoder

        X_df, y, dtype = self._make_data(np.float64)
        enc = QuantileEncoder(cols=["cat1"])
        enc.fit(X_df, y)
        self._run_test(enc, X_df[:8], dtype)

    def test_quantile_encoder_unknown_category(self):
        """Unknown categories map to the handle_unknown quantile value."""
        from category_encoders import QuantileEncoder
        from yobx.sklearn import to_onnx

        X_df, y, dtype = self._make_data(np.float32)
        enc = QuantileEncoder(cols=["cat1"])
        enc.fit(X_df, y)

        # Include an unknown category value (99)
        X_test_df = pd.DataFrame(
            {
                "cat1": [0.0, 1.0, 2.0, 99.0],
                "cat2": [0.0, 1.0, 2.0, 0.0],
                "num": [0.0, 1.0, 2.0, 3.0],
            }
        )
        X_np = X_test_df.values.astype(dtype)

        onx = to_onnx(enc, (X_np,))
        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_np})[0]

        expected = enc.transform(X_test_df).values.astype(dtype)
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_quantile_encoder_all_numeric_passthrough(self):
        """When cols=[], all columns pass through unchanged."""
        from category_encoders import QuantileEncoder
        from yobx.sklearn import to_onnx

        rng = np.random.default_rng(1)
        X_df = pd.DataFrame({"a": rng.standard_normal(20), "b": rng.standard_normal(20)})
        y = (X_df["a"] > 0).astype(int).values
        enc = QuantileEncoder(cols=[])
        enc.fit(X_df, y)

        X_np = X_df.values.astype(np.float32)
        onx = to_onnx(enc, (X_np,))
        sess = self.check_ort(onx)
        ort_result = sess.run(None, {"X": X_np})[0]

        expected = enc.transform(X_df).values.astype(np.float32)
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_quantile_encoder_pipeline(self):
        """QuantileEncoder in a sklearn Pipeline."""
        from category_encoders import QuantileEncoder
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from yobx.sklearn import to_onnx

        X_df, y, _dtype = self._make_data(np.float32)
        y_f = y.astype(np.float32)
        pipe = Pipeline(
            [("enc", QuantileEncoder(cols=["cat1", "cat2"])), ("reg", LinearRegression())]
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
