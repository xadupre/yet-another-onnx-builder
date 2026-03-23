"""
Unit tests for :class:`~yobx.sklearn.preprocessing.DataFrameTransformer` and
its ONNX converter.
"""

import unittest

import numpy as np

from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


def _run_onnx(onx, feeds):
    """Run an ONNX model with both the reference evaluator and onnxruntime."""
    from yobx.container import ExportArtifact

    proto = onx.proto if isinstance(onx, ExportArtifact) else onx
    ref = ExtendedReferenceEvaluator(onx)
    ref_outputs = ref.run(None, feeds)
    try:
        from onnxruntime import InferenceSession

        sess = InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
        ort_outputs = sess.run(None, feeds)
        assert len(ref_outputs) == len(ort_outputs)
        for ro, oo in zip(ref_outputs, ort_outputs):
            np.testing.assert_allclose(oo, ro, rtol=1e-5, atol=1e-6)
    except ImportError:
        pass
    return ref_outputs


@requires_sklearn("1.4")
class TestDataFrameTransformer(ExtTestCase):
    """Tests for DataFrameTransformer and its ONNX converter."""

    # ------------------------------------------------------------------
    # fit / transform
    # ------------------------------------------------------------------

    def test_fit_sets_output_names(self):
        """fit() must discover output column names via tracing."""
        from yobx.sklearn import DataFrameTransformer

        def func(df):
            return df.select([(df["a"] + df["b"]).alias("total")])

        t = DataFrameTransformer(func, {"a": np.float32, "b": np.float32})
        t.fit()
        self.assertEqual(list(t.output_names_), ["total"])

    def test_get_feature_names_out(self):
        """get_feature_names_out() must return the output column names."""
        from yobx.sklearn import DataFrameTransformer

        def func(df):
            return df.select(
                [(df["x"] * 2.0).alias("x2"), (df["y"] - 1.0).alias("y_minus_1")]
            )

        t = DataFrameTransformer(func, {"x": np.float32, "y": np.float32})
        t.fit()
        names = t.get_feature_names_out()
        self.assertEqual(list(names), ["x2", "y_minus_1"])

    def test_transform_pandas(self):
        """transform() must apply func to a pandas DataFrame via the ONNX model."""
        pd = __import__("pandas")
        from yobx.sklearn import DataFrameTransformer

        def func(df):
            return df.select([(df["a"] + df["b"]).alias("total")])

        t = DataFrameTransformer(func, {"a": np.float32, "b": np.float32})
        t.fit()
        df = pd.DataFrame(
            {"a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
             "b": np.array([4.0, 5.0, 6.0], dtype=np.float32)}
        )
        result = t.transform(df)
        np.testing.assert_allclose(result["total"].values, [5.0, 7.0, 9.0])

    def test_transform_raises_on_non_dataframe(self):
        """transform() must raise TypeError when X is not a DataFrame."""
        from yobx.sklearn import DataFrameTransformer

        def func(df):
            return df.select([(df["a"] + 1.0).alias("a1")])

        t = DataFrameTransformer(func, {"a": np.float32})
        t.fit()
        with self.assertRaises(TypeError):
            t.transform(np.array([1.0, 2.0]))

    def test_fit_caches_onnx_model(self):
        """fit() must cache an ONNX model (onnx_model_) for use in transform()."""
        from yobx.sklearn import DataFrameTransformer
        import onnx

        def func(df):
            return df.select([(df["a"] + 1.0).alias("a1")])

        t = DataFrameTransformer(func, {"a": np.float32})
        t.fit()
        self.assertIsInstance(t.onnx_model_, onnx.ModelProto)
        self.assertEqual(t.n_onnx_outputs_, 1)

    # ------------------------------------------------------------------
    # onnx_args helper
    # ------------------------------------------------------------------

    def test_onnx_args(self):
        """onnx_args() must return one (name, dtype, shape) triple per column."""
        from yobx.sklearn import DataFrameTransformer

        def func(df):
            return df.select([(df["a"] + 1.0).alias("a1")])

        t = DataFrameTransformer(func, {"a": np.float32, "b": np.float64})
        t.fit()
        args = t.onnx_args()
        self.assertEqual(len(args), 2)
        self.assertEqual(args[0][0], "a")
        self.assertEqual(args[0][1], np.dtype(np.float32))
        self.assertEqual(args[0][2], ("N",))
        self.assertEqual(args[1][0], "b")

    # ------------------------------------------------------------------
    # ONNX conversion
    # ------------------------------------------------------------------

    def test_single_output_add(self):
        """Simple column addition exported to ONNX."""
        from yobx.sklearn import DataFrameTransformer, to_onnx

        def func(df):
            return df.select([(df["a"] + df["b"]).alias("total")])

        t = DataFrameTransformer(func, {"a": np.float32, "b": np.float32})
        t.fit()

        onx = to_onnx(t, t.onnx_args())
        input_names = [inp.name for inp in onx.proto.graph.input]
        self.assertIn("a", input_names)
        self.assertIn("b", input_names)
        output_names = [out.name for out in onx.proto.graph.output]
        self.assertIn("total", output_names)

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run_onnx(onx, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b, rtol=1e-5)

    def test_multi_output(self):
        """Multiple output columns exported to ONNX."""
        from yobx.sklearn import DataFrameTransformer, to_onnx

        def func(df):
            return df.select(
                [
                    (df["a"] + df["b"]).alias("total"),
                    (df["a"] * 2.0).alias("a_doubled"),
                ]
            )

        t = DataFrameTransformer(func, {"a": np.float32, "b": np.float32})
        t.fit()
        self.assertEqual(list(t.get_feature_names_out()), ["total", "a_doubled"])

        onx = to_onnx(t, t.onnx_args())
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        total, a_doubled = _run_onnx(onx, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b, rtol=1e-5)
        np.testing.assert_allclose(a_doubled, a * 2.0, rtol=1e-5)

    def test_filter_and_select(self):
        """Filter + select combination exported to ONNX."""
        from yobx.sklearn import DataFrameTransformer, to_onnx

        def func(df):
            df = df.filter(df["a"] > 0)
            return df.select([(df["a"] + df["b"]).alias("total")])

        t = DataFrameTransformer(func, {"a": np.float32, "b": np.float32})
        t.fit()

        onx = to_onnx(t, t.onnx_args())
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run_onnx(onx, {"a": a, "b": b})
        np.testing.assert_allclose(total, np.array([5.0, 9.0], dtype=np.float32), rtol=1e-5)

    def test_scalar_multiply(self):
        """Scalar multiplication exported to ONNX."""
        from yobx.sklearn import DataFrameTransformer, to_onnx

        def func(df):
            return df.select([(df["x"] * 3.0).alias("x3")])

        t = DataFrameTransformer(func, {"x": np.float32})
        t.fit()

        onx = to_onnx(t, t.onnx_args())
        x = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        (x3,) = _run_onnx(onx, {"x": x})
        np.testing.assert_allclose(x3, x * 3.0, rtol=1e-5)

    def test_three_columns(self):
        """Three input columns — verify all are registered as ONNX inputs."""
        from yobx.sklearn import DataFrameTransformer, to_onnx

        def func(df):
            return df.select([(df["a"] + df["b"] + df["c"]).alias("s")])

        t = DataFrameTransformer(func, {"a": np.float32, "b": np.float32, "c": np.float32})
        t.fit()

        onx = to_onnx(t, t.onnx_args())
        input_names = {inp.name for inp in onx.proto.graph.input}
        self.assertIn("a", input_names)
        self.assertIn("b", input_names)
        self.assertIn("c", input_names)

        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        c = np.array([5.0, 6.0], dtype=np.float32)
        (s,) = _run_onnx(onx, {"a": a, "b": b, "c": c})
        np.testing.assert_allclose(s, a + b + c, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
