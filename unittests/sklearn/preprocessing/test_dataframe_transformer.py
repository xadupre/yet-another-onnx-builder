"""
Unit tests for :class:`~yobx.sklearn.preprocessing.DataFrameTransformer`.

:class:`~yobx.sklearn.preprocessing.DataFrameTransformer` wraps a
:class:`~yobx.sql.TracedDataFrame`-API function as a scikit-learn transformer
and provides automatic ONNX export without requiring ``extra_converters``.
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
    """Tests for the DataFrameTransformer class."""

    # ------------------------------------------------------------------
    # Fit / basic attributes
    # ------------------------------------------------------------------

    def test_fit_sets_input_dtypes(self):
        """fit() must populate input_dtypes_."""
        from yobx.sklearn.preprocessing import DataFrameTransformer

        def _func(df):
            return df.select([(df["a"] + df["b"]).alias("total")])

        t = DataFrameTransformer(func=_func, input_dtypes={"a": np.float32, "b": np.float32})
        t.fit()
        self.assertIsInstance(t.input_dtypes_, dict)
        self.assertIn("a", t.input_dtypes_)
        self.assertIn("b", t.input_dtypes_)
        self.assertEqual(t.input_dtypes_["a"], np.dtype("float32"))

    def test_onnx_args_shape(self):
        """onnx_args() should return one (name, dtype, shape) triple per column."""
        from yobx.sklearn.preprocessing import DataFrameTransformer

        def _func(df):
            return df.select([df["x"].alias("x_out")])

        t = DataFrameTransformer(func=_func, input_dtypes={"x": np.float32})
        t.fit()
        args = t.onnx_args()
        self.assertEqual(len(args), 1)
        name, dtype, shape = args[0]
        self.assertEqual(name, "x")
        self.assertEqual(dtype, np.dtype("float32"))
        self.assertEqual(shape, ("N",))

    # ------------------------------------------------------------------
    # Single-output ONNX export
    # ------------------------------------------------------------------

    def test_single_output_export(self):
        """Single-output transformer: no extra_converters needed."""
        from yobx.sklearn import to_onnx
        from yobx.sklearn.preprocessing import DataFrameTransformer

        def _add(df):
            return df.select([(df["a"] + df["b"]).alias("total")])

        t = DataFrameTransformer(func=_add, input_dtypes={"a": np.float32, "b": np.float32})
        t.fit()
        onx = to_onnx(t, t.onnx_args())

        input_names = {inp.name for inp in onx.proto.graph.input}
        self.assertIn("a", input_names)
        self.assertIn("b", input_names)
        self.assertIn("total", onx.output_names)

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run_onnx(onx, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b, rtol=1e-5)

    def test_scalar_multiply(self):
        """Scalar multiplication exported without extra_converters."""
        from yobx.sklearn import to_onnx
        from yobx.sklearn.preprocessing import DataFrameTransformer

        def _scale(df):
            return df.select([(df["x"] * 3.0).alias("x3")])

        t = DataFrameTransformer(func=_scale, input_dtypes={"x": np.float32})
        t.fit()
        onx = to_onnx(t, t.onnx_args())

        x = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        (x3,) = _run_onnx(onx, {"x": x})
        np.testing.assert_allclose(x3, x * 3.0, rtol=1e-5)

    # ------------------------------------------------------------------
    # Multi-output ONNX export
    # ------------------------------------------------------------------

    def test_multi_output_export(self):
        """Multi-output transformer: output names from traced query."""
        from yobx.sklearn import to_onnx
        from yobx.sklearn.preprocessing import DataFrameTransformer

        def _multi(df):
            return df.select(
                [
                    (df["a"] + df["b"]).alias("total"),
                    (df["a"] * 2.0).alias("a_doubled"),
                ]
            )

        t = DataFrameTransformer(func=_multi, input_dtypes={"a": np.float32, "b": np.float32})
        t.fit()
        onx = to_onnx(t, t.onnx_args())
        self.assertEqual(onx.output_names, ["total", "a_doubled"])

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        total, a_doubled = _run_onnx(onx, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b, rtol=1e-5)
        np.testing.assert_allclose(a_doubled, a * 2.0, rtol=1e-5)

    # ------------------------------------------------------------------
    # Row filtering
    # ------------------------------------------------------------------

    def test_filter_and_select(self):
        """Row filtering + column selection exported correctly."""
        from yobx.sklearn import to_onnx
        from yobx.sklearn.preprocessing import DataFrameTransformer

        def _filter(df):
            df = df.filter(df["a"] > 0)
            return df.select([(df["a"] + df["b"]).alias("total")])

        t = DataFrameTransformer(func=_filter, input_dtypes={"a": np.float32, "b": np.float32})
        t.fit()
        onx = to_onnx(t, t.onnx_args())

        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run_onnx(onx, {"a": a, "b": b})
        np.testing.assert_allclose(total, np.array([5.0, 9.0], dtype=np.float32), rtol=1e-5)

    # ------------------------------------------------------------------
    # transform() method
    # ------------------------------------------------------------------

    def test_transform_dict_input(self):
        """transform() with a dict of arrays produces correct results."""
        from yobx.sklearn.preprocessing import DataFrameTransformer

        def _add(df):
            return df.select([(df["a"] + df["b"]).alias("total")])

        t = DataFrameTransformer(func=_add, input_dtypes={"a": np.float32, "b": np.float32})
        t.fit()

        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        result = t.transform({"a": a, "b": b})
        np.testing.assert_allclose(result, np.array([[4.0], [6.0]], dtype=np.float32), rtol=1e-5)

    def test_transform_pandas_input(self):
        """transform() with a pandas DataFrame produces correct results."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        from yobx.sklearn.preprocessing import DataFrameTransformer

        def _add(df):
            return df.select([(df["a"] + df["b"]).alias("total")])

        t = DataFrameTransformer(func=_add, input_dtypes={"a": np.float32, "b": np.float32})
        t.fit()

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = t.transform(df)
        np.testing.assert_allclose(result, np.array([[4.0], [6.0]], dtype=np.float32), rtol=1e-5)

    # ------------------------------------------------------------------
    # Import path
    # ------------------------------------------------------------------

    def test_importable_from_yobx_sklearn(self):
        """DataFrameTransformer can be imported from yobx.sklearn."""
        from yobx.sklearn import DataFrameTransformer as DFT

        self.assertEqual(DFT.__name__, "DataFrameTransformer")

    def test_importable_from_yobx_sklearn_preprocessing(self):
        """DataFrameTransformer can be imported from yobx.sklearn.preprocessing."""
        from yobx.sklearn.preprocessing import DataFrameTransformer as DFT

        self.assertEqual(DFT.__name__, "DataFrameTransformer")

    # ------------------------------------------------------------------
    # Register_sklearn_converters consistency
    # ------------------------------------------------------------------

    def test_converter_registered_after_register_call(self):
        """The built-in converter is present after register_sklearn_converters()."""
        from yobx.sklearn import register_sklearn_converters
        from yobx.sklearn.preprocessing import DataFrameTransformer
        from yobx.sklearn.register import get_sklearn_converter

        register_sklearn_converters()
        conv = get_sklearn_converter(DataFrameTransformer)
        self.assertIsNotNone(conv)


if __name__ == "__main__":
    unittest.main()
