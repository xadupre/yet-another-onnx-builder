"""
Unit tests demonstrating how a user-defined scikit-learn transformer that
processes :class:`pandas.DataFrame` objects can be converted to ONNX.

The pattern shown here uses:

* A plain ``BaseEstimator`` / ``TransformerMixin`` subclass (no wrapper class
  required — the type is resolved by ``extra_converters`` at export time).
* A tracing function written with the :class:`~yobx.sql.TracedDataFrame` API.
* :func:`~yobx.sql.sql_convert.parsed_query_to_onnx_graph` with
  ``_finalize=False`` to embed the dataframe query in the caller-managed graph.
* :func:`~yobx.sklearn.to_onnx` with ``extra_converters`` to wire the
  user-defined transformer to the converter.
"""

import unittest

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from yobx.ext_test_case import ExtTestCase, requires_sklearn
from yobx.reference import ExtendedReferenceEvaluator


# ---------------------------------------------------------------------------
# Module-level helpers shared by multiple tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# A sample user-defined transformer: computes total = a + b
# ---------------------------------------------------------------------------


class _AddColumnsTransformer(BaseEstimator, TransformerMixin):
    """User-defined transformer: adds two float32 columns a and b -> total."""

    _INPUT_DTYPES = {"a": np.float32, "b": np.float32}

    def fit(self, X=None, y=None):
        self.input_dtypes_ = {k: np.dtype(v) for k, v in self._INPUT_DTYPES.items()}
        # Declare the number of ONNX output tensors so that
        # get_n_expected_outputs() returns the correct value.
        self.n_onnx_outputs_ = 1
        return self

    def transform(self, df):
        import pandas as pd

        return pd.DataFrame({"total": df["a"].values + df["b"].values})

    def get_feature_names_out(self, input_features=None):
        return np.array(["total"])


def _add_columns_tracing_func(df):
    """TracedDataFrame-compatible counterpart of _AddColumnsTransformer.transform."""
    return df.select([(df["a"] + df["b"]).alias("total")])


def _add_columns_converter(g, sts, outputs, estimator, *inputs, name="add_cols"):
    """ONNX converter for _AddColumnsTransformer.

    Uses dataframe tracing to emit the ONNX nodes and embeds them in the
    caller's graph via :func:`~yobx.sql.sql_convert.parsed_query_to_onnx_graph`
    with ``_finalize=False`` to let the caller register the outputs.
    """
    from yobx.sql import trace_dataframe
    from yobx.sql.sql_convert import parsed_query_to_onnx_graph

    pq = trace_dataframe(_add_columns_tracing_func, estimator.input_dtypes_)
    out_names = parsed_query_to_onnx_graph(
        g, sts, list(outputs), pq, estimator.input_dtypes_, _finalize=False
    )
    return out_names[0]


# ---------------------------------------------------------------------------
# A multi-output user-defined transformer
# ---------------------------------------------------------------------------


class _MultiOutTransformer(BaseEstimator, TransformerMixin):
    """User-defined transformer: produces two output columns from two inputs."""

    _INPUT_DTYPES = {"a": np.float32, "b": np.float32}

    def fit(self, X=None, y=None):
        self.input_dtypes_ = {k: np.dtype(v) for k, v in self._INPUT_DTYPES.items()}
        self.n_onnx_outputs_ = 2
        return self

    def transform(self, df):
        import pandas as pd

        return pd.DataFrame(
            {
                "total": df["a"].values + df["b"].values,
                "a_doubled": df["a"].values * 2.0,
            }
        )

    def get_feature_names_out(self, input_features=None):
        return np.array(["total", "a_doubled"])


def _multi_out_tracing_func(df):
    return df.select(
        [
            (df["a"] + df["b"]).alias("total"),
            (df["a"] * 2.0).alias("a_doubled"),
        ]
    )


def _multi_out_converter(g, sts, outputs, estimator, *inputs, name="multi_out"):
    from yobx.sql import trace_dataframe
    from yobx.sql.sql_convert import parsed_query_to_onnx_graph

    pq = trace_dataframe(_multi_out_tracing_func, estimator.input_dtypes_)
    out_names = parsed_query_to_onnx_graph(
        g, sts, list(outputs), pq, estimator.input_dtypes_, _finalize=False
    )
    return out_names[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_sklearn("1.4")
class TestCustomDataFrameTransformerOnnx(ExtTestCase):
    """Converting user-defined DataFrame-processing transformers to ONNX."""

    def _make_args(self, input_dtypes):
        """Build (name, dtype, shape) arg tuples from an input_dtypes dict."""
        return tuple((col, dtype, ("N",)) for col, dtype in input_dtypes.items())

    # ------------------------------------------------------------------
    # fit / get_feature_names_out
    # ------------------------------------------------------------------

    def test_fit_sets_attributes(self):
        """fit() must set input_dtypes_ and n_onnx_outputs_."""
        t = _AddColumnsTransformer()
        t.fit()
        self.assertIsInstance(t.input_dtypes_, dict)
        self.assertEqual(t.n_onnx_outputs_, 1)

    def test_get_feature_names_out(self):
        """get_feature_names_out() must return ['total'] for _AddColumnsTransformer."""
        t = _AddColumnsTransformer()
        t.fit()
        self.assertEqual(list(t.get_feature_names_out()), ["total"])

    # ------------------------------------------------------------------
    # Single-output ONNX export
    # ------------------------------------------------------------------

    def test_single_output_add(self):
        """Simple column addition: user-defined transformer exported to ONNX."""
        from yobx.sklearn import to_onnx

        t = _AddColumnsTransformer()
        t.fit()
        args = self._make_args(t.input_dtypes_)
        onx = to_onnx(t, args, extra_converters={_AddColumnsTransformer: _add_columns_converter})

        input_names = {inp.name for inp in onx.proto.graph.input}
        self.assertIn("a", input_names)
        self.assertIn("b", input_names)
        output_names = [out.name for out in onx.proto.graph.output]
        self.assertIn("total", output_names)

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run_onnx(onx, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b, rtol=1e-5)

    def test_scalar_multiply(self):
        """Scalar multiplication: user-defined single-column transformer."""
        from yobx.sklearn import to_onnx

        class _ScaleTransformer(BaseEstimator, TransformerMixin):
            _INPUT_DTYPES = {"x": np.float32}

            def fit(self, X=None, y=None):
                self.input_dtypes_ = {k: np.dtype(v) for k, v in self._INPUT_DTYPES.items()}
                self.n_onnx_outputs_ = 1
                return self

            def transform(self, df):
                import pandas as pd

                return pd.DataFrame({"x3": df["x"].values * 3.0})

            def get_feature_names_out(self, input_features=None):
                return np.array(["x3"])

        def _scale_func(df):
            return df.select([(df["x"] * 3.0).alias("x3")])

        def _scale_converter(g, sts, outputs, estimator, *inputs, name="scale"):
            from yobx.sql import trace_dataframe
            from yobx.sql.sql_convert import parsed_query_to_onnx_graph

            pq = trace_dataframe(_scale_func, estimator.input_dtypes_)
            out_names = parsed_query_to_onnx_graph(
                g, sts, list(outputs), pq, estimator.input_dtypes_, _finalize=False
            )
            return out_names[0]

        t = _ScaleTransformer()
        t.fit()
        args = self._make_args(t.input_dtypes_)
        onx = to_onnx(t, args, extra_converters={_ScaleTransformer: _scale_converter})

        x = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        (x3,) = _run_onnx(onx, {"x": x})
        np.testing.assert_allclose(x3, x * 3.0, rtol=1e-5)

    # ------------------------------------------------------------------
    # Multi-output ONNX export
    # ------------------------------------------------------------------

    def test_multi_output(self):
        """Multiple output columns exported to ONNX via extra_converters."""
        from yobx.sklearn import to_onnx

        t = _MultiOutTransformer()
        t.fit()
        self.assertEqual(list(t.get_feature_names_out()), ["total", "a_doubled"])

        args = self._make_args(t.input_dtypes_)
        onx = to_onnx(t, args, extra_converters={_MultiOutTransformer: _multi_out_converter})

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        total, a_doubled = _run_onnx(onx, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b, rtol=1e-5)
        np.testing.assert_allclose(a_doubled, a * 2.0, rtol=1e-5)

    # ------------------------------------------------------------------
    # Filter + select
    # ------------------------------------------------------------------

    def test_filter_and_select(self):
        """Row filtering combined with column selection exported to ONNX."""
        from yobx.sklearn import to_onnx

        class _FilterTransformer(BaseEstimator, TransformerMixin):
            _INPUT_DTYPES = {"a": np.float32, "b": np.float32}

            def fit(self, X=None, y=None):
                self.input_dtypes_ = {k: np.dtype(v) for k, v in self._INPUT_DTYPES.items()}
                self.n_onnx_outputs_ = 1
                return self

            def transform(self, df):
                import pandas as pd

                mask = df["a"].values > 0
                return pd.DataFrame({"total": df["a"].values[mask] + df["b"].values[mask]})

            def get_feature_names_out(self, input_features=None):
                return np.array(["total"])

        def _filter_func(df):
            df = df.filter(df["a"] > 0)
            return df.select([(df["a"] + df["b"]).alias("total")])

        def _filter_converter(g, sts, outputs, estimator, *inputs, name="filter"):
            from yobx.sql import trace_dataframe
            from yobx.sql.sql_convert import parsed_query_to_onnx_graph

            pq = trace_dataframe(_filter_func, estimator.input_dtypes_)
            out_names = parsed_query_to_onnx_graph(
                g, sts, list(outputs), pq, estimator.input_dtypes_, _finalize=False
            )
            return out_names[0]

        t = _FilterTransformer()
        t.fit()
        args = self._make_args(t.input_dtypes_)
        onx = to_onnx(t, args, extra_converters={_FilterTransformer: _filter_converter})

        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run_onnx(onx, {"a": a, "b": b})
        np.testing.assert_allclose(total, np.array([5.0, 9.0], dtype=np.float32), rtol=1e-5)

    # ------------------------------------------------------------------
    # Three columns
    # ------------------------------------------------------------------

    def test_three_columns(self):
        """Three input columns: all are registered as ONNX inputs."""
        from yobx.sklearn import to_onnx

        class _ThreeColTransformer(BaseEstimator, TransformerMixin):
            _INPUT_DTYPES = {"a": np.float32, "b": np.float32, "c": np.float32}

            def fit(self, X=None, y=None):
                self.input_dtypes_ = {k: np.dtype(v) for k, v in self._INPUT_DTYPES.items()}
                self.n_onnx_outputs_ = 1
                return self

            def transform(self, df):
                import pandas as pd

                return pd.DataFrame(
                    {"s": df["a"].values + df["b"].values + df["c"].values}
                )

            def get_feature_names_out(self, input_features=None):
                return np.array(["s"])

        def _three_col_func(df):
            return df.select([(df["a"] + df["b"] + df["c"]).alias("s")])

        def _three_col_converter(g, sts, outputs, estimator, *inputs, name="three"):
            from yobx.sql import trace_dataframe
            from yobx.sql.sql_convert import parsed_query_to_onnx_graph

            pq = trace_dataframe(_three_col_func, estimator.input_dtypes_)
            out_names = parsed_query_to_onnx_graph(
                g, sts, list(outputs), pq, estimator.input_dtypes_, _finalize=False
            )
            return out_names[0]

        t = _ThreeColTransformer()
        t.fit()
        args = self._make_args(t.input_dtypes_)
        onx = to_onnx(
            t, args, extra_converters={_ThreeColTransformer: _three_col_converter}
        )

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

