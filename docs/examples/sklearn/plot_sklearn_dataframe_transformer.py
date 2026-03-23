"""
.. _l-plot-sklearn-dataframe-transformer:

Exporting a DataFrame-processing transformer to ONNX
=====================================================

Users can convert **any** custom :class:`sklearn.base.BaseEstimator` that
processes :class:`pandas.DataFrame` objects to ONNX by supplying a converter
function via the ``extra_converters`` parameter of :func:`yobx.sklearn.to_onnx`.

The converter is written using two primitives from the SQL/dataframe module:

* :func:`~yobx.sql.trace_dataframe` — traces a function that uses the
  :class:`~yobx.sql.TracedDataFrame` API and returns a
  :class:`~yobx.sql.parse.ParsedQuery` (the operation graph).
* :func:`~yobx.sql.sql_convert.parsed_query_to_onnx_graph` with
  ``_finalize=False`` — compiles the query into the caller-managed graph
  builder without registering its own model outputs (the caller does that).

No wrapper class is needed: the user implements a plain sklearn estimator, and
its ONNX type is resolved by ``extra_converters`` at export time.

This example covers:

1. **Single output column** — a simple column addition.
2. **Multiple output columns** — exporting several computed columns at once.
3. **Filter + select** — row filtering combined with a column selection.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from yobx.doc import plot_dot
from yobx.sklearn import to_onnx
from yobx.reference import ExtendedReferenceEvaluator

# %%
# Helper: build the ``extra_converters`` entry
# ---------------------------------------------
#
# The converter function follows the standard sklearn-converter signature:
# ``(g, sts, outputs, estimator, *input_tensor_names, name=...)``.
# It calls :func:`~yobx.sql.sql_convert.parsed_query_to_onnx_graph` with
# ``_finalize=False`` so that :func:`~yobx.sklearn.to_onnx` can register the
# outputs in the final model without duplication.


def make_dataframe_converter(tracing_func):
    """Return a converter that traces *tracing_func* and embeds it in the graph."""

    def converter(g, sts, outputs, estimator, *inputs, name="df_transform"):
        from yobx.sql import trace_dataframe
        from yobx.sql.sql_convert import parsed_query_to_onnx_graph

        pq = trace_dataframe(tracing_func, estimator.input_dtypes_)
        out_names = parsed_query_to_onnx_graph(
            g, sts, list(outputs), pq, estimator.input_dtypes_, _finalize=False
        )
        return out_names[0]

    return converter


# %%
# 1. Single output column
# -----------------------


class AddColumnsTransformer(BaseEstimator, TransformerMixin):
    """Adds columns 'a' and 'b' to produce 'total'."""

    _INPUT_DTYPES = {"a": np.float32, "b": np.float32}

    def fit(self, X=None, y=None):
        self.input_dtypes_ = {k: np.dtype(v) for k, v in self._INPUT_DTYPES.items()}
        return self

    def transform(self, df):
        return pd.DataFrame({"total": df["a"].values + df["b"].values})

    def get_feature_names_out(self, input_features=None):
        return np.array(["total"])


def add_columns_func(df):
    """TracedDataFrame-compatible version of AddColumnsTransformer.transform."""
    return df.select([(df["a"] + df["b"]).alias("total")])


t_add = AddColumnsTransformer()
t_add.fit()

# Build input descriptor tuples: one per column.
args_add = tuple((col, dtype, ("N",)) for col, dtype in t_add.input_dtypes_.items())

onx_add = to_onnx(
    t_add,
    args_add,
    extra_converters={AddColumnsTransformer: make_dataframe_converter(add_columns_func)},
)

print("ONNX inputs :", [i.name for i in onx_add.proto.graph.input])
print("ONNX outputs:", [o.name for o in onx_add.proto.graph.output])

ref = ExtendedReferenceEvaluator(onx_add)
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
(total,) = ref.run(None, {"a": a, "b": b})
print("\ntotal =", total)
assert np.allclose(total, a + b)
print("Single output ✓")

# %%
# 2. Multiple output columns
# --------------------------


class MultiOutTransformer(BaseEstimator, TransformerMixin):
    """Produces 'total' and 'a_doubled' from columns 'a' and 'b'."""

    _INPUT_DTYPES = {"a": np.float32, "b": np.float32}

    def fit(self, X=None, y=None):
        self.input_dtypes_ = {k: np.dtype(v) for k, v in self._INPUT_DTYPES.items()}
        return self

    def transform(self, df):
        return pd.DataFrame(
            {"total": df["a"].values + df["b"].values, "a_doubled": df["a"].values * 2.0}
        )

    def get_feature_names_out(self, input_features=None):
        return np.array(["total", "a_doubled"])


def multi_out_func(df):
    return df.select(
        [(df["a"] + df["b"]).alias("total"), (df["a"] * 2.0).alias("a_doubled")]
    )


t_multi = MultiOutTransformer()
t_multi.fit()
args_multi = tuple((col, dtype, ("N",)) for col, dtype in t_multi.input_dtypes_.items())

onx_multi = to_onnx(
    t_multi,
    args_multi,
    extra_converters={MultiOutTransformer: make_dataframe_converter(multi_out_func)},
)

ref_multi = ExtendedReferenceEvaluator(onx_multi)
total, a_doubled = ref_multi.run(None, {"a": a, "b": b})
print(f"\ntotal     = {total}")
print(f"a_doubled = {a_doubled}")
assert np.allclose(total, a + b) and np.allclose(a_doubled, a * 2.0)
print("Multiple outputs ✓")

# %%
# 3. Filter + select
# ------------------


class FilterTransformer(BaseEstimator, TransformerMixin):
    """Keeps rows where 'a' > 0 and returns 'total' = a + b."""

    _INPUT_DTYPES = {"a": np.float32, "b": np.float32}

    def fit(self, X=None, y=None):
        self.input_dtypes_ = {k: np.dtype(v) for k, v in self._INPUT_DTYPES.items()}
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array(["total"])


def filter_func(df):
    df = df.filter(df["a"] > 0)
    return df.select([(df["a"] + df["b"]).alias("total")])


t_filter = FilterTransformer()
t_filter.fit()
args_filter = tuple((col, dtype, ("N",)) for col, dtype in t_filter.input_dtypes_.items())

onx_filter = to_onnx(
    t_filter,
    args_filter,
    extra_converters={FilterTransformer: make_dataframe_converter(filter_func)},
)

ref_filter = ExtendedReferenceEvaluator(onx_filter)
a_mix = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
b_mix = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)
(total_pos,) = ref_filter.run(None, {"a": a_mix, "b": b_mix})
print(f"\nFiltered total = {total_pos}  (rows where a > 0 only)")
assert np.allclose(total_pos, np.array([5.0, 9.0], dtype=np.float32))
print("Filter + select ✓")

# %%
# 4. Visualize the ONNX graph
# ---------------------------

plot_dot(onx_multi)

