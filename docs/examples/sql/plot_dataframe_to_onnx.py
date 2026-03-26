"""
.. _l-plot-dataframe-to-onnx:

DataFrame function to ONNX
==========================

:func:`~yobx.sql.dataframe_to_onnx` converts a plain Python function that
operates on :class:`~yobx.xtracing.dataframe_trace.TracedDataFrame` objects
into a self-contained :class:`onnx.ModelProto`.  The function is *traced*:
each column is represented as a lightweight proxy, and every operation
performed on that proxy is recorded as an ONNX node.

Each input column becomes a **separate 1-D ONNX input tensor**.  The outputs
correspond to the columns produced by the final ``select`` (or ``join``) call.

A :class:`pandas.DataFrame` can be passed as the second argument to
:func:`~yobx.sql.dataframe_to_onnx` so that column names and dtypes are
inferred automatically from the data you already have.

This example covers:

1. **Basic SELECT** — column pass-through and arithmetic expression.
2. **Multi-column SELECT** — selecting and renaming several columns at once.
3. **WHERE clause** — row filtering with ``filter``.
4. **Aggregations** — ``sum()``, ``mean()``, ``min()``, ``max()``.
5. **Two independent dataframes** — function that receives two frames and
   combines their columns without a join.
6. **Two dataframes joined on a key column** — the ``join`` method.
7. **Graph visualisation** — inspecting the produced ONNX model.

In each section the **same function** that is passed to
:func:`~yobx.sql.dataframe_to_onnx` is also called directly on a real
:class:`pandas.DataFrame` to compute the reference values used to validate
the ONNX output.  Because the TracedDataFrame API (``df.select()``,
``df.filter()``, ``series.alias()``, and ``series.sum()/.mean()/.min()/.max()``)
is not part of the standard pandas interface, these methods are added to pandas
at the top of the script so that the **exact same function** works for both
tracing and reference execution.

See :ref:`l-plot-sql-to-onnx` for the equivalent SQL-string API and
:ref:`l-plot-lazyframe-to-onnx` for the Polars LazyFrame API.
"""

import numpy as np
import onnxruntime
import pandas as pd
from yobx.helpers.onnx_helper import pretty_onnx
from yobx.sql import dataframe_to_onnx

# ---------------------------------------------------------------------------
# Extend pandas with TracedDataFrame-compatible methods so that the transform
# functions can be executed on a *real* DataFrame (for reference values) using
# the exact same code path that is used for ONNX tracing.
# ---------------------------------------------------------------------------

# series.alias("name") → renamed Series (mirroring TracedSeries.alias)
pd.Series.alias = lambda self, name: self.rename(name)


# df.filter(boolean_series) → row-filtered DataFrame (mirroring TracedDataFrame.filter)
_pd_filter_orig = pd.DataFrame.filter


def _pd_filter_compat(self, cond=None, **kwargs):
    if isinstance(cond, pd.Series) and pd.api.types.is_bool_dtype(cond):
        return self.loc[cond]
    return _pd_filter_orig(self, cond, **kwargs)


pd.DataFrame.filter = _pd_filter_compat


# df.select([series, ...]) → DataFrame with those columns
def _pd_df_select(self, exprs):
    return pd.DataFrame({e.name: e.values for e in exprs if isinstance(e, pd.Series)})


pd.DataFrame.select = _pd_df_select


# series.sum()/.mean()/.min()/.max() → scalar-with-alias so that
# .alias("col_name") returns a named one-element Series usable in select().
class _Scalar(float):
    """float subclass that additionally supports ``.alias()`` for aggregations."""

    def alias(self, name):
        return pd.Series([float(self)], name=name)


def _make_agg(orig_fn):
    def _agg(self, *args, **kw):
        return _Scalar(orig_fn(self, *args, **kw))

    return _agg


for method_name in ("sum", "mean", "min", "max"):
    setattr(pd.Series, method_name, _make_agg(getattr(pd.Series, method_name)))

# df.join(other, left_key=..., right_key=...) → merged DataFrame
_pd_join_orig = pd.DataFrame.join


def _pd_join_compat(self, other, left_key=None, right_key=None, **kwargs):
    # TracedDataFrame.join() also uses inner join semantics.
    if left_key is not None and right_key is not None:
        return pd.merge(self, other, left_on=left_key, right_on=right_key, how="inner")
    return _pd_join_orig(self, other, **kwargs)


pd.DataFrame.join = _pd_join_compat

# %%
# 1. Basic SELECT — arithmetic expression
# ----------------------------------------
#
# The simplest case: define a function that selects one computed column.
# :func:`~yobx.sql.dataframe_to_onnx` traces the function with proxy
# objects and produces a single ``Add`` ONNX node.
#
# A :class:`pandas.DataFrame` is passed as the second argument so that
# column names and dtypes are inferred automatically.

a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
df = pd.DataFrame({"a": a, "b": b})

def transform_add(df):
    return df.select([(df["a"] + df["b"]).alias("total")])


artifact_add = dataframe_to_onnx(transform_add, df)

# Reference: call the same transform function on the real DataFrame.
ref_total = transform_add(df)["total"].to_numpy()
(ort_total,) = onnxruntime.InferenceSession(
    artifact_add.SerializeToString(), providers=["CPUExecutionProvider"]
).run(None, {"a": a, "b": b})
np.testing.assert_allclose(ref_total, ort_total, rtol=1e-5, atol=1e-6)
print("a + b =", ort_total)

# %%
# The ONNX model (one ``Add`` node, two inputs, one output):
print(pretty_onnx(artifact_add.proto))

# %%
# 2. Multi-column SELECT
# -----------------------
#
# Functions can produce several output columns at once.  Here we pass
# both source columns through unchanged and add a computed ``product`` column.

def transform_multi(df):
    return df.select([df["a"], df["b"], (df["a"] * df["b"]).alias("product")])


artifact_multi = dataframe_to_onnx(transform_multi, df)

# Reference: call the same transform function on the real DataFrame.
ref = transform_multi(df)
ref_a = ref["a"].to_numpy()
ref_b = ref["b"].to_numpy()
ref_prod = ref["product"].to_numpy()
ort_a, ort_b, ort_prod = onnxruntime.InferenceSession(
    artifact_multi.SerializeToString(), providers=["CPUExecutionProvider"]
).run(None, {"a": a, "b": b})
np.testing.assert_allclose(ref_a, ort_a, rtol=1e-5, atol=1e-6)
np.testing.assert_allclose(ref_b, ort_b, rtol=1e-5, atol=1e-6)
np.testing.assert_allclose(ref_prod, ort_prod, rtol=1e-5, atol=1e-6)
print("a      =", ort_a)
print("b      =", ort_b)
print("a * b  =", ort_prod)

# %%
# The ONNX model
print(pretty_onnx(artifact_multi.proto))

# %%
# 3. WHERE clause — row filtering
# --------------------------------
#
# ``df.filter(condition)`` is translated to a boolean mask followed by
# ``Compress`` nodes that select only the matching rows from every output
# column.


def transform_filter(df):
    df = df.filter(df["a"] > 1.5)
    return df.select([df["a"], df["b"]])


artifact_filter = dataframe_to_onnx(transform_filter, df)

# Reference: call the same transform function on the real DataFrame.
ref_filter = transform_filter(df)
ref_af = ref_filter["a"].to_numpy()
ref_bf = ref_filter["b"].to_numpy()
ort_af, ort_bf = onnxruntime.InferenceSession(
    artifact_filter.SerializeToString(), providers=["CPUExecutionProvider"]
).run(None, {"a": a, "b": b})
np.testing.assert_allclose(ref_af, ort_af, rtol=1e-5, atol=1e-6)
np.testing.assert_allclose(ref_bf, ort_bf, rtol=1e-5, atol=1e-6)
print("rows where a > 1.5:")
print("  a =", ort_af)
print("  b =", ort_bf)

# %%
# Filter and arithmetic can be chained freely:

a2 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
b2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
df2 = pd.DataFrame({"a": a2, "b": b2})


def transform_filter_add(df):
    df = df.filter(df["a"] > 0)
    return df.select([(df["a"] + df["b"]).alias("total")])


artifact_filter_add = dataframe_to_onnx(transform_filter_add, df2)

# Reference: call the same transform function on the real DataFrame.
ref_total2 = transform_filter_add(df2)["total"].to_numpy()
(ort_total2,) = onnxruntime.InferenceSession(
    artifact_filter_add.SerializeToString(), providers=["CPUExecutionProvider"]
).run(None, {"a": a2, "b": b2})
np.testing.assert_allclose(ref_total2, ort_total2, rtol=1e-5, atol=1e-6)
print("(a + b) WHERE a > 0 =", ort_total2)

# %%
# The ONNX model
print(pretty_onnx(artifact_filter_add.proto))

# %%
# 4. Aggregation functions
# -------------------------
#
# Column-level aggregation methods — ``sum()``, ``mean()``, ``min()``,
# ``max()`` — map to ``ReduceSum``, ``ReduceMean``, ``ReduceMin``, and
# ``ReduceMax`` ONNX nodes respectively.  The compatibility shim installed
# above makes these methods return a scalar-with-``.alias()`` so that the
# same function works on a real DataFrame.


def transform_agg(df):
    return df.select(
        [
            df["a"].sum().alias("sum_a"),
            df["b"].mean().alias("mean_b"),
            df["a"].min().alias("min_a"),
            df["b"].max().alias("max_b"),
        ]
    )


artifact_agg = dataframe_to_onnx(transform_agg, df)

# Reference: call the same transform function on the real DataFrame.
ref_agg = transform_agg(df)
ref_sum_a = float(ref_agg["sum_a"].iloc[0])
ref_mean_b = float(ref_agg["mean_b"].iloc[0])
ref_min_a = float(ref_agg["min_a"].iloc[0])
ref_max_b = float(ref_agg["max_b"].iloc[0])

sum_a, mean_b, min_a, max_b = onnxruntime.InferenceSession(
    artifact_agg.SerializeToString(), providers=["CPUExecutionProvider"]
).run(None, {"a": a, "b": b})
np.testing.assert_allclose(float(sum_a), ref_sum_a, rtol=1e-5)
np.testing.assert_allclose(float(mean_b), ref_mean_b, rtol=1e-5)
np.testing.assert_allclose(float(min_a), ref_min_a, rtol=1e-5)
np.testing.assert_allclose(float(max_b), ref_max_b, rtol=1e-5)
print(f"sum(a)  = {float(sum_a):.1f}")
print(f"mean(b) = {float(mean_b):.1f}")
print(f"min(a)  = {float(min_a):.1f}")
print(f"max(b)  = {float(max_b):.1f}")

# %%
# The ONNX model
print(pretty_onnx(artifact_agg.proto))

# %%
# 5. Two independent dataframes
# ------------------------------
#
# When *func* accepts two arguments, pass a **list** of DataFrames.  The
# columns of both frames are merged into a single ONNX input set (no join).

def transform_two(df1, df2):
    return df1.select([(df1["a"] + df2["b"]).alias("total")])


df_a = pd.DataFrame({"a": a})
df_b = pd.DataFrame({"b": b})
artifact_two = dataframe_to_onnx(transform_two, [df_a, df_b])

# Reference: call the same transform function on the real DataFrames.
ref_two = transform_two(df_a, df_b)["total"].to_numpy()
(ort_two,) = onnxruntime.InferenceSession(
    artifact_two.SerializeToString(), providers=["CPUExecutionProvider"]
).run(None, {"a": a, "b": b})
np.testing.assert_allclose(ref_two, ort_two, rtol=1e-5, atol=1e-6)
print("df1['a'] + df2['b'] =", ort_two)

# %%
# 6. Two dataframes joined on a key column
# -----------------------------------------
#
# Use the ``join`` method to perform an inner join on a shared key.  The left
# frame's key column is specified with ``left_key`` and the right frame's
# key column with ``right_key``.  The compatibility shim maps this to
# :func:`pandas.merge` so the same function works on real data.

cid = np.array([1, 2, 3], dtype=np.int64)
vals_a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
id_ = np.array([1, 2, 3], dtype=np.int64)
vals_b = np.array([100.0, 200.0, 300.0], dtype=np.float32)
df_left = pd.DataFrame({"cid": cid, "a": vals_a})
df_right = pd.DataFrame({"id": id_, "b": vals_b})

def transform_join(df1, df2):
    return df1.join(df2, left_key="cid", right_key="id")


artifact_join = dataframe_to_onnx(transform_join, [df_left, df_right])

feeds = {"cid": cid, "a": vals_a, "id": id_, "b": vals_b}

ort_join = onnxruntime.InferenceSession(
    artifact_join.SerializeToString(), providers=["CPUExecutionProvider"]
).run(None, feeds)
# Reference: call the same transform function on the real DataFrames.
ref_join = transform_join(df_left, df_right)
np.testing.assert_array_equal(ref_join["cid"].to_numpy(), ort_join[0])
np.testing.assert_allclose(ref_join["a"].to_numpy(), ort_join[1], rtol=1e-5)
np.testing.assert_array_equal(ref_join["id"].to_numpy(), ort_join[2])
np.testing.assert_allclose(ref_join["b"].to_numpy(), ort_join[3], rtol=1e-5)
print("join outputs:")
for col_name, col_val in zip(["cid", "a", "id", "b"], ort_join):
    print(f"  {col_name} = {col_val}")

# %%
# 7. Visualise the ONNX node types
# ---------------------------------
#
# The bar chart below compares how many ONNX nodes each function produces and
# which node types appear in the filter + arithmetic model.

import matplotlib.pyplot as plt  # noqa: E402

models = {
    "add": artifact_add.proto,
    "multi-col": artifact_multi.proto,
    "filter": artifact_filter_add.proto,
    "aggregation": artifact_agg.proto,
    "two frames": artifact_two.proto,
}

node_counts = [len(list(m.graph.node)) for m in models.values()]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
bars = ax.bar(list(models.keys()), node_counts, color="#4c72b0")
ax.set_ylabel("Number of ONNX nodes")
ax.set_title("ONNX node count per DataFrame function")
for bar, count in zip(bars, node_counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        str(count),
        ha="center",
        va="bottom",
        fontsize=9,
    )
ax.tick_params(axis="x", labelrotation=15)

op_types: dict[str, int] = {}
for node in artifact_filter_add.proto.graph.node:
    op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

ax2 = axes[1]
ax2.bar(list(op_types.keys()), list(op_types.values()), color="#dd8452")
ax2.set_ylabel("Count")
ax2.set_title("Node types in 'filter+arithmetic' model")
ax2.tick_params(axis="x", labelrotation=25)

plt.tight_layout()
plt.show()

# %%
# 8. Display the ONNX model graph
# --------------------------------
#
# :func:`~yobx.doc.plot_dot` renders the ONNX graph as an image so you can
# inspect nodes, edges, and tensor shapes at a glance.  Here we visualize the
# basic ``SELECT a + b`` function — a single ``Add`` node connecting two
# inputs to one output.

from yobx.doc import plot_dot  # noqa: E402

plot_dot(artifact_add.proto)
