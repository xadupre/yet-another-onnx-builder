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

This example covers:

1. **Basic SELECT** — column pass-through and arithmetic expression.
2. **Multi-column SELECT** — selecting and renaming several columns at once.
3. **WHERE clause** — row filtering with ``filter``.
4. **Aggregations** — ``sum()``, ``mean()``, ``min()``, ``max()``.
5. **Two independent dataframes** — function that receives two frames and
   combines their columns without a join.
6. **Two dataframes joined on a key column** — the ``join`` method.
7. **Graph visualisation** — inspecting the produced ONNX model.

In each section the same function is traced to ONNX and then validated by
comparing the :class:`~yobx.reference.ExtendedReferenceEvaluator` output
against :mod:`onnxruntime`.

See :ref:`l-plot-sql-to-onnx` for the equivalent SQL-string API and
:ref:`l-plot-lazyframe-to-onnx` for the Polars LazyFrame API.
"""

import numpy as np
import onnxruntime
from yobx.helpers.onnx_helper import pretty_onnx
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sql import dataframe_to_onnx

# %%
# 1. Basic SELECT — arithmetic expression
# ----------------------------------------
#
# The simplest case: define a function that selects one computed column.
# :func:`~yobx.sql.dataframe_to_onnx` traces the function with proxy
# objects and produces a single ``Add`` ONNX node.
#
# The same *artifact* is executed by both the reference evaluator and
# onnxruntime; their outputs are compared to verify correctness.

a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
dtypes = {"a": np.float32, "b": np.float32}

def transform_add(df):
    return df.select([(df["a"] + df["b"]).alias("total")])

artifact_add = dataframe_to_onnx(transform_add, dtypes)

(ref_total,) = ExtendedReferenceEvaluator(artifact_add).run(None, {"a": a, "b": b})
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

artifact_multi = dataframe_to_onnx(transform_multi, dtypes)

ref_a, ref_b, ref_prod = ExtendedReferenceEvaluator(artifact_multi).run(
    None, {"a": a, "b": b}
)
ort_a, ort_b, ort_prod = onnxruntime.InferenceSession(
    artifact_multi.SerializeToString(), providers=["CPUExecutionProvider"]
).run(None, {"a": a, "b": b})
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


artifact_filter = dataframe_to_onnx(transform_filter, dtypes)

ref_af, ref_bf = ExtendedReferenceEvaluator(artifact_filter).run(None, {"a": a, "b": b})
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


def transform_filter_add(df):
    df = df.filter(df["a"] > 0)
    return df.select([(df["a"] + df["b"]).alias("total")])


artifact_filter_add = dataframe_to_onnx(transform_filter_add, dtypes)

(ref_total2,) = ExtendedReferenceEvaluator(artifact_filter_add).run(
    None, {"a": a2, "b": b2}
)
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
# ``ReduceMax`` ONNX nodes respectively.


def transform_agg(df):
    return df.select(
        [
            df["a"].sum().alias("sum_a"),
            df["b"].mean().alias("mean_b"),
            df["a"].min().alias("min_a"),
            df["b"].max().alias("max_b"),
        ]
    )


artifact_agg = dataframe_to_onnx(transform_agg, dtypes)

ref_agg = ExtendedReferenceEvaluator(artifact_agg).run(None, {"a": a, "b": b})
ort_agg = onnxruntime.InferenceSession(
    artifact_agg.SerializeToString(), providers=["CPUExecutionProvider"]
).run(None, {"a": a, "b": b})
for ref_v, ort_v in zip(ref_agg, ort_agg):
    np.testing.assert_allclose(ref_v, ort_v, rtol=1e-5, atol=1e-6)
sum_a, mean_b, min_a, max_b = ort_agg
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
# When *func* accepts two arguments, pass a **list** of dtype dicts.  The
# columns of both frames are merged into a single ONNX input set (no join).

def transform_two(df1, df2):
    return df1.select([(df1["a"] + df2["b"]).alias("total")])

artifact_two = dataframe_to_onnx(transform_two, [{"a": np.float32}, {"b": np.float32}])

(ref_two,) = ExtendedReferenceEvaluator(artifact_two).run(None, {"a": a, "b": b})
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
# key column with ``right_key``.

dtypes1 = {"cid": np.int64, "a": np.float32}
dtypes2 = {"id": np.int64, "b": np.float32}

def transform_join(df1, df2):
    return df1.join(df2, left_key="cid", right_key="id")

artifact_join = dataframe_to_onnx(transform_join, [dtypes1, dtypes2])

cid = np.array([1, 2, 3], dtype=np.int64)
vals_a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
id_ = np.array([1, 2, 3], dtype=np.int64)
vals_b = np.array([100.0, 200.0, 300.0], dtype=np.float32)
feeds = {"cid": cid, "a": vals_a, "id": id_, "b": vals_b}

ref_join = ExtendedReferenceEvaluator(artifact_join).run(None, feeds)
ort_join = onnxruntime.InferenceSession(
    artifact_join.SerializeToString(), providers=["CPUExecutionProvider"]
).run(None, feeds)
for ref_v, ort_v in zip(ref_join, ort_join):
    np.testing.assert_allclose(ref_v, ort_v, rtol=1e-5, atol=1e-6)
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
