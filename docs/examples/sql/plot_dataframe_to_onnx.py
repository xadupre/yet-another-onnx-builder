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

See :ref:`l-plot-sql-to-onnx` for the equivalent SQL-string API and
:ref:`l-plot-lazyframe-to-onnx` for the Polars LazyFrame API.
"""

import numpy as np
import onnxruntime
from yobx.helpers.onnx_helper import pretty_onnx
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sql import dataframe_to_onnx

# %%
# Helper
# ------
#
# A small helper runs a model with both the reference evaluator and
# onnxruntime and verifies the results agree.


def run(artifact, feeds):
    """Run *artifact* through the reference evaluator and ORT; return ref outputs."""
    ref = ExtendedReferenceEvaluator(artifact)
    ref_outputs = ref.run(None, feeds)

    sess = onnxruntime.InferenceSession(
        artifact.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ort_outputs = sess.run(None, feeds)

    for r, o in zip(ref_outputs, ort_outputs):
        np.testing.assert_allclose(r, o, rtol=1e-5, atol=1e-6)
    return ref_outputs


# %%
# 1. Basic SELECT — arithmetic expression
# ----------------------------------------
#
# The simplest case: define a function that selects one computed column.
# :func:`~yobx.sql.dataframe_to_onnx` traces the function with proxy
# objects and produces a single ``Add`` ONNX node.

a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
dtypes = {"a": np.float32, "b": np.float32}


def transform_add(df):
    return df.select([(df["a"] + df["b"]).alias("total")])


artifact_add = dataframe_to_onnx(transform_add, dtypes)

(total,) = run(artifact_add, {"a": a, "b": b})
print("a + b =", total)
np.testing.assert_allclose(total, a + b)

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

a_out, b_out, product = run(artifact_multi, {"a": a, "b": b})
print("a      =", a_out)
print("b      =", b_out)
print("a * b  =", product)
np.testing.assert_allclose(product, a * b)

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

a_filt, b_filt = run(artifact_filter, {"a": a, "b": b})
print("rows where a > 1.5:")
print("  a =", a_filt)
print("  b =", b_filt)
np.testing.assert_allclose(a_filt, np.array([2.0, 3.0], dtype=np.float32))

# %%
# Filter and arithmetic can be chained freely:

def transform_filter_add(df):
    df = df.filter(df["a"] > 0)
    return df.select([(df["a"] + df["b"]).alias("total")])


a2 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
b2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)

artifact_filter_add = dataframe_to_onnx(transform_filter_add, dtypes)

(total2,) = run(artifact_filter_add, {"a": a2, "b": b2})
print("(a + b) WHERE a > 0 =", total2)
np.testing.assert_allclose(total2, np.array([5.0, 9.0], dtype=np.float32))

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

sum_a, mean_b, min_a, max_b = run(artifact_agg, {"a": a, "b": b})
print(f"sum(a)  = {float(sum_a):.1f}  (expected 6.0)")
print(f"mean(b) = {float(mean_b):.1f}  (expected 5.0)")
print(f"min(a)  = {float(min_a):.1f}  (expected 1.0)")
print(f"max(b)  = {float(max_b):.1f}  (expected 6.0)")
assert abs(float(sum_a) - 6.0) < 1e-5
assert abs(float(mean_b) - 5.0) < 1e-5

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

(total_two,) = run(artifact_two, {"a": a, "b": b})
print("df1['a'] + df2['b'] =", total_two)
np.testing.assert_allclose(total_two, a + b)

# %%
# 6. Two dataframes joined on a key column
# -----------------------------------------
#
# Use the ``join`` method to perform an inner join on a shared key.  The left
# frame's key column is specified with ``left_key`` and the right frame's
# key column with ``right_key``.

def transform_join(df1, df2):
    return df1.join(df2, left_key="cid", right_key="id")


dtypes1 = {"cid": np.int64, "a": np.float32}
dtypes2 = {"id": np.int64, "b": np.float32}

artifact_join = dataframe_to_onnx(transform_join, [dtypes1, dtypes2])

cid = np.array([1, 2, 3], dtype=np.int64)
vals_a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
id_ = np.array([1, 2, 3], dtype=np.int64)
vals_b = np.array([100.0, 200.0, 300.0], dtype=np.float32)

join_out = run(artifact_join, {"cid": cid, "a": vals_a, "id": id_, "b": vals_b})
print("join outputs:")
for col_name, col_val in zip(["cid", "a", "id", "b"], join_out):
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
