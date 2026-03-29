"""
.. _l-plot-dataframe-to-onnx:

DataFrame tracing to ONNX
==========================

:func:`~yobx.sql.dataframe_to_onnx` converts a Python function that operates on a
:class:`~yobx.xtracing.dataframe_trace.TracedDataFrame` proxy into a
self-contained :class:`onnx.ModelProto`.  Instead of executing the function,
the tracer records every operation — column access, arithmetic, filtering,
aggregation — as an AST node, then compiles that AST to ONNX through the
existing SQL converter.

Each column name in *input_dtypes* becomes a separate **1-D ONNX input tensor**,
matching the columnar representation used in tabular data pipelines.

This example covers:

1. **Basic SELECT** — column pass-through and arithmetic expressions.
2. **WHERE clause** — row filtering with comparison predicates.
3. **Aggregations** — ``sum()``, ``mean()``, ``min()``, ``max()``.
4. **Column assignment** — adding derived columns with :meth:`~yobx.xtracing.dataframe_trace.TracedDataFrame.assign`.
5. **Multiple output dataframes** — a function that returns a tuple of frames.
6. **Graph visualization** — inspecting the produced ONNX model.

See :ref:`l-design-sql-dataframe` for the full design discussion.
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
# The simplest case: select a computed column from two source columns.
# ``input_dtypes`` maps each source column name to its numpy dtype.


def transform_add(df):
    return df.select([(df["a"] + df["b"]).alias("total")])


dtypes = {"a": np.float32, "b": np.float32}
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

artifact_add = dataframe_to_onnx(transform_add, dtypes)

(total,) = run(artifact_add, {"a": a, "b": b})
print("a + b =", total)
np.testing.assert_allclose(total, a + b)

# %%
# The ONNX model
print(pretty_onnx(artifact_add))

# %%
# 2. WHERE clause — row filtering
# --------------------------------
#
# ``df.filter(condition)`` is translated to a boolean mask followed by
# ``Compress`` nodes that select only the matching rows from every output column.


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
# The ONNX model
print(pretty_onnx(artifact_filter))

# %%
# 3. Aggregation functions
# -------------------------
#
# Column aggregation methods — ``sum()``, ``mean()`` (→ ``ReduceMean``),
# ``min()``, ``max()`` — are mapped to the corresponding ONNX reduction nodes.


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
print(pretty_onnx(artifact_agg))

# %%
# 4. Column assignment — adding a derived column
# -----------------------------------------------
#
# :meth:`~yobx.xtracing.dataframe_trace.TracedDataFrame.assign` adds new
# columns computed from existing ones.  The result retains all source columns
# together with the new derived column.


def transform_assign(df):
    df = df.assign(diff=df["a"] - df["b"])
    return df.select(["a", "b", "diff"])


artifact_assign = dataframe_to_onnx(transform_assign, dtypes)
a_out, b_out, diff_out = run(artifact_assign, {"a": a, "b": b})

print("a     =", a_out)
print("b     =", b_out)
print("a - b =", diff_out)
np.testing.assert_allclose(diff_out, a - b)

# %%
# The ONNX model
print(pretty_onnx(artifact_assign))

# %%
# 5. Filter + arithmetic combined
# ---------------------------------
#
# Filters and computed expressions can be chained freely.  Here we keep only
# rows where ``a > 0`` and then compute ``a + b`` for those rows.


def transform_combined(df):
    df = df.filter(df["a"] > 0)
    return df.select([(df["a"] + df["b"]).alias("total")])


a2 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
b2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)

artifact_combined = dataframe_to_onnx(transform_combined, dtypes)
(total2,) = run(artifact_combined, {"a": a2, "b": b2})

print("(a + b) WHERE a > 0 =", total2)
np.testing.assert_allclose(total2, np.array([5.0, 9.0], dtype=np.float32))

# %%
# The ONNX model
print(pretty_onnx(artifact_combined))

# %%
# 6. Multiple output dataframes
# ------------------------------
#
# When the traced function returns a **tuple** (or list) of
# :class:`~yobx.xtracing.dataframe_trace.TracedDataFrame` objects, all outputs
# are collected into a single ONNX graph with shared inputs and multiple output
# tensors.


def transform_multi_out(df):
    out1 = df.select([(df["a"] + df["b"]).alias("sum_ab")])
    out2 = df.select([(df["a"] - df["b"]).alias("diff_ab")])
    return out1, out2


artifact_multi = dataframe_to_onnx(transform_multi_out, dtypes)
sum_ab, diff_ab = run(artifact_multi, {"a": a, "b": b})

print("a + b =", sum_ab)
print("a - b =", diff_ab)
np.testing.assert_allclose(sum_ab, a + b)
np.testing.assert_allclose(diff_ab, a - b)

# %%
# The ONNX model
print(pretty_onnx(artifact_multi))

# %%
# 7. Visualise the ONNX node types
# ---------------------------------
#
# The bar chart below compares how many ONNX nodes each transformed DataFrame
# produces and which node types appear in the combined filter+arithmetic model.

import matplotlib.pyplot as plt  # noqa: E402

models = {
    "basic add": artifact_add,
    "WHERE filter": artifact_filter,
    "aggregation": artifact_agg,
    "assign": artifact_assign,
    "filter+arith": artifact_combined,
    "multi-output": artifact_multi,
}

node_counts = [len(m.proto.graph.node) for m in models.values()]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: node count per transformation
ax = axes[0]
bars = ax.bar(list(models.keys()), node_counts, color="#4c72b0")
ax.set_ylabel("Number of ONNX nodes")
ax.set_title("ONNX node count per DataFrame transformation")
for bar, count in zip(bars, node_counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        str(count),
        ha="center",
        va="bottom",
        fontsize=9,
    )
ax.tick_params(axis="x", labelrotation=20)

# Right: node types in the combined filter+arithmetic model
op_types: dict[str, int] = {}
for node in artifact_combined.proto.graph.node:
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
# combined filter + arithmetic model.

from yobx.doc import plot_dot  # noqa: E402

plot_dot(artifact_combined.proto)
