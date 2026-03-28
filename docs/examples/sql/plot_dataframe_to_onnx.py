"""
.. _l-plot-dataframe-to-onnx:

DataFrame function to ONNX
==========================

:func:`~yobx.sql.dataframe_to_onnx` converts a Python function that operates
on a :class:`~yobx.xtracing.dataframe_trace.TracedDataFrame` into a
self-contained :class:`onnx.ModelProto`.  The function body is *traced* rather
than executed: every column access, arithmetic expression, filter predicate,
or aggregation is recorded as an AST node and then compiled to ONNX operators
by the SQL back-end.

Each **source column** becomes a separate **1-D ONNX input tensor** named after
the column.  The outputs correspond to the columns produced by the final
``select`` (or filter pass-through) step.

This example covers:

1. **Basic SELECT** — column pass-through and arithmetic expressions.
2. **WHERE clause** — row filtering with comparison predicates.
3. **Aggregations** — ``sum()``, ``mean()``, ``min()``, ``max()``.
4. **Filter + arithmetic combined** — chaining ``filter`` and ``select``.
5. **Two independent dataframes** — operating on separate tables.
6. **Graph visualization** — inspecting the produced ONNX model.

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
# A small helper traces the function, converts it to ONNX, runs the model
# through both the reference evaluator and onnxruntime, and verifies that
# the results agree.


def run(func, dtypes, feeds):
    """Trace *func*, convert to ONNX, run with both evaluators; return outputs."""
    artifact = dataframe_to_onnx(func, dtypes)
    ref = ExtendedReferenceEvaluator(artifact)
    ref_outputs = ref.run(None, feeds)

    sess = onnxruntime.InferenceSession(
        artifact.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ort_outputs = sess.run(None, feeds)

    for r, o in zip(ref_outputs, ort_outputs):
        np.testing.assert_allclose(r, o, rtol=1e-5, atol=1e-6)
    return ref_outputs, artifact


# %%
# 1. Basic SELECT — arithmetic expression
# ----------------------------------------
#
# The simplest case: define a function that selects a computed column from
# two source columns.  ``input_dtypes`` maps each **source** column name to
# its numpy dtype.

dtypes = {"a": np.float32, "b": np.float32}
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)


def transform_add(df):
    return df.select([(df["a"] + df["b"]).alias("total")])


(total,), artifact_add = run(transform_add, dtypes, {"a": a, "b": b})
print("a + b =", total)
np.testing.assert_allclose(total, a + b)

# %%
# Inspect the ONNX graph inputs — one per source column.
print("ONNX inputs:", [inp.name for inp in artifact_add.proto.graph.input])

# %%
# The ONNX model
print(pretty_onnx(artifact_add.proto))

# %%
# 2. WHERE clause — row filtering
# --------------------------------
#
# ``df.filter(condition)`` is translated to a boolean mask followed by
# ``Compress`` nodes that select only the matching rows from every column.


def transform_where(df):
    return df.filter(df["a"] > 1.5)


(a_filt, b_filt), artifact_where = run(transform_where, dtypes, {"a": a, "b": b})
print("rows where a > 1.5:")
print("  a =", a_filt)
print("  b =", b_filt)
np.testing.assert_allclose(a_filt, np.array([2.0, 3.0], dtype=np.float32))

# %%
# The ONNX model
print(pretty_onnx(artifact_where.proto))

# %%
# 3. Aggregation functions
# -------------------------
#
# Column methods ``sum()``, ``mean()``, ``min()``, ``max()``, and ``count()``
# are mapped to the corresponding ``ReduceSum``, ``ReduceMean``, ``ReduceMin``,
# ``ReduceMax``, and ``ReduceSum`` ONNX nodes (for ``count`` a constant-filled
# tensor of ones is summed).


def transform_agg(df):
    return df.select(
        [
            df["a"].sum().alias("s"),
            df["b"].mean().alias("m"),
        ]
    )


(s_arr, m_arr), artifact_agg = run(transform_agg, dtypes, {"a": a, "b": b})
s = float(s_arr)
m = float(m_arr)
print(f"sum(a)  = {s:.1f}  (expected 6.0)")
print(f"mean(b) = {m:.1f}  (expected 5.0)")
assert abs(s - 6.0) < 1e-5
assert abs(m - 5.0) < 1e-5

# %%
# The ONNX model
print(pretty_onnx(artifact_agg.proto))

# %%
# 4. Filter + arithmetic combined
# --------------------------------
#
# Filters and computed expressions can be freely chained.  Here we keep only
# rows where ``a > 0`` and then compute ``a + b`` for those rows.

a2 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
b2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)


def transform_combined(df):
    df = df.filter(df["a"] > 0)
    return df.select([(df["a"] + df["b"]).alias("total")])


(total2,), artifact_combined = run(transform_combined, dtypes, {"a": a2, "b": b2})
print("(a + b) WHERE a > 0 =", total2)
np.testing.assert_allclose(total2, np.array([5.0, 9.0], dtype=np.float32))

# %%
# The ONNX model
print(pretty_onnx(artifact_combined.proto))

# %%
# 5. Two independent dataframes
# ------------------------------
#
# Pass a **list of dtype dicts** to ``dataframe_to_onnx`` when the function
# accepts more than one dataframe.  Each dataframe's columns become separate
# ONNX inputs.  The function below combines a column from ``df1`` with a
# column from ``df2`` without any join.

dtypes_two = [{"a": np.float32}, {"b": np.float32}]


def transform_two_dfs(df1, df2):
    return df1.select([(df1["a"] + df2["b"]).alias("total")])


(total3,), artifact_two = run(
    transform_two_dfs,
    dtypes_two,
    {"a": a, "b": b},
)
print("df1['a'] + df2['b'] =", total3)
np.testing.assert_allclose(total3, a + b)

# %%
# The ONNX model
print(pretty_onnx(artifact_two.proto))

# %%
# 6. Visualise the ONNX node types
# ---------------------------------
#
# The bar chart below compares how many ONNX nodes each DataFrame function
# produces and which node types appear in the combined filter+arithmetic model.

import matplotlib.pyplot as plt  # noqa: E402

models = {
    "basic add": artifact_add.proto,
    "WHERE filter": artifact_where.proto,
    "aggregation": artifact_agg.proto,
    "filter+arith": artifact_combined.proto,
    "two DFs": artifact_two.proto,
}

node_counts = [len(list(m.graph.node)) for m in models.values()]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: node count per function
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
# 7. Display the ONNX model graph
# --------------------------------
#
# :func:`~yobx.doc.plot_dot` renders the ONNX graph as an image so you can
# inspect nodes, edges, and tensor shapes at a glance.  Here we visualize the
# basic ``a + b`` model — a single ``Add`` node connecting two inputs to one
# output.

from yobx.doc import plot_dot  # noqa: E402

plot_dot(artifact_add)
