"""
.. _l-plot-lazyframe-to-onnx:

Polars LazyFrame to ONNX
========================

:func:`~yobx.sql.lazyframe_to_onnx` converts a :class:`polars.LazyFrame`
directly into a self-contained :class:`onnx.ModelProto`.  Internally the
function calls :meth:`polars.LazyFrame.explain` to obtain the logical
execution plan, translates it into a SQL query, and then delegates to
:func:`~yobx.sql.sql_to_onnx`.

Each *source* column becomes a separate **1-D ONNX input tensor**; the
outputs correspond to the expressions in the ``select`` (or ``agg``) step.

This example covers:

1. **Basic SELECT** — column pass-through and arithmetic expressions.
2. **WHERE clause** — row filtering with comparison predicates.
3. **Aggregations** — ``sum()``, ``mean()``, ``min()``, ``max()``.
4. **Filter + arithmetic combined** — chaining ``filter`` and ``select``.
5. **Graph visualization** — inspecting the produced ONNX model.

See :ref:`l-plot-sql-to-onnx` for the lower-level SQL → ONNX API.
"""

import numpy as np
import onnxruntime
import polars as pl
from yobx.helpers.onnx_helper import pretty_onnx
from yobx.sql import lazyframe_to_onnx

# %%
# 1. Basic SELECT — arithmetic expression
# ----------------------------------------
#
# The simplest case: select a computed column from two source columns.
# ``input_dtypes`` maps each **source** column name to its numpy dtype;
# only columns that actually appear in the plan need to be listed.

lf_add = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
lf_add = lf_add.select([(pl.col("a") + pl.col("b")).alias("total")])

dtypes = {"a": np.float64, "b": np.float64}
artifact_add = lazyframe_to_onnx(lf_add, dtypes)

a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
sess = onnxruntime.InferenceSession(
    artifact_add.SerializeToString(), providers=["CPUExecutionProvider"]
)
(total,) = sess.run(None, {"a": a, "b": b})
print("a + b =", total)
np.testing.assert_allclose(total, a + b)

# %%
# The ONNX model
print(pretty_onnx(artifact_add.proto))

# %%
# 2. WHERE clause — row filtering
# --------------------------------
#
# ``filter`` is translated to a boolean mask followed by ``Compress`` nodes
# that select only the matching rows from every output column.

lf_where = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
lf_where = lf_where.filter(pl.col("a") > 1.5).select([pl.col("a"), pl.col("b")])

artifact_where = lazyframe_to_onnx(lf_where, dtypes)
sess = onnxruntime.InferenceSession(
    artifact_where.SerializeToString(), providers=["CPUExecutionProvider"]
)
a_filt, b_filt = sess.run(None, {"a": a, "b": b})

print("rows where a > 1.5:")
print("  a =", a_filt)
print("  b =", b_filt)
np.testing.assert_allclose(a_filt, np.array([2.0, 3.0]))

# %%
# The ONNX model
print(pretty_onnx(artifact_where.proto))

# %%
# 3. Aggregation functions
# -------------------------
#
# Polars aggregation methods — ``sum()``, ``mean()`` (→ ``AVG``), ``min()``,
# ``max()`` — are mapped to the corresponding ``ReduceSum``, ``ReduceMean``,
# ``ReduceMin``, and ``ReduceMax`` ONNX nodes.

lf_agg = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
lf_agg = lf_agg.select([pl.col("a").sum().alias("s"), pl.col("b").mean().alias("m")])

dtypes_agg = {"a": np.float64, "b": np.float64}
artifact_agg = lazyframe_to_onnx(lf_agg, dtypes_agg)
sess = onnxruntime.InferenceSession(
    artifact_agg.SerializeToString(), providers=["CPUExecutionProvider"]
)
s_arr, m_arr = sess.run(None, {"a": a, "b": b})
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
# Filters and computed expressions can be chained freely.  Here we keep only
# rows where ``a > 0`` and then compute ``a + b`` for those rows.

lf_combined = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
lf_combined = lf_combined.filter(pl.col("a") > 0).select(
    [(pl.col("a") + pl.col("b")).alias("total")]
)

a2 = np.array([1.0, -2.0, 3.0], dtype=np.float64)
b2 = np.array([4.0, 5.0, 6.0], dtype=np.float64)

artifact_combined = lazyframe_to_onnx(lf_combined, dtypes)
sess = onnxruntime.InferenceSession(
    artifact_combined.SerializeToString(), providers=["CPUExecutionProvider"]
)
(total2,) = sess.run(None, {"a": a2, "b": b2})
print("(a + b) WHERE a > 0 =", total2)
np.testing.assert_allclose(total2, np.array([5.0, 9.0]))

# %%
# The ONNX model
print(pretty_onnx(artifact_combined.proto))

# %%
# 5. Visualise the ONNX node types
# ---------------------------------
#
# The bar chart below compares how many ONNX nodes each LazyFrame plan
# produces and which node types appear in the combined filter+arithmetic model.

import matplotlib.pyplot as plt  # noqa: E402

models = {
    "basic add": artifact_add.proto,
    "WHERE filter": artifact_where.proto,
    "aggregation": artifact_agg.proto,
    "filter+arith": artifact_combined.proto,
}

node_counts = [len(list(m.graph.node)) for m in models.values()]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: node count per LazyFrame plan
ax = axes[0]
bars = ax.bar(list(models.keys()), node_counts, color="#4c72b0")
ax.set_ylabel("Number of ONNX nodes")
ax.set_title("ONNX node count per LazyFrame plan")
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
# 6. Display the ONNX model graph
# --------------------------------
#
# :func:`~yobx.doc.plot_dot` renders the ONNX graph as an image so you can
# inspect nodes, edges, and tensor shapes at a glance.  Here we visualize the
# combined filter + arithmetic model.

from yobx.doc import plot_dot  # noqa: E402

plot_dot(artifact_combined.proto)
