"""
.. _l-plot-sql-to-onnx:

SQL queries to ONNX
===================

:func:`~yobx.sql.sql_to_onnx` converts a SQL query string into a
self-contained :class:`onnx.ModelProto`.  Each referenced column becomes a
**separate 1-D ONNX input**, matching the columnar representation used in
tabular data pipelines.

This example covers:

1. **Basic SELECT** — column pass-through and arithmetic.
2. **WHERE clause** — row filtering with comparison predicates.
3. **Aggregations** — ``SUM``, ``AVG``, ``MIN``, ``MAX`` in the SELECT list.
4. **Custom Python functions** — user-defined numpy functions called directly
   from SQL via the ``custom_functions`` parameter; the function body is
   traced to ONNX nodes using :func:`~yobx.xtracing.trace_numpy_function`.
5. **Graph visualization** — rendering the produced ONNX model with
   :func:`~yobx.doc.plot_dot`.

See :ref:`l-design-sql` for the full design discussion.
"""

import numpy as np
import onnxruntime
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sql import sql_to_onnx

# %%
# Helper
# ------
#
# A small helper runs a model with both the reference evaluator and
# onnxruntime and verifies the results agree.


def run(onx, feeds):
    """Run *onx* through the reference evaluator and ORT; return ref outputs."""
    ref = ExtendedReferenceEvaluator(onx)
    ref_outputs = ref.run(None, feeds)

    sess = onnxruntime.InferenceSession(
        onx.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ort_outputs = sess.run(None, feeds)

    for r, o in zip(ref_outputs, ort_outputs):
        np.testing.assert_allclose(r, o, rtol=1e-5, atol=1e-6)
    return ref_outputs


# %%
# 1. Basic SELECT
# ---------------
#
# The simplest query selects two columns and computes their element-wise sum.
# Each column in *input_dtypes* becomes a separate ONNX graph input.

dtypes = {"a": np.float32, "b": np.float32}
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

onx_add = sql_to_onnx("SELECT a + b AS total FROM t", dtypes)

(total,) = run(onx_add, {"a": a, "b": b})
print("a + b =", total)

# Inspect the ONNX graph inputs — one per column
print("ONNX inputs:", [inp.name for inp in onx_add.graph.input])

# %%
# 2. WHERE clause (row filtering)
# --------------------------------
#
# The ``WHERE`` clause is translated to a boolean mask followed by
# ``Compress`` nodes that select only the matching rows from every column.

onx_where = sql_to_onnx("SELECT a, b FROM t WHERE a > 1.5", dtypes)
a_filt, b_filt = run(onx_where, {"a": a, "b": b})

print("rows where a > 1.5:")
print("  a =", a_filt)
print("  b =", b_filt)
assert list(a_filt) == [2.0, 3.0]

# %%
# 3. Aggregation functions
# -------------------------
#
# ``SUM``, ``AVG``, ``MIN``, and ``MAX`` in the SELECT list are emitted as
# ``ReduceSum``, ``ReduceMean``, ``ReduceMin``, and ``ReduceMax`` ONNX nodes.

onx_agg = sql_to_onnx("SELECT SUM(a) AS s, AVG(b) AS m FROM t", dtypes)
s_arr, m_arr = run(onx_agg, {"a": a, "b": b})
s = float(s_arr)
m = float(m_arr)

print(f"SUM(a) = {s:.1f}  (expected 6.0)")
print(f"AVG(b) = {m:.1f}  (expected 5.0)")
assert abs(s - 6.0) < 1e-5
assert abs(m - 5.0) < 1e-5

# %%
# 4. WHERE + arithmetic SELECT
# ----------------------------
#
# Filters and expressions can be combined freely.

onx_combined = sql_to_onnx("SELECT a + b AS total FROM t WHERE a > 0", dtypes)
a2 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
b2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)

(total2,) = run(onx_combined, {"a": a2, "b": b2})
print("a + b WHERE a > 0 =", total2)
np.testing.assert_allclose(total2, np.array([5.0, 9.0], dtype=np.float32))

# %%
# 5. Custom Python functions
# ---------------------------
#
# Any numpy-backed Python function can be called by name directly in SQL.
# Pass it as a ``custom_functions`` dictionary entry: the key is the
# function name as it appears in the SQL string, and the value is the
# callable.
#
# Under the hood :func:`~yobx.xtracing.trace_numpy_function` replaces the
# real inputs with lightweight proxy objects that record every numpy
# operation as an ONNX node, so the resulting graph is equivalent to
# running the Python function at inference time.


def clip_sqrt(x):
    """Safe square root: sqrt(max(x, 0))."""
    return np.sqrt(np.maximum(x, np.float32(0)))


dtypes_a = {"a": np.float32}
a3 = np.array([4.0, -1.0, 9.0, 0.0], dtype=np.float32)

onx_func = sql_to_onnx(
    "SELECT clip_sqrt(a) AS r FROM t", dtypes_a, custom_functions={"clip_sqrt": clip_sqrt}
)

(r,) = run(onx_func, {"a": a3})
expected = clip_sqrt(a3)
print("clip_sqrt(a) =", r)
np.testing.assert_allclose(r, expected, atol=1e-6)
print("clip_sqrt ✓")

# %%
# 6. Custom function in WHERE clause
# -----------------------------------
#
# Custom functions also work in ``WHERE`` predicates.

onx_where_func = sql_to_onnx(
    "SELECT a FROM t WHERE clip_sqrt(a) > 1", dtypes_a, custom_functions={"clip_sqrt": clip_sqrt}
)

(a_filt2,) = run(onx_where_func, {"a": a3})
print("a WHERE clip_sqrt(a) > 1 =", a_filt2)
np.testing.assert_allclose(a_filt2, a3[clip_sqrt(a3) > 1], atol=1e-6)
print("WHERE custom function ✓")

# %%
# 7. Two-argument custom function
# ---------------------------------
#
# Functions with more than one argument receive one tensor per argument.


def weighted_sum(x, y, alpha=0.5):
    """Compute alpha * x + (1 - alpha) * y."""
    return alpha * x + (np.float32(1) - np.float32(alpha)) * y


dtypes_ab = {"a": np.float32, "b": np.float32}
onx_ws = sql_to_onnx(
    "SELECT wsum(a, b) AS ws FROM t", dtypes_ab, custom_functions={"wsum": weighted_sum}
)

(ws,) = run(onx_ws, {"a": a, "b": b})
expected_ws = weighted_sum(a, b)
print("weighted_sum(a, b) =", ws)
np.testing.assert_allclose(ws, expected_ws, atol=1e-6)
print("Two-argument custom function ✓")

# %%
# 8. Visualise the ONNX node types
# -----------------------------------
#
# The bar chart below compares how many ONNX nodes each query produces and
# which node types appear in the custom-function query.

import matplotlib.pyplot as plt  # noqa: E402

models = {
    "basic add": onx_add,
    "WHERE filter": onx_where,
    "aggregation": onx_agg,
    "custom func": onx_func,
    "custom WHERE": onx_where_func,
}

node_counts = [len(list(m.graph.node)) for m in models.values()]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: node count per query
ax = axes[0]
bars = ax.bar(list(models.keys()), node_counts, color="#4c72b0")
ax.set_ylabel("Number of ONNX nodes")
ax.set_title("ONNX node count per SQL query")
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

# Right: node types in custom-function model
op_types: dict[str, int] = {}
for node in onx_func.graph.node:
    op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

ax2 = axes[1]
ax2.bar(list(op_types.keys()), list(op_types.values()), color="#dd8452")
ax2.set_ylabel("Count")
ax2.set_title("Node types in 'clip_sqrt' query")
ax2.tick_params(axis="x", labelrotation=25)

plt.tight_layout()
plt.show()

# %%
# 9. Display the ONNX model graph
# --------------------------------
#
# :func:`~yobx.doc.plot_dot` renders the ONNX graph as an image so you can
# inspect nodes, edges, and tensor shapes at a glance.  Here we visualize the
# basic ``SELECT a + b`` query — a single ``Add`` node connecting two inputs
# to one output.

from yobx.doc import plot_dot  # noqa: E402

plot_dot(onx_add)
