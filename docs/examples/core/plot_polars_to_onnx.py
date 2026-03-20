"""
.. _l-plot-polars-to-onnx:

Polars LazyFrame to ONNX
=========================

:func:`~yobx.sql.to_onnx` is a convenience wrapper around
:func:`~yobx.sql.sql_to_onnx` that accepts a :class:`polars.LazyFrame` (or
:class:`polars.DataFrame`) and infers the ONNX input types automatically from
the polars schema — no need to write an explicit ``input_dtypes`` mapping.

There are two calling conventions:

**Query-embedded** (preferred) — call :meth:`polars.LazyFrame.sql` to attach
a SQL query to the frame and pass the result directly to
:func:`~yobx.sql.to_onnx`.  The SQL and input schema are both extracted from
the frame's logical plan::

    onx = to_onnx(src.sql("SELECT a + b AS total FROM self WHERE a > 0"))

**Explicit query** — pass the source frame for schema inference together with
a SQL string (the original calling convention, still fully supported)::

    onx = to_onnx(src, "SELECT a + b AS total FROM t WHERE a > 0")

This example covers:

1. **Query-embedded** — using ``LazyFrame.sql()`` to embed the query.
2. **Explicit query** — passing a SQL string directly.
3. **Multiple columns and arithmetic** — element-wise expressions.
4. **Custom functions** — user-defined numpy functions called from SQL.

See :ref:`l-design-sql` for the full design discussion.
"""

import numpy as np
import polars as pl

from yobx.reference import ExtendedReferenceEvaluator
from yobx.sql import to_onnx

# %%
# Helper
# ------
#
# A small helper runs a model with the reference evaluator and returns
# the outputs.


def run(onx, feeds):
    """Run *onx* through the reference evaluator and return outputs."""
    ref = ExtendedReferenceEvaluator(onx)
    return ref.run(None, feeds)


# %%
# 1. Query-embedded calling convention (preferred)
# ------------------------------------------------
#
# Use :meth:`polars.LazyFrame.sql` to define the query.  The LazyFrame
# returned by ``.sql()`` carries both the SQL and the source schema, so
# :func:`~yobx.sql.to_onnx` needs no extra arguments.
#
# Polars uses ``"self"`` as the default table name inside ``.sql()``.

src = pl.LazyFrame(
    {
        "a": pl.Series([1.0, -2.0, 3.0], dtype=pl.Float32),
        "b": pl.Series([4.0, 5.0, 6.0], dtype=pl.Float32),
    }
)

onx = to_onnx(src.sql("SELECT a + b AS total FROM self WHERE a > 0"))

# ONNX inputs are named after the polars source columns
print("ONNX inputs:", [inp.name for inp in onx.graph.input])

# Run with numpy arrays (one array per column)
a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
(total,) = run(onx, {"a": a, "b": b})
print("a + b WHERE a > 0 =", total)
np.testing.assert_allclose(total, np.array([5.0, 9.0], dtype=np.float32))

# %%
# 2. Explicit query (backward-compatible calling convention)
# ----------------------------------------------------------
#
# Pass the source frame for schema inference together with a SQL string.
# The frame is used **only for its schema** (never collected or executed).

onx2 = to_onnx(src, "SELECT a + b AS total FROM t WHERE a > 0")
(total2,) = run(onx2, {"a": a, "b": b})
print("Explicit query result:", total2)
np.testing.assert_allclose(total2, total, atol=1e-6)

# %%
# 3. Integer and mixed dtypes
# ----------------------------
#
# The polars schema can include integer or boolean columns.

lf3 = pl.LazyFrame(
    {"x": pl.Series([10, 20, 30], dtype=pl.Int64), "y": pl.Series([1, 2, 3], dtype=pl.Int64)}
)

onx3 = to_onnx(lf3.sql("SELECT x * y AS prod FROM self"))
x = np.array([10, 20, 30], dtype=np.int64)
y = np.array([1, 2, 3], dtype=np.int64)
(prod,) = run(onx3, {"x": x, "y": y})
print("x * y =", prod)
np.testing.assert_array_equal(prod, x * y)

# %%
# 4. Custom Python functions
# ---------------------------
#
# Custom numpy functions work exactly the same way as with
# :func:`~yobx.sql.sql_to_onnx`.


def safe_sqrt(x):
    """Square root clamped at zero."""
    return np.sqrt(np.maximum(x, np.float32(0)))


lf4 = pl.LazyFrame({"v": pl.Series([4.0, -1.0, 9.0], dtype=pl.Float32)})
onx4 = to_onnx(lf4, "SELECT safe_sqrt(v) AS r FROM t", custom_functions={"safe_sqrt": safe_sqrt})

v = np.array([4.0, -1.0, 9.0], dtype=np.float32)
(r,) = run(onx4, {"v": v})
print("safe_sqrt(v) =", r)
np.testing.assert_allclose(r, safe_sqrt(v), atol=1e-6)
print("Custom function ✓")

# %%
# 5. Inspecting the ONNX model
# -----------------------------
#
# The returned :class:`onnx.ModelProto` is identical to what
# :func:`~yobx.sql.sql_to_onnx` would produce with manually specified dtypes.

print("Number of ONNX nodes in basic model:", len(list(onx.graph.node)))
print("Node types:", sorted({n.op_type for n in onx.graph.node}))
