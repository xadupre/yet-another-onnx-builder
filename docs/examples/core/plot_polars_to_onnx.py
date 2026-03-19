"""
.. _l-plot-polars-to-onnx:

Polars LazyFrame to ONNX
=========================

:func:`~yobx.sql.to_onnx` is a convenience wrapper around
:func:`~yobx.sql.sql_to_onnx` that accepts a :class:`polars.LazyFrame` (or
:class:`polars.DataFrame`) and infers the ONNX input types automatically from
the polars schema — no need to write an explicit ``input_dtypes`` mapping.

The LazyFrame is used **only for its schema** (column names and types).  It is
never collected or executed.  This means you can pass even a lazy pipeline that
hasn't been materialised yet.

This example covers:

1. **Basic SELECT + WHERE** — schema inference and row filtering.
2. **Multiple columns and arithmetic** — element-wise expressions.
3. **JOIN query** — passing a second frame for the right-table schema.
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
# 1. Basic SELECT + WHERE — schema from a LazyFrame
# --------------------------------------------------
#
# We define a lazy frame whose schema has two ``Float32`` columns.
# :func:`~yobx.sql.to_onnx` reads the schema (without collecting the frame)
# and builds an ONNX model that applies the SQL query at inference time.

lf = pl.LazyFrame(
    {
        "a": pl.Series([1.0, -2.0, 3.0], dtype=pl.Float32),
        "b": pl.Series([4.0, 5.0, 6.0], dtype=pl.Float32),
    }
)

onx = to_onnx(lf, "SELECT a + b AS total FROM t WHERE a > 0")

# ONNX inputs are named after the polars columns
print("ONNX inputs:", [inp.name for inp in onx.graph.input])

# Run with numpy arrays (one array per column)
a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
(total,) = run(onx, {"a": a, "b": b})
print("a + b WHERE a > 0 =", total)
np.testing.assert_allclose(total, np.array([5.0, 9.0], dtype=np.float32))

# %%
# 2. Integer and mixed dtypes
# ----------------------------
#
# The polars schema can include integer or boolean columns.

lf2 = pl.LazyFrame(
    {
        "x": pl.Series([10, 20, 30], dtype=pl.Int64),
        "y": pl.Series([1, 2, 3], dtype=pl.Int64),
    }
)

onx2 = to_onnx(lf2, "SELECT x * y AS prod FROM t")
x = np.array([10, 20, 30], dtype=np.int64)
y = np.array([1, 2, 3], dtype=np.int64)
(prod,) = run(onx2, {"x": x, "y": y})
print("x * y =", prod)
np.testing.assert_array_equal(prod, x * y)

# %%
# 3. Using a polars DataFrame (already collected)
# ------------------------------------------------
#
# :func:`~yobx.sql.to_onnx` also accepts a ``polars.DataFrame`` — only the
# ``schema`` property is accessed, the data is not read.

df = pl.DataFrame(
    {
        "a": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32),
        "b": pl.Series([4.0, 5.0, 6.0], dtype=pl.Float32),
    }
)

onx3 = to_onnx(df, "SELECT AVG(a) AS mean_a FROM t")
a3 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
(mean_a,) = run(onx3, {"a": a3})
print(f"AVG(a) = {float(mean_a):.1f}  (expected 2.0)")

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
