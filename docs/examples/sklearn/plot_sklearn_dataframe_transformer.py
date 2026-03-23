"""
.. _l-plot-sklearn-dataframe-transformer:

Exporting DataFrameTransformer to ONNX
=======================================

:class:`~yobx.sklearn.DataFrameTransformer` wraps a user-defined function that
operates on a :class:`pandas.DataFrame`-like object (using the
:class:`~yobx.sql.TracedDataFrame` API) and converts it to an ONNX model via
:func:`yobx.sklearn.to_onnx`.

Under the hood the function is *traced* during :meth:`~yobx.sklearn.DataFrameTransformer.fit`:
each column operation is recorded as an AST node and then compiled to ONNX
by the same infrastructure used by :func:`yobx.sql.dataframe_to_onnx`.

The resulting ONNX model has **one input tensor per named column** and
**one output tensor per output column**.

This example covers:

1. **Single output column** — a simple column addition.
2. **Multiple output columns** — exporting several computed columns at once.
3. **Filter + select** — row filtering combined with a column selection.
4. **Pandas transform via ONNX** — using ``transform()`` on a real
   :class:`pandas.DataFrame` (inference is delegated to :mod:`onnxruntime`).
"""

import numpy as np
import pandas as pd

from yobx.doc import plot_dot
from yobx.sklearn import DataFrameTransformer, to_onnx
from yobx.reference import ExtendedReferenceEvaluator

# %%
# 1. Single output column
# -----------------------
#
# Define a function using the :class:`~yobx.sql.TracedDataFrame` API
# (``df.select``, column arithmetic, ``.alias``).  The same syntax is
# compatible with the tracer and with real DataFrames (via the ONNX model).


def add_columns(df):
    """Return the sum of columns 'a' and 'b' as a new column 'total'."""
    return df.select([(df["a"] + df["b"]).alias("total")])


t_add = DataFrameTransformer(add_columns, {"a": np.float32, "b": np.float32})
t_add.fit()

print("Output column names:", t_add.get_feature_names_out())

# %%
# Export to ONNX.  Pass ``transformer.onnx_args()`` as the *args* parameter
# to describe one input per column.

onx_add = to_onnx(t_add, t_add.onnx_args())

print("\nONNX graph inputs :", [i.name for i in onx_add.proto.graph.input])
print("ONNX graph outputs:", [o.name for o in onx_add.proto.graph.output])

# %%
# Run with the reference evaluator and verify the result.

ref_add = ExtendedReferenceEvaluator(onx_add)
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
(total,) = ref_add.run(None, {"a": a, "b": b})
print("\ntotal =", total)
assert np.allclose(total, a + b), "Mismatch!"
print("Single output ✓")

# %%
# 2. Multiple output columns
# --------------------------
#
# A single :class:`DataFrameTransformer` can produce any number of output
# columns.  Each SELECT item becomes a separate ONNX output tensor.


def multi_out(df):
    return df.select(
        [
            (df["a"] + df["b"]).alias("total"),
            (df["a"] * 2.0).alias("a_doubled"),
            (df["b"] - df["a"]).alias("diff"),
        ]
    )


t_multi = DataFrameTransformer(multi_out, {"a": np.float32, "b": np.float32})
t_multi.fit()

print("\nOutput columns:", list(t_multi.get_feature_names_out()))

onx_multi = to_onnx(t_multi, t_multi.onnx_args())

ref_multi = ExtendedReferenceEvaluator(onx_multi)
total, a_doubled, diff = ref_multi.run(None, {"a": a, "b": b})

print(f"total     = {total}")
print(f"a_doubled = {a_doubled}")
print(f"diff      = {diff}")

assert np.allclose(total, a + b)
assert np.allclose(a_doubled, a * 2.0)
assert np.allclose(diff, b - a)
print("Multiple outputs ✓")

# %%
# 3. Filter + select
# ------------------
#
# Row filtering (``df.filter``) is fully supported.  Only rows matching the
# condition are passed through to the SELECT expression.


def positive_total(df):
    df = df.filter(df["a"] > 0)
    return df.select([(df["a"] + df["b"]).alias("total")])


t_filter = DataFrameTransformer(positive_total, {"a": np.float32, "b": np.float32})
t_filter.fit()

onx_filter = to_onnx(t_filter, t_filter.onnx_args())

a_mix = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
b_mix = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)

ref_filter = ExtendedReferenceEvaluator(onx_filter)
(total_pos,) = ref_filter.run(None, {"a": a_mix, "b": b_mix})
print(f"\nFiltered total = {total_pos}  (only rows where a > 0)")
assert np.allclose(total_pos, np.array([5.0, 9.0], dtype=np.float32))
print("Filter + select ✓")

# %%
# 4. Pandas ``transform()`` via ONNX
# -----------------------------------
#
# After :meth:`~yobx.sklearn.DataFrameTransformer.fit`, calling
# :meth:`~yobx.sklearn.DataFrameTransformer.transform` on a real
# :class:`pandas.DataFrame` runs the compiled ONNX model with
# :mod:`onnxruntime` internally, so no separate conversion step is needed
# for inference.


df = pd.DataFrame({"a": a.tolist(), "b": b.tolist()})
result_df = t_add.transform(df)
print("\nDataFrame result:")
print(result_df)
assert np.allclose(result_df["total"].values, a + b)
print("Pandas transform ✓")

# %%
# 5. Visualize the ONNX graph
# ---------------------------

plot_dot(onx_multi)
