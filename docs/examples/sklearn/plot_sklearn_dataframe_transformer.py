"""
.. _l-plot-sklearn-dataframe-transformer:

DataFrameTransformer — traceable sklearn transformers for DataFrame inputs
==========================================================================

:class:`~yobx.sklearn.preprocessing.DataFrameTransformer` wraps a
:class:`~yobx.sql.TracedDataFrame`-API function as a scikit-learn transformer
with **automatic ONNX export** — no ``extra_converters`` needed.

The tracing function is written with the
:class:`~yobx.sql.TracedDataFrame` API (``select``, ``filter``, arithmetic
operators, etc.) and is used both as the ONNX blueprint and as the execution
engine for :meth:`~yobx.sklearn.preprocessing.DataFrameTransformer.transform`.

This example covers:

1. **Single-output transformation** — adding two columns.
2. **Multi-output transformation** — producing several output columns.
3. **Row filtering** — combining ``filter`` and ``select``.
4. **transform() method** — executing the transformation on real data.
"""

import numpy as np
import onnxruntime

from yobx.doc import plot_dot
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sklearn import to_onnx
from yobx.sklearn.preprocessing import DataFrameTransformer

# %%
# 1. Single-output transformation
# ---------------------------------
#
# Define a tracing function using the :class:`~yobx.sql.TracedDataFrame` API.
# ``select`` picks (and optionally computes) output columns; ``.alias()``
# assigns the output name.


def _add_columns(df):
    """Return a single output column *total* = a + b."""
    return df.select([(df["a"] + df["b"]).alias("total")])


# Wrap it in a DataFrameTransformer and fit.
t_add = DataFrameTransformer(
    func=_add_columns,
    input_dtypes={"a": np.float32, "b": np.float32},
)
t_add.fit()

# Export to ONNX — onnx_args() supplies the (name, dtype, shape) descriptors.
onx_add = to_onnx(t_add, t_add.onnx_args())

print("ONNX graph inputs :", [i.name for i in onx_add.proto.graph.input])
print("ONNX graph outputs:", onx_add.output_names)

# Verify numerical correctness.
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

sess = onnxruntime.InferenceSession(
    onx_add.proto.SerializeToString(), providers=["CPUExecutionProvider"]
)
(total_onnx,) = sess.run(None, {"a": a, "b": b})

print("\nONNX output:", total_onnx)
np.testing.assert_allclose(total_onnx, a + b, rtol=1e-5)
print("Single-output ✓")

# %%
# 2. Multi-output transformation
# --------------------------------
#
# Return multiple output columns by listing several items in ``select``.


def _multi_out(df):
    """Return *total* = a + b and *a_doubled* = a * 2."""
    return df.select(
        [
            (df["a"] + df["b"]).alias("total"),
            (df["a"] * 2.0).alias("a_doubled"),
        ]
    )


t_multi = DataFrameTransformer(
    func=_multi_out,
    input_dtypes={"a": np.float32, "b": np.float32},
)
t_multi.fit()
onx_multi = to_onnx(t_multi, t_multi.onnx_args())

print("\nMulti-output names:", onx_multi.output_names)
assert onx_multi.output_names == ["total", "a_doubled"]

sess_multi = onnxruntime.InferenceSession(
    onx_multi.proto.SerializeToString(), providers=["CPUExecutionProvider"]
)
total_m, a_doubled_m = sess_multi.run(None, {"a": a, "b": b})
np.testing.assert_allclose(total_m, a + b, rtol=1e-5)
np.testing.assert_allclose(a_doubled_m, a * 2.0, rtol=1e-5)
print("Multi-output ✓")

# %%
# 3. Row filtering
# ------------------
#
# Use ``filter`` to drop rows before applying ``select``.


def _filter_positive(df):
    """Keep only rows where *a* > 0 and return *total* = a + b."""
    df = df.filter(df["a"] > 0)
    return df.select([(df["a"] + df["b"]).alias("total")])


t_filter = DataFrameTransformer(
    func=_filter_positive,
    input_dtypes={"a": np.float32, "b": np.float32},
)
t_filter.fit()
onx_filter = to_onnx(t_filter, t_filter.onnx_args())

a_f = np.array([1.0, -2.0, 3.0], dtype=np.float32)
b_f = np.array([4.0, 5.0, 6.0], dtype=np.float32)

sess_filter = onnxruntime.InferenceSession(
    onx_filter.proto.SerializeToString(), providers=["CPUExecutionProvider"]
)
(total_f,) = sess_filter.run(None, {"a": a_f, "b": b_f})
np.testing.assert_allclose(total_f, np.array([5.0, 9.0], dtype=np.float32), rtol=1e-5)
print("\nFilter + select ✓")

# %%
# 4. transform() — executing on real data
# ------------------------------------------
#
# :meth:`~DataFrameTransformer.transform` compiles the tracing function to ONNX
# and evaluates it via the reference evaluator.  The compiled model is cached
# so subsequent calls are fast.


result_dict = t_add.transform({"a": a, "b": b})
print("\ntransform() with dict input:", result_dict.ravel())
np.testing.assert_allclose(result_dict.ravel(), a + b, rtol=1e-5)

try:
    import pandas as pd

    df_in = pd.DataFrame({"a": a, "b": b})
    result_pd = t_add.transform(df_in)
    print("transform() with DataFrame input:", result_pd.ravel())
    np.testing.assert_allclose(result_pd.ravel(), a + b, rtol=1e-5)
    print("transform() ✓")
except ImportError:
    print("pandas not installed — skipping DataFrame transform test")

# %%
# 5. Visualize the ONNX graph
# ----------------------------
#
# Inspect the flat ONNX graph produced by the single-output transformer:

plot_dot(onx_add)
