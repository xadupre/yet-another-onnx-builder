"""
.. _l-plot-polars-schema-to-onnx:

Converting a Polars DataFrame schema to ONNX
=============================================

:func:`yobx.polars.to_onnx` converts a :class:`polars.DataFrame` (or its
:class:`polars.Schema`) into an :class:`onnx.ModelProto`.

Each column becomes one ONNX graph input tensor typed according to the
column's Polars dtype; all inputs are forwarded to identical outputs through
``Identity`` nodes.  The primary use-case is to capture the *schema contract*
of a DataFrame as a portable, runtime-agnostic ONNX artifact that can be:

* inspected or visualised with standard ONNX tooling,
* used as a type-checking stub in an inference pipeline,
* extended with additional ONNX nodes to form a full preprocessing graph.

The workflow is:

1. **Create** (or load) a :class:`polars.DataFrame`.
2. Call :func:`yobx.polars.to_onnx` to obtain the schema model.
3. Optionally **run** the model with :epkg:`onnxruntime` to verify that
   numeric columns pass through unchanged.
4. **Visualise** the graph.
"""

# %%
import numpy as np
import onnx
import polars as pl
import onnxruntime
from yobx.doc import plot_dot
from yobx.polars import to_onnx

# %%
# 1. Build a sample DataFrame
# ----------------------------
#
# We create a small DataFrame with several column types to exercise the
# dtype-mapping logic.

df = pl.DataFrame(
    {
        "user_id": pl.Series([1, 2, 3], dtype=pl.Int32),
        "age": pl.Series([25, 30, 22], dtype=pl.Int64),
        "score": pl.Series([0.91, 0.75, 0.88], dtype=pl.Float32),
        "revenue": pl.Series([120.5, 200.0, 95.3], dtype=pl.Float64),
        "active": pl.Series([True, False, True], dtype=pl.Boolean),
        "name": pl.Series(["Alice", "Bob", "Carol"], dtype=pl.String),
    }
)
print("DataFrame schema:")
print(df.schema)

# %%
# 2. Convert to ONNX
# -------------------
#
# By default a symbolic batch dimension ``"N"`` is added as the first axis
# of every input tensor.

onx = to_onnx(df)

print("\nOpset                  :", onx.opset_import[0].version)
print("Number of graph nodes  :", len(onx.graph.node))
print("Node op-types          :", [n.op_type for n in onx.graph.node])
print(
    "Graph inputs           :",
    [(inp.name, inp.type.tensor_type.elem_type) for inp in onx.graph.input],
)

# %%
# 3. Inspect the ONNX graph
# --------------------------
#
# The ONNX ``TensorProto`` element-type integers map to Polars dtypes as
# follows (non-exhaustive):
#
# .. list-table::
#    :header-rows: 1
#    :widths: 30 30 20
#
#    * - Polars dtype
#      - ONNX element type
#      - TensorProto int
#    * - ``Float32``
#      - ``FLOAT``
#      - 1
#    * - ``Float64``
#      - ``DOUBLE``
#      - 11
#    * - ``Int8`` / ``Int16`` / ``Int32`` / ``Int64``
#      - ``INT8`` … ``INT64``
#      - 3–7
#    * - ``UInt8`` … ``UInt64``
#      - ``UINT8`` … ``UINT64``
#      - 2, 4, 12, 13
#    * - ``Boolean``
#      - ``BOOL``
#      - 9
#    * - ``String`` / ``Utf8``
#      - ``STRING``
#      - 8
#    * - ``Date``
#      - ``INT32``
#      - 6
#    * - ``Datetime`` / ``Duration``
#      - ``INT64``
#      - 7


for inp in onx.graph.input:
    elem_name = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
    batch_dim = inp.type.tensor_type.shape.dim[0].dim_param
    print(f"  {inp.name:12s} → ONNX {elem_name:8s}  batch_dim={batch_dim!r}")

# %%
# 4. Run with onnxruntime (numeric columns only)
# -----------------------------------------------
#
# ``Identity`` nodes are pass-through, so numeric outputs equal the inputs.
# We verify this for the ``score`` column.

feeds = {"score": np.array([0.91, 0.75, 0.88], dtype=np.float32)}

# Use a fixed batch size for the runtime session
onx_fixed = to_onnx(df, batch_dim=3)
sess = onnxruntime.InferenceSession(
    onx_fixed.SerializeToString(), providers=["CPUExecutionProvider"]
)

# Provide all required numeric inputs (STRING inputs are skipped by ORT demo)
all_feeds = {
    "user_id": np.array([1, 2, 3], dtype=np.int32),
    "age": np.array([25, 30, 22], dtype=np.int64),
    "score": np.array([0.91, 0.75, 0.88], dtype=np.float32),
    "revenue": np.array([120.5, 200.0, 95.3], dtype=np.float64),
    "active": np.array([True, False, True], dtype=bool),
    "name": np.array(["Alice", "Bob", "Carol"]),
}
results = sess.run([f"{k}_out" for k in all_feeds], all_feeds)
for name, result in zip(all_feeds, results):
    np.testing.assert_array_equal(all_feeds[name], result)
    print(f"  {name}_out matches input ✓")

# %%
# 5. Convert from a Schema only (no data required)
# --------------------------------------------------
#
# When you only have the schema (e.g. from reading a Parquet file header),
# pass it directly.

schema = pl.Schema(
    {"product_id": pl.UInt32, "price": pl.Float64, "in_stock": pl.Boolean, "category": pl.String}
)
onx_schema = to_onnx(schema, batch_dim="batch")
print("\nSchema-only model inputs:")
for inp in onx_schema.graph.input:
    elem_name = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
    print(f"  {inp.name:12s} → {elem_name}")

# %%
# 6. Visualise the ONNX graph
# ----------------------------

plot_dot(onx_schema)
