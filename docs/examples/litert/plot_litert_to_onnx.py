"""
.. _l-plot-litert-to-onnx:

Converting a TFLite/LiteRT model to ONNX
=========================================

:func:`yobx.litert.to_onnx` converts a :epkg:`TFLite`/:epkg:`LiteRT`
``.tflite`` model into an :class:`onnx.ModelProto` that can be executed with
any ONNX-compatible runtime.

The converter:

1. Parses the binary FlatBuffer that every ``.tflite`` file uses with a
   **pure-Python** parser — no ``tensorflow`` or ``ai_edge_litert`` package
   is required.
2. Walks every operator in the requested subgraph (default: subgraph 0).
3. Converts each operator to its ONNX equivalent via a registry of op-level
   converters.

This example demonstrates the converter with a hand-crafted minimal TFLite
model so that no external TFLite dependencies are needed to run the gallery.
"""

# %%
import numpy as np
import onnxruntime
from yobx.doc import plot_dot
from yobx.litert import to_onnx
from yobx.litert.litert_helper import (
    _make_sample_tflite_model,
    parse_tflite_model,
    BuiltinOperator,
)

# %%
# Build a minimal TFLite FlatBuffer
# -----------------------------------
#
# :func:`~yobx.litert.litert_helper._make_sample_tflite_model` returns the
# bytes of a minimal TFLite model with a single RELU operator
# (input: float32 [1, 4], output: float32 [1, 4]).  In a real workflow you
# would pass the path to a ``.tflite`` file produced by the TFLite converter
# or downloaded from a model hub.

model_bytes = _make_sample_tflite_model()
print(f"Minimal TFLite model size: {len(model_bytes)} bytes")

# %%
# Inspect the parsed model
# -------------------------
#
# :func:`~yobx.litert.litert_helper.parse_tflite_model` returns a plain
# Python :class:`~yobx.litert.litert_helper.TFLiteModel` object.

tflite_model = parse_tflite_model(model_bytes)
sg = tflite_model.subgraphs[0]

print(f"TFLite model version : {tflite_model.version}")
print(f"Number of subgraphs  : {len(tflite_model.subgraphs)}")
print(f"Number of tensors    : {len(sg.tensors)}")
print(f"Number of operators  : {len(sg.operators)}")
for i, op in enumerate(sg.operators):
    in_names = [sg.tensors[j].name for j in op.inputs if j >= 0]
    out_names = [sg.tensors[j].name for j in op.outputs if j >= 0]
    print(f"  Op {i}: {op.name}  inputs={in_names}  outputs={out_names}")

# %%
# Convert to ONNX
# ----------------
#
# :func:`yobx.litert.to_onnx` accepts the raw bytes (or a file path).
# We provide a dummy NumPy input so the converter can infer the input dtype
# and mark axis 0 as dynamic.

X = np.random.default_rng(0).standard_normal((1, 4)).astype(np.float32)

onx = to_onnx(model_bytes, (X,), input_names=["input"])

print(f"\nONNX opset version : {onx.opset_import[0].version}")
print(f"Number of nodes    : {len(onx.graph.node)}")
print(f"Node op-types      : {[n.op_type for n in onx.graph.node]}")
print(f"Graph input name   : {onx.graph.input[0].name}")
print(f"Graph output name  : {onx.graph.output[0].name}")

# %%
# Run and verify
# ---------------
#
# The ONNX model produces the same output as a manual ``np.maximum(X, 0)``.

sess = onnxruntime.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
(result,) = sess.run(None, {onx.graph.input[0].name: X})

expected = np.maximum(X, 0.0)
print(f"\nInput  : {X[0]}")
print(f"Expected (ReLU): {expected[0]}")
print(f"ONNX output    : {result[0]}")

assert np.allclose(expected, result, atol=1e-6), "Mismatch!"
print("Outputs match ✓")

# %%
# Dynamic batch dimension
# -------------------------
#
# By default :func:`to_onnx` marks axis 0 as dynamic.

batch_dim = onx.graph.input[0].type.tensor_type.shape.dim[0]
print(f"\nBatch dim param : {batch_dim.dim_param!r}")
assert batch_dim.dim_param, "Expected a dynamic batch dimension"

# Verify the converted model works for different batch sizes.
for n in (1, 3, 7):
    X_batch = np.random.default_rng(n).standard_normal((n, 4)).astype(np.float32)
    (out,) = sess.run(None, {onx.graph.input[0].name: X_batch})
    assert np.allclose(np.maximum(X_batch, 0), out, atol=1e-6), f"Mismatch for batch={n}"

print("Dynamic batch verified for batch sizes 1, 3, 7 ✓")

# %%
# Custom op converter
# ---------------------
#
# Use ``extra_converters`` to override or extend the built-in converters.
# Here we replace ``RELU`` with ``Clip(0, 6)`` (i.e. ``Relu6``).


def custom_relu6(g, sts, outputs, op):
    """Replace RELU with Clip(0, 6)."""
    return g.op.Clip(
        op.inputs[0],
        np.array(0.0, dtype=np.float32),
        np.array(6.0, dtype=np.float32),
        outputs=outputs,
        name="relu6",
    )


onx_custom = to_onnx(
    model_bytes,
    (X,),
    input_names=["input"],
    extra_converters={BuiltinOperator.RELU: custom_relu6},
)

custom_op_types = [n.op_type for n in onx_custom.graph.node]
print(f"\nOp-types with custom converter: {custom_op_types}")
assert "Clip" in custom_op_types, "Expected Clip node from custom converter"
assert "Relu" not in custom_op_types, "Relu should have been replaced"
print("Custom converter verified ✓")

# %%
# Visualise the graph
# ---------------------

plot_dot(onx)
