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
import struct

import numpy as np
import onnxruntime
from yobx.doc import plot_dot
from yobx.litert import to_onnx


# %%
# Build a minimal TFLite FlatBuffer
# -----------------------------------
#
# We create a tiny model with a single RELU operator
# (input: float32 [1, 4], output: float32 [1, 4]) using a minimal FlatBuffer
# builder.  In a real workflow you would simply pass the path to a ``.tflite``
# file produced by the TFLite converter or downloaded from a model hub.


class _FlatBuilder:
    """Minimal left-to-right FlatBuffer writer with deferred forward refs."""

    def __init__(self):
        self._buf = bytearray()
        self._refs = []
        self._positions = {}

    def pos(self):
        return len(self._buf)

    def align(self, n=4):
        r = self.pos() % n
        if r:
            self._buf.extend(b"\x00" * (n - r))

    def write(self, data):
        p = self.pos()
        self._buf.extend(data)
        return p

    def u8(self, v):   return self.write(struct.pack("<B", v))
    def i8(self, v):   return self.write(struct.pack("<b", v))
    def u16(self, v):  return self.write(struct.pack("<H", v))
    def i32(self, v):  return self.write(struct.pack("<i", v))
    def u32(self, v):  return self.write(struct.pack("<I", v))

    def patch_i32(self, pos, v): struct.pack_into("<i", self._buf, pos, v)
    def patch_u32(self, pos, v): struct.pack_into("<I", self._buf, pos, v)

    def reserve(self, target_id):
        p = self.pos()
        self._refs.append((p, target_id))
        self.write(b"\x00\x00\x00\x00")
        return p

    def mark(self, target_id):
        p = self.pos()
        self._positions[target_id] = p
        return p

    def finalize(self):
        for field_pos, tid in self._refs:
            self.patch_u32(field_pos, self._positions[tid] - field_pos)
        return bytes(self._buf)

    def begin_table(self):
        self.align(4)
        p = self.pos()
        self.i32(0)
        return p

    def end_table_with_vtable(self, soffset_pos, field_positions):
        vtable_start = self.pos()
        n = len(field_positions)
        self.u16(4 + 2 * n)
        self.u16(vtable_start - soffset_pos)
        for fp in field_positions:
            self.u16(fp - soffset_pos if fp is not None else 0)
        self.patch_i32(soffset_pos, soffset_pos - vtable_start)


def _build_relu_tflite():
    b = _FlatBuilder()
    b.reserve("model")      # root offset placeholder
    b.write(b"TFL3")        # file identifier

    # Model
    b.mark("model"); sp = b.begin_table()
    f0 = b.pos(); b.u32(3)
    f1 = b.pos(); b.reserve("opcodes_vec")
    f2 = b.pos(); b.reserve("subgraphs_vec")
    f4 = b.pos(); b.reserve("buffers_vec")
    b.end_table_with_vtable(sp, [f0, f1, f2, None, f4])

    # OperatorCode: RELU = 19
    b.align(4); b.mark("opcodes_vec"); b.u32(1); b.reserve("opcode0")
    b.align(4); b.mark("opcode0"); sp = b.begin_table()
    f0 = b.pos(); b.i8(19); b.align(4)
    f4 = b.pos(); b.i32(19)
    b.end_table_with_vtable(sp, [f0, None, None, None, f4])

    # SubGraph
    b.align(4); b.mark("subgraphs_vec"); b.u32(1); b.reserve("sg0")
    b.align(4); b.mark("sg0"); sp = b.begin_table()
    f0 = b.pos(); b.reserve("tensors_vec")
    f1 = b.pos(); b.reserve("sg_in")
    f2 = b.pos(); b.reserve("sg_out")
    f3 = b.pos(); b.reserve("ops_vec")
    b.end_table_with_vtable(sp, [f0, f1, f2, f3])

    # Tensors
    b.align(4); b.mark("tensors_vec"); b.u32(2); b.reserve("t0"); b.reserve("t1")

    for tname_id, name, buf_idx in [("t0", "input", 1), ("t1", "relu", 2)]:
        b.align(4); b.mark(tname_id); sp = b.begin_table()
        fa = b.pos(); b.reserve(tname_id + "_shape")
        fb = b.pos(); b.i8(0); b.align(4)
        fc = b.pos(); b.u32(buf_idx)
        fd = b.pos(); b.reserve(tname_id + "_name")
        b.end_table_with_vtable(sp, [fa, fb, fc, fd])
        b.align(4); b.mark(tname_id + "_shape"); b.u32(2); b.i32(1); b.i32(4)
        enc = name.encode("utf-8")
        b.align(4); b.mark(tname_id + "_name"); b.u32(len(enc)); b.write(enc + b"\x00")

    b.align(4); b.mark("sg_in"); b.u32(1); b.i32(0)
    b.align(4); b.mark("sg_out"); b.u32(1); b.i32(1)

    # Operator
    b.align(4); b.mark("ops_vec"); b.u32(1); b.reserve("op0")
    b.align(4); b.mark("op0"); sp = b.begin_table()
    f0 = b.pos(); b.u32(0)
    f1 = b.pos(); b.reserve("op_in")
    f2 = b.pos(); b.reserve("op_out")
    b.end_table_with_vtable(sp, [f0, f1, f2])
    b.align(4); b.mark("op_in"); b.u32(1); b.i32(0)
    b.align(4); b.mark("op_out"); b.u32(1); b.i32(1)

    # Buffers (3 empty)
    b.align(4); b.mark("buffers_vec"); b.u32(3)
    b.reserve("buf0"); b.reserve("buf1"); b.reserve("buf2")
    for bid in ("buf0", "buf1", "buf2"):
        b.align(4); b.mark(bid); sp = b.begin_table()
        b.end_table_with_vtable(sp, [])

    return b.finalize()


model_bytes = _build_relu_tflite()
print(f"Minimal TFLite model size: {len(model_bytes)} bytes")

# %%
# Inspect the parsed model
# -------------------------
#
# :func:`~yobx.litert.litert_helper.parse_tflite_model` returns a plain
# Python :class:`~yobx.litert.litert_helper.TFLiteModel` object.

from yobx.litert.litert_helper import parse_tflite_model

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

sess = onnxruntime.InferenceSession(
    onx.SerializeToString(), providers=["CPUExecutionProvider"]
)
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

from yobx.litert.litert_helper import BuiltinOperator


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
