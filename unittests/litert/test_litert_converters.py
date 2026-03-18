"""
Unit tests for :mod:`yobx.litert` — the LiteRT/TFLite → ONNX converter.

Tests are structured in three groups:

1. Helper utilities (no external deps needed).
2. Converter unit tests — exercises individual op converters via
   :class:`~yobx.xbuilder.GraphBuilder` directly, without a real TFLite model.
3. Integration tests — exercises :func:`~yobx.litert.to_onnx` end-to-end with
   a hand-crafted TFLite FlatBuffer.
"""

import struct
import unittest

import numpy as np
from onnx import TensorProto

from yobx.ext_test_case import ExtTestCase
from yobx.xbuilder import GraphBuilder, OptimizationOptions


# ---------------------------------------------------------------------------
# Minimal TFLite FlatBuffer builder for self-contained tests
# ---------------------------------------------------------------------------


class _MinimalFlatBuilder:
    """Writes a FlatBuffer left-to-right with deferred (forward) references.

    In FlatBuffer, all uoffset references must be *positive* (they point to
    higher addresses than the reference field itself).  This builder therefore
    writes referenced objects *after* the object that contains the reference.
    Placeholders are recorded and patched at the end.

    Vtables are written immediately after their table's data fields; the
    soffset stored at the table start is thus *negative* (pointing forward).
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._refs: list = []        # (field_pos, target_id)
        self._positions: dict = {}   # target_id → absolute position

    def pos(self) -> int:
        return len(self._buf)

    def align(self, n: int = 4) -> None:
        r = self.pos() % n
        if r:
            self._buf.extend(b"\x00" * (n - r))

    def write(self, data: bytes) -> int:
        p = self.pos()
        self._buf.extend(data)
        return p

    # scalar writers
    def u8(self, v: int) -> int:   return self.write(struct.pack("<B", v))
    def i8(self, v: int) -> int:   return self.write(struct.pack("<b", v))
    def u16(self, v: int) -> int:  return self.write(struct.pack("<H", v))
    def i32(self, v: int) -> int:  return self.write(struct.pack("<i", v))
    def u32(self, v: int) -> int:  return self.write(struct.pack("<I", v))

    def patch_i32(self, pos: int, v: int) -> None:
        struct.pack_into("<i", self._buf, pos, v)

    def patch_u32(self, pos: int, v: int) -> None:
        struct.pack_into("<I", self._buf, pos, v)

    def reserve(self, target_id: str) -> int:
        """Write a placeholder uint32 forward reference and record it."""
        p = self.pos()
        self._refs.append((p, target_id))
        self.write(b"\x00\x00\x00\x00")
        return p

    def mark(self, target_id: str) -> int:
        """Record the current position as the target for *target_id*."""
        p = self.pos()
        self._positions[target_id] = p
        return p

    def finalize(self) -> bytes:
        """Patch all forward references and return the completed buffer."""
        for field_pos, target_id in self._refs:
            target = self._positions[target_id]
            offset = target - field_pos
            assert offset >= 0, (
                f"Backward reference from pos {field_pos} to target {target_id!r} "
                f"at pos {target}"
            )
            self.patch_u32(field_pos, offset)
        return bytes(self._buf)

    # --- FlatBuffer table helpers ---

    def begin_table(self) -> int:
        """Start a table: write a placeholder soffset and return its position."""
        self.align(4)
        p = self.pos()
        self.i32(0)  # placeholder soffset
        return p

    def end_table_with_vtable(
        self,
        soffset_pos: int,
        field_offsets: list,
    ) -> None:
        """Write the vtable for a table and patch its soffset.

        :param soffset_pos: position of the soffset field (= table start).
        :param field_offsets: list of per-field absolute positions;
            *None* entries produce a vtable slot of 0 (absent field).
        """
        vtable_start = self.pos()
        n_fields = len(field_offsets)
        vtable_size = 4 + 2 * n_fields
        obj_size = vtable_start - soffset_pos
        self.u16(vtable_size)
        self.u16(obj_size)
        for fp in field_offsets:
            self.u16(fp - soffset_pos if fp is not None else 0)
        # soffset = table_start - vtable_start (negative)
        self.patch_i32(soffset_pos, soffset_pos - vtable_start)

    # --- FlatBuffer scalar vector helpers ---

    def i32_vec(self, values) -> int:
        """Write a vector of int32 and return its position."""
        self.align(4)
        p = self.mark(f"vec_{self.pos()}")
        self.u32(len(values))
        for v in values:
            self.i32(int(v))
        return p

    def fbs_string(self, s: str) -> int:
        """Write a FlatBuffer string and return its position."""
        self.align(4)
        enc = s.encode("utf-8")
        p = self.pos()
        self.u32(len(enc))
        self.write(enc + b"\x00")
        return p

    def begin_table_vec(self, target_ids: list) -> int:
        """Write a vector of table references and return its position.

        The actual table positions are patched later via :meth:`finalize`.
        """
        self.align(4)
        p = self.pos()
        self.u32(len(target_ids))
        for tid in target_ids:
            self.reserve(tid)
        return p


def _make_relu_tflite_model() -> bytes:
    """Build a minimal valid TFLite FlatBuffer with a single RELU operator.

    Layout::

        Input  tensor: "x",    float32, shape [1, 4]
        Output tensor: "relu", float32, shape [1, 4]
        Operator:      RELU (builtin_code 19)
    """
    b = _MinimalFlatBuilder()

    # [0-3] root offset placeholder
    b.reserve("model")

    # [4-7] TFLite file identifier
    b.write(b"TFL3")

    # ── MODEL TABLE ────────────────────────────────────────────────────────
    b.mark("model")
    soffset_model = b.begin_table()
    f0_model = b.pos(); b.u32(3)                  # version = 3
    f1_model = b.pos(); b.reserve("opcodes_vec")   # operator_codes
    f2_model = b.pos(); b.reserve("subgraphs_vec") # subgraphs
    f3_model = None                                 # description (absent)
    f4_model = b.pos(); b.reserve("buffers_vec")   # buffers
    b.end_table_with_vtable(soffset_model, [f0_model, f1_model, f2_model, None, f4_model])

    # ── OPERATOR CODES VECTOR ──────────────────────────────────────────────
    b.align(4); b.mark("opcodes_vec")
    b.u32(1)
    b.reserve("opcode0")

    # ── OPCODE 0 TABLE: RELU = 19 ─────────────────────────────────────────
    b.align(4); b.mark("opcode0")
    soffset_oc = b.begin_table()
    f0_oc = b.pos(); b.i8(19)    # builtin_code (int8)
    b.align(4)
    f4_oc = b.pos(); b.i32(19)  # extended_builtin_code (int32)
    b.end_table_with_vtable(soffset_oc, [f0_oc, None, None, None, f4_oc])

    # ── SUBGRAPHS VECTOR ──────────────────────────────────────────────────
    b.align(4); b.mark("subgraphs_vec")
    b.u32(1)
    b.reserve("subgraph0")

    # ── SUBGRAPH 0 TABLE ──────────────────────────────────────────────────
    b.align(4); b.mark("subgraph0")
    soffset_sg = b.begin_table()
    f0_sg = b.pos(); b.reserve("tensors_vec")    # tensors
    f1_sg = b.pos(); b.reserve("sg_inputs")      # inputs
    f2_sg = b.pos(); b.reserve("sg_outputs")     # outputs
    f3_sg = b.pos(); b.reserve("operators_vec")  # operators
    b.end_table_with_vtable(soffset_sg, [f0_sg, f1_sg, f2_sg, f3_sg])

    # ── TENSORS VECTOR ────────────────────────────────────────────────────
    b.align(4); b.mark("tensors_vec")
    b.u32(2)
    b.reserve("tensor0")
    b.reserve("tensor1")

    # ── TENSOR 0: "x", float32, shape [1,4], buffer=1 ────────────────────
    b.align(4); b.mark("tensor0")
    soffset_t0 = b.begin_table()
    f0_t0 = b.pos(); b.reserve("t0_shape")   # shape
    f1_t0 = b.pos(); b.i8(0)                 # type = FLOAT32
    b.align(4)
    f2_t0 = b.pos(); b.u32(1)                # buffer = 1
    f3_t0 = b.pos(); b.reserve("t0_name")    # name
    b.end_table_with_vtable(soffset_t0, [f0_t0, f1_t0, f2_t0, f3_t0])

    # shape [1, 4]
    b.align(4); b.mark("t0_shape")
    b.u32(2); b.i32(1); b.i32(4)

    # name "x"
    b.align(4); b.mark("t0_name")
    b.u32(1); b.write(b"x\x00")

    # ── TENSOR 1: "relu", float32, shape [1,4], buffer=2 ─────────────────
    b.align(4); b.mark("tensor1")
    soffset_t1 = b.begin_table()
    f0_t1 = b.pos(); b.reserve("t1_shape")
    f1_t1 = b.pos(); b.i8(0)
    b.align(4)
    f2_t1 = b.pos(); b.u32(2)               # buffer = 2
    f3_t1 = b.pos(); b.reserve("t1_name")
    b.end_table_with_vtable(soffset_t1, [f0_t1, f1_t1, f2_t1, f3_t1])

    b.align(4); b.mark("t1_shape")
    b.u32(2); b.i32(1); b.i32(4)

    b.align(4); b.mark("t1_name")
    b.u32(4); b.write(b"relu\x00")

    # ── SUBGRAPH INPUT/OUTPUT VECTORS ─────────────────────────────────────
    b.align(4); b.mark("sg_inputs")
    b.u32(1); b.i32(0)

    b.align(4); b.mark("sg_outputs")
    b.u32(1); b.i32(1)

    # ── OPERATORS VECTOR ──────────────────────────────────────────────────
    b.align(4); b.mark("operators_vec")
    b.u32(1)
    b.reserve("operator0")

    # ── OPERATOR 0 TABLE ──────────────────────────────────────────────────
    b.align(4); b.mark("operator0")
    soffset_op = b.begin_table()
    f0_op = b.pos(); b.u32(0)               # opcode_index = 0
    f1_op = b.pos(); b.reserve("op_inputs")
    f2_op = b.pos(); b.reserve("op_outputs")
    b.end_table_with_vtable(soffset_op, [f0_op, f1_op, f2_op])

    b.align(4); b.mark("op_inputs")
    b.u32(1); b.i32(0)

    b.align(4); b.mark("op_outputs")
    b.u32(1); b.i32(1)

    # ── BUFFERS VECTOR ────────────────────────────────────────────────────
    b.align(4); b.mark("buffers_vec")
    b.u32(3)
    b.reserve("buf0"); b.reserve("buf1"); b.reserve("buf2")

    # Empty buffer tables
    for bid in ("buf0", "buf1", "buf2"):
        b.align(4); b.mark(bid)
        soffset_buf = b.begin_table()
        b.end_table_with_vtable(soffset_buf, [])

    return b.finalize()


# ---------------------------------------------------------------------------
# 1. Helper utilities — no external deps
# ---------------------------------------------------------------------------


class TestLiteRTHelpers(ExtTestCase):
    def test_dtype_mapping_float32(self):
        from yobx.litert.litert_helper import litert_dtype_to_np_dtype

        self.assertEqual(litert_dtype_to_np_dtype(0), np.dtype("float32"))

    def test_dtype_mapping_int8(self):
        from yobx.litert.litert_helper import litert_dtype_to_np_dtype

        self.assertEqual(litert_dtype_to_np_dtype(9), np.dtype("int8"))

    def test_dtype_mapping_uint8(self):
        from yobx.litert.litert_helper import litert_dtype_to_np_dtype

        self.assertEqual(litert_dtype_to_np_dtype(3), np.dtype("uint8"))

    def test_dtype_mapping_invalid(self):
        from yobx.litert.litert_helper import litert_dtype_to_np_dtype

        with self.assertRaises(ValueError):
            litert_dtype_to_np_dtype(99)

    def test_builtin_op_name(self):
        from yobx.litert.litert_helper import BuiltinOperator, builtin_op_name

        self.assertEqual(builtin_op_name(BuiltinOperator.RELU), "RELU")
        self.assertEqual(builtin_op_name(BuiltinOperator.FULLY_CONNECTED), "FULLY_CONNECTED")
        self.assertIn("UNKNOWN", builtin_op_name(9999))

    def test_parse_minimal_model_structure(self):
        """parse_tflite_model() should correctly decode a hand-crafted model."""
        from yobx.litert.litert_helper import BuiltinOperator, parse_tflite_model

        model_bytes = _make_relu_tflite_model()
        model = parse_tflite_model(model_bytes)

        self.assertEqual(model.version, 3)
        self.assertEqual(len(model.subgraphs), 1)

        sg = model.subgraphs[0]
        self.assertEqual(len(sg.tensors), 2)
        self.assertEqual(len(sg.inputs), 1)
        self.assertEqual(len(sg.outputs), 1)
        self.assertEqual(len(sg.operators), 1)

        op = sg.operators[0]
        self.assertEqual(op.opcode, BuiltinOperator.RELU)
        self.assertEqual(op.inputs, (0,))
        self.assertEqual(op.outputs, (1,))

    def test_parse_tensor_shapes(self):
        """Parsed tensors carry correct shape and dtype."""
        from yobx.litert.litert_helper import parse_tflite_model

        model = parse_tflite_model(_make_relu_tflite_model())
        t0 = model.subgraphs[0].tensors[0]
        self.assertEqual(t0.shape, (1, 4))
        self.assertEqual(t0.dtype, 0)  # FLOAT32

    def test_register_converters_idempotent(self):
        """register_litert_converters() is idempotent."""
        from yobx.litert import register_litert_converters

        register_litert_converters()
        register_litert_converters()
        from yobx.litert.register import LITERT_OP_CONVERTERS

        self.assertGreater(len(LITERT_OP_CONVERTERS), 0)


# ---------------------------------------------------------------------------
# 2. Converter unit tests
# ---------------------------------------------------------------------------


class TestLiteRTConverterUnits(ExtTestCase):
    """Exercise individual op converters via GraphBuilder, no TFLite file."""

    def _gb(self) -> GraphBuilder:
        """Return a GraphBuilder with pattern optimisation disabled."""
        return GraphBuilder({"": 18}, optimization_options=OptimizationOptions(patterns=None))

    def _run(self, g: GraphBuilder, feeds: dict) -> list:
        from onnxruntime import InferenceSession

        onx = g.to_onnx()
        sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        return sess.run(None, feeds)

    def _proxy(self, op_litert, input_names, output_names):
        from yobx.litert.convert import _OpProxy

        return _OpProxy(op_litert, list(input_names), list(output_names))

    def _op(self, opcode, inputs=(0,), outputs=(0,), options=None):
        from yobx.litert.litert_helper import TFLiteOperator

        return TFLiteOperator(
            opcode=opcode,
            custom_code="",
            inputs=inputs,
            outputs=outputs,
            builtin_options=options or {},
        )

    def test_relu(self):
        from yobx.litert.litert_helper import BuiltinOperator
        from yobx.litert.ops.activations import convert_relu

        g = self._gb()
        g.make_tensor_input("X", TensorProto.FLOAT, (3,))
        proxy = self._proxy(self._op(BuiltinOperator.RELU), ["X"], ["Y"])
        convert_relu(g, {}, ["Y"], proxy)
        g.make_tensor_output("Y", indexed=False, allow_untyped_output=True)

        X = np.array([-1.0, 0.5, 2.0], dtype=np.float32)
        (result,) = self._run(g, {"X": X})
        self.assertEqualArray(np.maximum(X, 0), result)

    def test_tanh(self):
        from yobx.litert.litert_helper import BuiltinOperator
        from yobx.litert.ops.activations import convert_tanh

        g = self._gb()
        g.make_tensor_input("X", TensorProto.FLOAT, (3,))
        proxy = self._proxy(self._op(BuiltinOperator.TANH), ["X"], ["Y"])
        convert_tanh(g, {}, ["Y"], proxy)
        g.make_tensor_output("Y", indexed=False, allow_untyped_output=True)

        X = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        (result,) = self._run(g, {"X": X})
        self.assertEqualArray(np.tanh(X), result, atol=1e-6)

    def test_softmax(self):
        from yobx.litert.litert_helper import BuiltinOperator
        from yobx.litert.ops.activations import convert_softmax

        g = self._gb()
        g.make_tensor_input("X", TensorProto.FLOAT, (2, 4))
        proxy = self._proxy(self._op(BuiltinOperator.SOFTMAX), ["X"], ["Y"])
        convert_softmax(g, {}, ["Y"], proxy)
        g.make_tensor_output("Y", indexed=False, allow_untyped_output=True)

        X = np.random.default_rng(0).standard_normal((2, 4)).astype(np.float32)
        (result,) = self._run(g, {"X": X})
        e = np.exp(X - X.max(axis=-1, keepdims=True))
        expected = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_add(self):
        from yobx.litert.litert_helper import BuiltinOperator
        from yobx.litert.ops.elementwise import convert_add

        g = self._gb()
        g.make_tensor_input("A", TensorProto.FLOAT, (3,))
        g.make_tensor_input("B", TensorProto.FLOAT, (3,))
        proxy = self._proxy(
            self._op(BuiltinOperator.ADD, inputs=(0, 1), outputs=(2,)), ["A", "B"], ["C"]
        )
        convert_add(g, {}, ["C"], proxy)
        g.make_tensor_output("C", indexed=False, allow_untyped_output=True)

        A = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        B = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (result,) = self._run(g, {"A": A, "B": B})
        self.assertEqualArray(A + B, result)

    def test_mul(self):
        from yobx.litert.litert_helper import BuiltinOperator
        from yobx.litert.ops.elementwise import convert_mul

        g = self._gb()
        g.make_tensor_input("A", TensorProto.FLOAT, (3,))
        g.make_tensor_input("B", TensorProto.FLOAT, (3,))
        proxy = self._proxy(
            self._op(BuiltinOperator.MUL, inputs=(0, 1), outputs=(0,)), ["A", "B"], ["C"]
        )
        convert_mul(g, {}, ["C"], proxy)
        g.make_tensor_output("C", indexed=False, allow_untyped_output=True)

        A = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        B = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        (result,) = self._run(g, {"A": A, "B": B})
        self.assertEqualArray(A * B, result)

    def test_abs(self):
        from yobx.litert.litert_helper import BuiltinOperator
        from yobx.litert.ops.elementwise import convert_abs

        g = self._gb()
        g.make_tensor_input("X", TensorProto.FLOAT, (4,))
        proxy = self._proxy(self._op(BuiltinOperator.ABS), ["X"], ["Y"])
        convert_abs(g, {}, ["Y"], proxy)
        g.make_tensor_output("Y", indexed=False, allow_untyped_output=True)

        X = np.array([-2.0, -1.0, 0.0, 3.0], dtype=np.float32)
        (result,) = self._run(g, {"X": X})
        self.assertEqualArray(np.abs(X), result)

    def test_sqrt(self):
        from yobx.litert.litert_helper import BuiltinOperator
        from yobx.litert.ops.elementwise import convert_sqrt

        g = self._gb()
        g.make_tensor_input("X", TensorProto.FLOAT, (3,))
        proxy = self._proxy(self._op(BuiltinOperator.SQRT), ["X"], ["Y"])
        convert_sqrt(g, {}, ["Y"], proxy)
        g.make_tensor_output("Y", indexed=False, allow_untyped_output=True)

        X = np.array([1.0, 4.0, 9.0], dtype=np.float32)
        (result,) = self._run(g, {"X": X})
        self.assertEqualArray(np.sqrt(X), result, atol=1e-6)

    def test_leaky_relu(self):
        from yobx.litert.litert_helper import BuiltinOperator
        from yobx.litert.ops.activations import convert_leaky_relu

        alpha = 0.1
        g = self._gb()
        g.make_tensor_input("X", TensorProto.FLOAT, (3,))
        proxy = self._proxy(
            self._op(BuiltinOperator.LEAKY_RELU, options={"alpha": alpha}), ["X"], ["Y"]
        )
        convert_leaky_relu(g, {}, ["Y"], proxy)
        g.make_tensor_output("Y", indexed=False, allow_untyped_output=True)

        X = np.array([-1.0, 0.5, 2.0], dtype=np.float32)
        (result,) = self._run(g, {"X": X})
        expected = np.where(X >= 0, X, alpha * X).astype(np.float32)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_reshape(self):
        from yobx.litert.litert_helper import BuiltinOperator
        from yobx.litert.ops.reshape_ops import convert_reshape

        g = self._gb()
        g.make_tensor_input("X", TensorProto.FLOAT, (2, 3))
        g.make_tensor_input("shape", TensorProto.INT32, (2,))
        proxy = self._proxy(
            self._op(BuiltinOperator.RESHAPE, inputs=(0, 1), outputs=(0,)),
            ["X", "shape"],
            ["Y"],
        )
        convert_reshape(g, {}, ["Y"], proxy)
        g.make_tensor_output("Y", indexed=False, allow_untyped_output=True)

        X = np.arange(6, dtype=np.float32).reshape(2, 3)
        shape = np.array([3, 2], dtype=np.int32)
        (result,) = self._run(g, {"X": X, "shape": shape})
        self.assertEqualArray(X.reshape(3, 2), result)


# ---------------------------------------------------------------------------
# 3. Integration tests — end-to-end with the hand-crafted TFLite model
# ---------------------------------------------------------------------------


class TestLiteRTEndToEnd(ExtTestCase):
    def test_to_onnx_minimal_relu(self):
        """to_onnx() on the hand-crafted RELU model produces a valid ONNX graph."""
        from yobx.litert import to_onnx

        model_bytes = _make_relu_tflite_model()
        X = np.zeros((1, 4), dtype=np.float32)
        onx = to_onnx(model_bytes, (X,), input_names=["x"])

        # Graph must contain a Relu node.
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Relu", op_types)

        # Output is numerically correct.
        from onnxruntime import InferenceSession

        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        feeds = {onx.graph.input[0].name: X}
        (result,) = sess.run(None, feeds)
        self.assertEqualArray(np.maximum(X, 0), result)

    def test_to_onnx_input_names_mismatch_raises(self):
        """to_onnx() raises ValueError when input_names length mismatches."""
        from yobx.litert import to_onnx

        model_bytes = _make_relu_tflite_model()
        X = np.zeros((1, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            to_onnx(model_bytes, (X,), input_names=["a", "b"])

    def test_to_onnx_subgraph_out_of_range_raises(self):
        """to_onnx() raises ValueError for an invalid subgraph_index."""
        from yobx.litert import to_onnx

        model_bytes = _make_relu_tflite_model()
        X = np.zeros((1, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            to_onnx(model_bytes, (X,), subgraph_index=5)

    def test_to_onnx_from_file(self, tmp_path=None):
        """to_onnx() accepts a file path as well as raw bytes."""
        import os
        import tempfile

        from yobx.litert import to_onnx

        model_bytes = _make_relu_tflite_model()
        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as fh:
            fh.write(model_bytes)
            path = fh.name
        try:
            X = np.zeros((1, 4), dtype=np.float32)
            onx = to_onnx(path, (X,), input_names=["x"])
            op_types = [n.op_type for n in onx.graph.node]
            self.assertIn("Relu", op_types)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
