"""
Unit tests for :mod:`yobx.litert` — the LiteRT/TFLite → ONNX converter.

Tests are structured in three groups:

1. Helper utilities (no external deps needed).
2. Converter unit tests — exercises individual op converters via
   :class:`~yobx.xbuilder.GraphBuilder` directly, without a real TFLite model.
3. Integration tests — exercises :func:`~yobx.litert.to_onnx` end-to-end with
   a hand-crafted TFLite FlatBuffer.
"""

import unittest

import numpy as np
from onnx import TensorProto

from yobx.ext_test_case import ExtTestCase
from yobx.litert.litert_helper import _make_sample_tflite_model
from yobx.xbuilder import GraphBuilder, OptimizationOptions

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

        model_bytes = _make_sample_tflite_model()
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

        model = parse_tflite_model(_make_sample_tflite_model())
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
            self._op(BuiltinOperator.RESHAPE, inputs=(0, 1), outputs=(0,)), ["X", "shape"], ["Y"]
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

        model_bytes = _make_sample_tflite_model()
        X = np.zeros((1, 4), dtype=np.float32)
        onx = to_onnx(model_bytes, (X,), input_names=["x"])

        # Graph must contain a Relu node.
        op_types = [n.op_type for n in onx.proto.graph.node]
        self.assertIn("Relu", op_types)

        # Output is numerically correct.
        from onnxruntime import InferenceSession

        sess = InferenceSession(onx.proto.SerializeToString(), providers=["CPUExecutionProvider"])
        feeds = {onx.proto.graph.input[0].name: X}
        (result,) = sess.run(None, feeds)
        self.assertEqualArray(np.maximum(X, 0), result)

    def test_to_onnx_input_names_mismatch_raises(self):
        """to_onnx() raises ValueError when input_names length mismatches."""
        from yobx.litert import to_onnx

        model_bytes = _make_sample_tflite_model()
        X = np.zeros((1, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            to_onnx(model_bytes, (X,), input_names=["a", "b"])

    def test_to_onnx_subgraph_out_of_range_raises(self):
        """to_onnx() raises ValueError for an invalid subgraph_index."""
        from yobx.litert import to_onnx

        model_bytes = _make_sample_tflite_model()
        X = np.zeros((1, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            to_onnx(model_bytes, (X,), subgraph_index=5)

    def test_to_onnx_from_file(self, tmp_path=None):
        """to_onnx() accepts a file path as well as raw bytes."""
        import os
        import tempfile

        from yobx.litert import to_onnx

        model_bytes = _make_sample_tflite_model()
        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as fh:
            fh.write(model_bytes)
            path = fh.name
        try:
            X = np.zeros((1, 4), dtype=np.float32)
            onx = to_onnx(path, (X,), input_names=["x"])
            op_types = [n.op_type for n in onx.proto.graph.node]
            self.assertIn("Relu", op_types)
        finally:
            os.unlink(path)

    def test_to_onnx_all_subgraphs_single_model(self):
        """to_onnx(subgraph_index=None) on a single-subgraph model returns one artifact."""
        from yobx.litert import to_onnx
        from yobx.container import ExportArtifact

        model_bytes = _make_sample_tflite_model()
        X = np.zeros((1, 4), dtype=np.float32)
        artifact = to_onnx(model_bytes, [(X,)], input_names=[["x"]], subgraph_index=None)

        self.assertIsInstance(artifact, ExportArtifact)
        op_types = [n.op_type for n in artifact.proto.graph.node]
        self.assertIn("Relu", op_types)

    def test_to_onnx_all_subgraphs_multi_model(self):
        """to_onnx(subgraph_index=None) merges two subgraphs into one ONNX model."""
        from yobx.litert import to_onnx
        from yobx.litert.litert_helper import _make_multi_subgraph_tflite_model
        from yobx.container import ExportArtifact
        from onnxruntime import InferenceSession

        model_bytes = _make_multi_subgraph_tflite_model()
        # Provide per-subgraph dummy inputs.
        X0 = np.random.default_rng(0).standard_normal((1, 4)).astype(np.float32)
        X1 = np.random.default_rng(1).standard_normal((1, 3)).astype(np.float32)
        artifact = to_onnx(model_bytes, [(X0,), (X1,)], subgraph_index=None)

        # Must be a single artifact, not a list.
        self.assertIsInstance(artifact, ExportArtifact)

        # The merged graph should contain two Relu nodes (one per subgraph).
        op_types = [n.op_type for n in artifact.proto.graph.node]
        relu_count = sum(1 for t in op_types if t == "Relu")
        self.assertEqual(relu_count, 2)

        # The merged model has two sets of inputs/outputs (prefixed sg0_/sg1_).
        input_names = [inp.name for inp in artifact.proto.graph.input]
        output_names = [out.name for out in artifact.proto.graph.output]
        self.assertEqual(len(input_names), 2)
        self.assertEqual(len(output_names), 2)

        # Verify outputs numerically.
        sess = InferenceSession(
            artifact.proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        feeds = {input_names[0]: X0, input_names[1]: X1}
        out0, out1 = sess.run(None, feeds)
        self.assertEqualArray(np.maximum(X0, 0), out0)
        self.assertEqualArray(np.maximum(X1, 0), out1)

    def test_to_onnx_all_subgraphs_no_args(self):
        """to_onnx(subgraph_index=None) works with no explicit args (inferred from model)."""
        from yobx.litert import to_onnx
        from yobx.litert.litert_helper import _make_multi_subgraph_tflite_model
        from yobx.container import ExportArtifact

        model_bytes = _make_multi_subgraph_tflite_model()
        artifact = to_onnx(model_bytes, subgraph_index=None)

        self.assertIsInstance(artifact, ExportArtifact)
        # Two inputs and two outputs from the two merged subgraphs.
        self.assertEqual(len(artifact.proto.graph.input), 2)
        self.assertEqual(len(artifact.proto.graph.output), 2)

    def test_to_onnx_all_subgraphs_input_names_mismatch_raises(self):
        from yobx.litert import to_onnx
        from yobx.litert.litert_helper import _make_multi_subgraph_tflite_model

        model_bytes = _make_multi_subgraph_tflite_model()
        with self.assertRaises(ValueError):
            to_onnx(
                model_bytes,
                subgraph_index=None,
                input_names=[["a", "b"], ["c"]],  # sg0 has 1 input, not 2
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
