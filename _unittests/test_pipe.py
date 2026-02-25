"""Unit tests for onnx_pipe."""

import unittest

import numpy as np
import onnx
import onnxruntime as rt
from onnx import TensorProto, helper

from onnx_pipe import OnnxPipe, op


class TestOnnxPipeOp(unittest.TestCase):
    """Tests for the op() helper."""

    def test_op_creates_onnxpipe(self):
        pipe = op("Abs")
        self.assertIsInstance(pipe, OnnxPipe)

    def test_op_default_input_output_names(self):
        pipe = op("Abs")
        self.assertEqual(pipe.input_names, ["X"])
        self.assertEqual(pipe.output_names, ["Y"])

    def test_op_custom_input_output_names(self):
        pipe = op("Add", input_names=["A", "B"], output_names=["C"])
        self.assertEqual(pipe.input_names, ["A", "B"])
        self.assertEqual(pipe.output_names, ["C"])

    def test_op_to_onnx_returns_model_proto(self):
        pipe = op("Relu")
        model = pipe.to_onnx()
        self.assertIsInstance(model, onnx.ModelProto)

    def test_op_model_is_valid(self):
        pipe = op("Abs")
        onnx.checker.check_model(pipe.to_onnx())

    def test_op_runnable(self):
        pipe = op("Abs")
        sess = rt.InferenceSession(pipe.to_onnx().SerializeToString())
        x = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
        (result,) = sess.run(None, {"X": x})
        np.testing.assert_array_equal(result, np.abs(x))


class TestOnnxPipePipeOperator(unittest.TestCase):
    """Tests for the | operator on OnnxPipe."""

    def test_pipe_two_ops(self):
        pipe = op("Abs") | op("Relu")
        self.assertIsInstance(pipe, OnnxPipe)

    def test_pipe_model_is_valid(self):
        pipe = op("Abs") | op("Relu")
        onnx.checker.check_model(pipe.to_onnx())

    def test_pipe_runnable(self):
        pipe = op("Abs") | op("Relu")
        sess = rt.InferenceSession(pipe.to_onnx().SerializeToString())
        x = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
        (result,) = sess.run(None, {"X": x})
        # Abs then Relu: abs(x) is always >= 0, so relu has no effect
        np.testing.assert_array_equal(result, np.abs(x))

    def test_pipe_three_ops(self):
        pipe = op("Abs") | op("Relu") | op("Sigmoid")
        onnx.checker.check_model(pipe.to_onnx())
        sess = rt.InferenceSession(pipe.to_onnx().SerializeToString())
        x = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
        (result,) = sess.run(None, {"X": x})
        expected = 1.0 / (1.0 + np.exp(-np.abs(x)))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_pipe_input_output_names(self):
        pipe = op("Abs") | op("Relu")
        self.assertEqual(pipe.input_names, ["X"])
        # output name comes from the second model after prefixing
        self.assertEqual(len(pipe.output_names), 1)

    def test_pipe_wrong_type_raises(self):
        pipe = op("Abs")
        with self.assertRaises(TypeError):
            _ = pipe | "not_a_pipe"

    def test_pipe_existing_model(self):
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node = helper.make_node("Neg", ["X"], ["Y"])
        graph = helper.make_graph([node], "neg_graph", [X], [Y])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 20)]
        )
        pipe = OnnxPipe(model) | op("Abs")
        onnx.checker.check_model(pipe.to_onnx())
        sess = rt.InferenceSession(pipe.to_onnx().SerializeToString())
        x = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
        (result,) = sess.run(None, {"X": x})
        np.testing.assert_array_equal(result, np.abs(-x))

    def test_pipe_repr(self):
        pipe = op("Abs")
        r = repr(pipe)
        self.assertIn("OnnxPipe", r)
        self.assertIn("X", r)
        self.assertIn("Y", r)


if __name__ == "__main__":
    unittest.main()
