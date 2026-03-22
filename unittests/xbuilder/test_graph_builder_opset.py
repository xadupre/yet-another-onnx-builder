"""
Unit tests for every ONNX operator created via GraphBuilder / Opset.

Each test builds a minimal ONNX graph using gr.op.<OpType>(...),
exports it with gr.to_onnx(), and validates the output with
onnxruntime.
"""

import unittest
import numpy as np
from onnx import TensorProto
from onnxruntime import InferenceSession
from yobx.ext_test_case import ExtTestCase
from yobx.xbuilder.graph_builder import GraphBuilder

TFLOAT = TensorProto.FLOAT
TINT64 = TensorProto.INT64
TBOOL = TensorProto.BOOL


def _run(model, feeds):
    """Run an ONNX model through onnxruntime and return the first output."""
    sess = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


def _builder(opset=18):
    return GraphBuilder(opset, ir_version=9)


class TestOpsetAbs(ExtTestCase):
    def test_abs(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        out = gr.op.Abs("X", outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.abs(x), result)


class TestOpsetAdd(ExtTestCase):
    def test_add(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        gr.make_tensor_input("Y", TFLOAT, ("a",))
        out = gr.op.Add("X", "Y", outputs=["Z"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x, "Y": y})
        self.assertEqualArray(x + y, result)


class TestOpsetAnd(ExtTestCase):
    def test_and(self):
        gr = _builder()
        gr.make_tensor_input("X", TBOOL, ("a",))
        gr.make_tensor_input("Y", TBOOL, ("a",))
        out = gr.op.And("X", "Y", outputs=["Z"])
        gr.make_tensor_output(out, TBOOL, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([True, False, True], dtype=bool)
        y = np.array([True, True, False], dtype=bool)
        (result,) = _run(onx, {"X": x, "Y": y})
        self.assertEqualArray(x & y, result)


class TestOpsetArgMax(ExtTestCase):
    def test_argmax(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.ArgMax("X", axis=1, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TINT64, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.argmax(x, axis=1), result)


class TestOpsetArgMin(ExtTestCase):
    def test_argmin(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.ArgMin("X", axis=1, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TINT64, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.argmin(x, axis=1), result)


class TestOpsetCast(ExtTestCase):
    def test_cast(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        out = gr.op.Cast("X", to=TINT64, outputs=["Y"])
        gr.make_tensor_output(out, TINT64, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1.5, 2.7, -3.1], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.astype(np.int64), result)


class TestOpsetCastLike(ExtTestCase):
    def test_castlike(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        gr.make_tensor_input("target_type", TINT64, ("a",))
        out = gr.op.CastLike("X", "target_type", outputs=["Y"])
        gr.make_tensor_output(out, TINT64, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1.5, 2.7, -3.1], dtype=np.float32)
        t = np.array([0, 1, 2], dtype=np.int64)
        (result,) = _run(onx, {"X": x, "target_type": t})
        self.assertEqualArray(x.astype(np.int64), result)


class TestOpsetCelu(ExtTestCase):
    def test_celu(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        out = gr.op.Celu("X", alpha=1.0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        expected = np.where(x >= 0, x, 1.0 * (np.exp(x / 1.0) - 1))
        self.assertEqualArray(expected, result, atol=1e-6)


class TestOpsetCompress(ExtTestCase):
    def test_compress(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        gr.make_tensor_input("condition", TBOOL, ("a",))
        out = gr.op.Compress("X", "condition", outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, (None,), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        cond = np.array([True, False, True, False], dtype=bool)
        (result,) = _run(onx, {"X": x, "condition": cond})
        self.assertEqualArray(x[cond], result)


class TestOpsetConcat(ExtTestCase):
    def test_concat(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        gr.make_tensor_input("Y", TFLOAT, ("b",))
        out = gr.op.Concat("X", "Y", axis=0, outputs=["Z"])
        gr.make_tensor_output(out, TFLOAT, (None,), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1.0, 2.0], dtype=np.float32)
        y = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x, "Y": y})
        self.assertEqualArray(np.concatenate([x, y]), result)


class TestOpsetConstant(ExtTestCase):
    def test_constant(self):
        gr = _builder()
        out = gr.op.Constant(value_float=3.14, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, (), indexed=False)
        onx = gr.to_onnx()
        (result,) = _run(onx, {})
        self.assertAlmostEqual(np.float32(3.14), result, atol=1e-5)


class TestOpsetConstantOfShape(ExtTestCase):
    def test_constant_of_shape(self):
        gr = _builder()
        gr.make_tensor_input("shape", TINT64, ("n",))
        out = gr.op.ConstantOfShape("shape", outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, (None,), indexed=False)
        onx = gr.to_onnx()
        shape = np.array([4], dtype=np.int64)
        (result,) = _run(onx, {"shape": shape})
        self.assertEqualArray(np.zeros(4, dtype=np.float32), result)


class TestOpsetCos(ExtTestCase):
    def test_cos(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        out = gr.op.Cos("X", outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([0.0, np.pi / 2, np.pi], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.cos(x), result, atol=1e-6)


class TestOpsetCosh(ExtTestCase):
    def test_cosh(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        out = gr.op.Cosh("X", outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.cosh(x), result, atol=1e-6)


class TestOpsetCumSum(ExtTestCase):
    def test_cumsum(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        axis = gr.make_initializer("", np.int64(0))
        out = gr.op.CumSum("X", axis, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.cumsum(x), result)


class TestOpsetDiv(ExtTestCase):
    def test_div(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        gr.make_tensor_input("Y", TFLOAT, ("a",))
        out = gr.op.Div("X", "Y", outputs=["Z"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([6.0, 4.0, 9.0], dtype=np.float32)
        y = np.array([2.0, 2.0, 3.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x, "Y": y})
        self.assertEqualArray(x / y, result)


class TestOpsetDropout(ExtTestCase):
    def test_dropout(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        out, mask = gr.op.Dropout("X", outputs=["Y", "mask"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        gr.make_tensor_output(mask, TBOOL, ("a",), indexed=False)
        onx = gr.to_onnx()
        # ratio=0 => identity (no dropout at inference by default)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _run(onx, {"X": x})
        self.assertEqualArray(x, result[0])


class TestOpsetElu(ExtTestCase):
    def test_elu(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        out = gr.op.Elu("X", alpha=1.0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        expected = np.where(x >= 0, x, 1.0 * (np.exp(x) - 1))
        self.assertEqualArray(expected, result, atol=1e-6)


class TestOpsetEqual(ExtTestCase):
    def test_equal(self):
        gr = _builder()
        gr.make_tensor_input("X", TINT64, ("a",))
        gr.make_tensor_input("Y", TINT64, ("a",))
        out = gr.op.Equal("X", "Y", outputs=["Z"])
        gr.make_tensor_output(out, TBOOL, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1, 2, 3], dtype=np.int64)
        y = np.array([1, 0, 3], dtype=np.int64)
        (result,) = _run(onx, {"X": x, "Y": y})
        self.assertEqualArray(x == y, result)


class TestOpsetExp(ExtTestCase):
    def test_exp(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        out = gr.op.Exp("X", outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.exp(x), result, atol=1e-6)


class TestOpsetExpand(ExtTestCase):
    def test_expand(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        shape = gr.make_initializer("", np.array([3, 4], dtype=np.int64))
        out = gr.op.Expand("X", shape, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("c", "d"), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.broadcast_to(x, (3, 4)), result)


class TestOpsetFlatten(ExtTestCase):
    def test_flatten(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.Flatten("X", axis=1, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a", None), indexed=False)
        onx = gr.to_onnx()
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.reshape(2, -1), result)


class TestOpsetGather(ExtTestCase):
    def test_gather(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        indices = gr.make_initializer("", np.array([0, 2], dtype=np.int64))
        out = gr.op.Gather("X", indices, axis=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("c", "b"), indexed=False)
        onx = gr.to_onnx()
        x = np.arange(9, dtype=np.float32).reshape(3, 3)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x[[0, 2]], result)


class TestOpsetGatherElements(ExtTestCase):
    def test_gather_elements(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        gr.make_tensor_input("indices", TINT64, ("a", "b"))
        out = gr.op.GatherElements("X", "indices", axis=1, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a", "b"), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        idx = np.array([[1, 0], [0, 1]], dtype=np.int64)
        (result,) = _run(onx, {"X": x, "indices": idx})
        expected = np.take_along_axis(x, idx, axis=1)
        self.assertEqualArray(expected, result)


class TestOpsetGatherND(ExtTestCase):
    def test_gather_nd(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        indices = gr.make_initializer("", np.array([[0, 0], [1, 1]], dtype=np.int64))
        out = gr.op.GatherND("X", indices, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, (None,), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.array([1.0, 4.0], dtype=np.float32), result)


class TestOpsetGemm(ExtTestCase):
    def test_gemm(self):
        gr = _builder()
        gr.make_tensor_input("A", TFLOAT, ("m", "k"))
        gr.make_tensor_input("B", TFLOAT, ("k", "n"))
        out = gr.op.Gemm("A", "B", outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("m", "n"), indexed=False)
        onx = gr.to_onnx()
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((3, 4), dtype=np.float32)
        (result,) = _run(onx, {"A": a, "B": b})
        self.assertEqualArray(a @ b, result)


class TestOpsetGreater(ExtTestCase):
    def test_greater(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        gr.make_tensor_input("Y", TFLOAT, ("a",))
        out = gr.op.Greater("X", "Y", outputs=["Z"])
        gr.make_tensor_output(out, TBOOL, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x, "Y": y})
        self.assertEqualArray(x > y, result)


class TestOpsetGreaterOrEqual(ExtTestCase):
    def test_greater_or_equal(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        gr.make_tensor_input("Y", TFLOAT, ("a",))
        out = gr.op.GreaterOrEqual("X", "Y", outputs=["Z"])
        gr.make_tensor_output(out, TBOOL, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x, "Y": y})
        self.assertEqualArray(x >= y, result)


class TestOpsetIdentity(ExtTestCase):
    def test_identity(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        out = gr.op.Identity("X", outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x, result)


class TestOpsetLess(ExtTestCase):
    def test_less(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        gr.make_tensor_input("Y", TFLOAT, ("a",))
        out = gr.op.Less("X", "Y", outputs=["Z"])
        gr.make_tensor_output(out, TBOOL, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x, "Y": y})
        self.assertEqualArray(x < y, result)


class TestOpsetLessOrEqual(ExtTestCase):
    def test_less_or_equal(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        gr.make_tensor_input("Y", TFLOAT, ("a",))
        out = gr.op.LessOrEqual("X", "Y", outputs=["Z"])
        gr.make_tensor_output(out, TBOOL, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x, "Y": y})
        self.assertEqualArray(x <= y, result)


class TestOpsetLog(ExtTestCase):
    def test_log(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        out = gr.op.Log("X", outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1.0, np.e, np.e**2], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.log(x), result, atol=1e-6)


class TestOpsetLogSoftmax(ExtTestCase):
    def test_log_softmax(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.LogSoftmax("X", axis=1, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a", "b"), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        shifted = x - x.max(axis=1, keepdims=True)
        expected = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        self.assertEqualArray(expected, result, atol=1e-6)


class TestOpsetMatMul(ExtTestCase):
    def test_matmul(self):
        gr = _builder()
        gr.make_tensor_input("A", TFLOAT, ("m", "k"))
        gr.make_tensor_input("B", TFLOAT, ("k", "n"))
        out = gr.op.MatMul("A", "B", outputs=["C"])
        gr.make_tensor_output(out, TFLOAT, ("m", "n"), indexed=False)
        onx = gr.to_onnx()
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        b = np.arange(6, dtype=np.float32).reshape(3, 2)
        (result,) = _run(onx, {"A": a, "B": b})
        self.assertEqualArray(a @ b, result)


class TestOpsetMaxPool(ExtTestCase):
    def test_maxpool(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("n", "c", "h", "w"))
        out, indices = gr.op.MaxPool("X", kernel_shape=[2, 2], outputs=["Y", "indices"])
        gr.make_tensor_output(
            out, TFLOAT, ("n", "c", None, None), indexed=False
        )
        gr.make_tensor_output(
            indices, TINT64, ("n", "c", None, None), indexed=False
        )
        onx = gr.to_onnx()
        x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        result = _run(onx, {"X": x})
        self.assertEqual(result[0].shape, (1, 1, 3, 3))


class TestOpsetMul(ExtTestCase):
    def test_mul(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        gr.make_tensor_input("Y", TFLOAT, ("a",))
        out = gr.op.Mul("X", "Y", outputs=["Z"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x, "Y": y})
        self.assertEqualArray(x * y, result)


class TestOpsetNeg(ExtTestCase):
    def test_neg(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        out = gr.op.Neg("X", outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(-x, result)


class TestOpsetNot(ExtTestCase):
    def test_not(self):
        gr = _builder()
        gr.make_tensor_input("X", TBOOL, ("a",))
        out = gr.op.Not("X", outputs=["Y"])
        gr.make_tensor_output(out, TBOOL, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([True, False, True], dtype=bool)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(~x, result)


class TestOpsetOr(ExtTestCase):
    def test_or(self):
        gr = _builder()
        gr.make_tensor_input("X", TBOOL, ("a",))
        gr.make_tensor_input("Y", TBOOL, ("a",))
        out = gr.op.Or("X", "Y", outputs=["Z"])
        gr.make_tensor_output(out, TBOOL, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([True, False, False], dtype=bool)
        y = np.array([False, True, False], dtype=bool)
        (result,) = _run(onx, {"X": x, "Y": y})
        self.assertEqualArray(x | y, result)


class TestOpsetPow(ExtTestCase):
    def test_pow(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        gr.make_tensor_input("Y", TFLOAT, ("a",))
        out = gr.op.Pow("X", "Y", outputs=["Z"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        y = np.array([2.0, 3.0, 2.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x, "Y": y})
        self.assertEqualArray(x**y, result, atol=1e-6)


class TestOpsetRange(ExtTestCase):
    def test_range(self):
        gr = _builder()
        start = gr.make_initializer("", np.int64(0))
        limit = gr.make_initializer("", np.int64(5))
        delta = gr.make_initializer("", np.int64(1))
        out = gr.op.Range(start, limit, delta, outputs=["Y"])
        gr.make_tensor_output(out, TINT64, (None,), indexed=False)
        onx = gr.to_onnx()
        (result,) = _run(onx, {})
        self.assertEqualArray(np.arange(5, dtype=np.int64), result)


class TestOpsetReciprocal(ExtTestCase):
    def test_reciprocal(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        out = gr.op.Reciprocal("X", outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([2.0, 4.0, 8.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(1.0 / x, result, atol=1e-6)


class TestOpsetReduceMax(ExtTestCase):
    def test_reduce_max(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceMax("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.max(axis=1), result)

    def test_reduce_max_opset10(self):
        gr = _builder(opset=10)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.ReduceMax("X", axes=[1], keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.max(axis=1), result)

    def test_reduce_max_opset20(self):
        gr = _builder(opset=20)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceMax("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.max(axis=1), result)


class TestOpsetReduceMean(ExtTestCase):
    def test_reduce_mean(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceMean("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.mean(axis=1), result, atol=1e-6)

    def test_reduce_mean_opset10(self):
        gr = _builder(opset=10)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.ReduceMean("X", axes=[1], keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.mean(axis=1), result, atol=1e-6)

    def test_reduce_mean_opset20(self):
        gr = _builder(opset=20)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceMean("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.mean(axis=1), result, atol=1e-6)


class TestOpsetReduceMin(ExtTestCase):
    def test_reduce_min(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceMin("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.min(axis=1), result)

    def test_reduce_min_opset10(self):
        gr = _builder(opset=10)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.ReduceMin("X", axes=[1], keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.min(axis=1), result)

    def test_reduce_min_opset20(self):
        gr = _builder(opset=20)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceMin("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.min(axis=1), result)


class TestOpsetReduceSum(ExtTestCase):
    def test_reduce_sum(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceSum("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.sum(axis=1), result)

    def test_reduce_sum_opset10(self):
        gr = _builder(opset=10)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.ReduceSum("X", axes=[1], keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.sum(axis=1), result)

    def test_reduce_sum_opset20(self):
        gr = _builder(opset=20)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceSum("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.sum(axis=1), result)


class TestOpsetReduceProd(ExtTestCase):
    def test_reduce_prod(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceProd("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.5, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.prod(axis=1), result, atol=1e-6)

    def test_reduce_prod_opset10(self):
        gr = _builder(opset=10)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.ReduceProd("X", axes=[1], keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.5, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.prod(axis=1), result, atol=1e-6)

    def test_reduce_prod_opset20(self):
        gr = _builder(opset=20)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceProd("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.5, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.prod(axis=1), result, atol=1e-6)


class TestOpsetReduceLogSumExp(ExtTestCase):
    def test_reduce_log_sum_exp(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceLogSumExp("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        expected = np.log(np.sum(np.exp(x), axis=1))
        self.assertEqualArray(expected, result, atol=1e-4)

    def test_reduce_log_sum_exp_opset10(self):
        gr = _builder(opset=10)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.ReduceLogSumExp("X", axes=[1], keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        expected = np.log(np.sum(np.exp(x), axis=1))
        self.assertEqualArray(expected, result, atol=1e-4)

    def test_reduce_log_sum_exp_opset20(self):
        gr = _builder(opset=20)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceLogSumExp("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        expected = np.log(np.sum(np.exp(x), axis=1))
        self.assertEqualArray(expected, result, atol=1e-4)


class TestOpsetRelu(ExtTestCase):
    def test_relu(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        out = gr.op.Relu("X", outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.maximum(x, 0), result)


class TestOpsetReshape(ExtTestCase):
    def test_reshape(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        shape = gr.make_initializer("", np.array([2, 3], dtype=np.int64))
        out = gr.op.Reshape("X", shape, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("c", "d"), indexed=False)
        onx = gr.to_onnx()
        x = np.arange(6, dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.reshape(2, 3), result)


class TestOpsetScatterElements(ExtTestCase):
    def test_scatter_elements(self):
        gr = _builder()
        gr.make_tensor_input("data", TFLOAT, ("a", "b"))
        gr.make_tensor_input("indices", TINT64, ("a", "b"))
        gr.make_tensor_input("updates", TFLOAT, ("a", "b"))
        out = gr.op.ScatterElements("data", "indices", "updates", axis=1, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a", "b"), indexed=False)
        onx = gr.to_onnx()
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        indices = np.array([[0, 1], [2, 0]], dtype=np.int64)
        updates = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        (result,) = _run(onx, {"data": data, "indices": indices, "updates": updates})
        expected = data.copy()
        for i in range(2):
            for j in range(2):
                expected[i, indices[i, j]] = updates[i, j]
        self.assertEqualArray(expected, result)


class TestOpsetScatterND(ExtTestCase):
    def test_scatter_nd(self):
        gr = _builder()
        gr.make_tensor_input("data", TFLOAT, ("a", "b"))
        indices = gr.make_initializer("", np.array([[0], [2]], dtype=np.int64))
        updates = gr.make_initializer(
            "", np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        )
        out = gr.op.ScatterND("data", indices, updates, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a", "b"), indexed=False)
        onx = gr.to_onnx()
        data = np.zeros((3, 2), dtype=np.float32)
        (result,) = _run(onx, {"data": data})
        expected = data.copy()
        expected[0] = [10.0, 20.0]
        expected[2] = [30.0, 40.0]
        self.assertEqualArray(expected, result)


class TestOpsetShape(ExtTestCase):
    def test_shape(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.Shape("X", outputs=["Y"])
        gr.make_tensor_output(out, TINT64, (None,), indexed=False)
        onx = gr.to_onnx()
        x = np.zeros((3, 4), dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.array([3, 4], dtype=np.int64), result)


class TestOpsetSigmoid(ExtTestCase):
    def test_sigmoid(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        out = gr.op.Sigmoid("X", outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        expected = 1 / (1 + np.exp(-x))
        self.assertEqualArray(expected, result, atol=1e-6)


class TestOpsetSin(ExtTestCase):
    def test_sin(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        out = gr.op.Sin("X", outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([0.0, np.pi / 2, np.pi], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.sin(x), result, atol=1e-6)


class TestOpsetSinh(ExtTestCase):
    def test_sinh(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        out = gr.op.Sinh("X", outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.sinh(x), result, atol=1e-6)


class TestOpsetSlice(ExtTestCase):
    def test_slice(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        starts = gr.make_initializer("", np.array([1], dtype=np.int64))
        ends = gr.make_initializer("", np.array([4], dtype=np.int64))
        out = gr.op.Slice("X", starts, ends, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, (None,), indexed=False)
        onx = gr.to_onnx()
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x[1:4], result)


class TestOpsetSoftmax(ExtTestCase):
    def test_softmax(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.Softmax("X", axis=1, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a", "b"), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        shifted = x - x.max(axis=1, keepdims=True)
        expected = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
        self.assertEqualArray(expected, result, atol=1e-6)


class TestOpsetSqrt(ExtTestCase):
    def test_sqrt(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        out = gr.op.Sqrt("X", outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1.0, 4.0, 9.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.sqrt(x), result, atol=1e-6)


class TestOpsetSqueeze(ExtTestCase):
    def test_squeeze(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b", "c"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.Squeeze("X", axes, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a", "c"), indexed=False)
        onx = gr.to_onnx()
        x = np.zeros((2, 1, 3), dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.squeeze(axis=1), result)


class TestOpsetSub(ExtTestCase):
    def test_sub(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        gr.make_tensor_input("Y", TFLOAT, ("a",))
        out = gr.op.Sub("X", "Y", outputs=["Z"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([5.0, 3.0, 1.0], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x, "Y": y})
        self.assertEqualArray(x - y, result)


class TestOpsetTile(ExtTestCase):
    def test_tile(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        repeats = gr.make_initializer("", np.array([3], dtype=np.int64))
        out = gr.op.Tile("X", repeats, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, (None,), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1.0, 2.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.tile(x, 3), result)


class TestOpsetTopK(ExtTestCase):
    def test_topk(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        k = gr.make_initializer("", np.array([2], dtype=np.int64))
        values, indices = gr.op.TopK("X", k, outputs=["values", "indices"])
        gr.make_tensor_output(values, TFLOAT, (None,), indexed=False)
        gr.make_tensor_output(indices, TINT64, (None,), indexed=False)
        onx = gr.to_onnx()
        x = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float32)
        result = _run(onx, {"X": x})
        self.assertEqualArray(np.array([5.0, 4.0], dtype=np.float32), result[0])
        self.assertEqualArray(np.array([4, 2], dtype=np.int64), result[1])


class TestOpsetTranspose(ExtTestCase):
    def test_transpose(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.Transpose("X", perm=[1, 0], outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("b", "a"), indexed=False)
        onx = gr.to_onnx()
        x = np.arange(6, dtype=np.float32).reshape(2, 3)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.T, result)


class TestOpsetTrilu(ExtTestCase):
    def test_trilu_upper(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.Trilu("X", upper=1, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a", "b"), indexed=False)
        onx = gr.to_onnx()
        x = np.ones((3, 3), dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.triu(x), result)


class TestOpsetUnsqueeze(ExtTestCase):
    def test_unsqueeze(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        axes = gr.make_initializer("", np.array([0], dtype=np.int64))
        out = gr.op.Unsqueeze("X", axes, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("c", "a"), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.expand_dims(x, 0), result)


class TestOpsetWhere(ExtTestCase):
    def test_where(self):
        gr = _builder()
        gr.make_tensor_input("condition", TBOOL, ("a",))
        gr.make_tensor_input("X", TFLOAT, ("a",))
        gr.make_tensor_input("Y", TFLOAT, ("a",))
        out = gr.op.Where("condition", "X", "Y", outputs=["Z"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        cond = np.array([True, False, True], dtype=bool)
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (result,) = _run(onx, {"condition": cond, "X": x, "Y": y})
        self.assertEqualArray(np.where(cond, x, y), result)


# ---------------------------------------------------------------------------
# Tests for the AnyOpset helper methods
# ---------------------------------------------------------------------------


class TestReduceMaxAnyOpset(ExtTestCase):
    def test_reduce_max_any_opset_two_inputs(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceMaxAnyOpset("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.max(axis=1), result)

    def test_reduce_max_any_opset_one_input(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.ReduceMaxAnyOpset("X", keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, (), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.array(x.max(), dtype=np.float32), result)

    def test_reduce_max_any_opset_opset10(self):
        gr = _builder(opset=10)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.ReduceMaxAnyOpset(
            "X", np.array([1], dtype=np.int64), keepdims=0, outputs=["Y"]
        )
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.max(axis=1), result)

    def test_reduce_max_any_opset_opset20(self):
        gr = _builder(opset=20)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceMaxAnyOpset("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.max(axis=1), result)


class TestReduceMinAnyOpset(ExtTestCase):
    def test_reduce_min_any_opset_two_inputs(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceMinAnyOpset("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.min(axis=1), result)

    def test_reduce_min_any_opset_opset10(self):
        gr = _builder(opset=10)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.ReduceMinAnyOpset(
            "X", np.array([1], dtype=np.int64), keepdims=0, outputs=["Y"]
        )
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.min(axis=1), result)

    def test_reduce_min_any_opset_opset20(self):
        gr = _builder(opset=20)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceMinAnyOpset("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.min(axis=1), result)


class TestReduceMeanAnyOpset(ExtTestCase):
    def test_reduce_mean_any_opset_two_inputs(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceMeanAnyOpset("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.mean(axis=1), result, atol=1e-6)

    def test_reduce_mean_any_opset_opset10(self):
        gr = _builder(opset=10)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.ReduceMeanAnyOpset(
            "X", np.array([1], dtype=np.int64), keepdims=0, outputs=["Y"]
        )
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.mean(axis=1), result, atol=1e-6)

    def test_reduce_mean_any_opset_opset20(self):
        gr = _builder(opset=20)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceMeanAnyOpset("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.mean(axis=1), result, atol=1e-6)


class TestReduceSumAnyOpset(ExtTestCase):
    def test_reduce_sum_any_opset_two_inputs(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceSumAnyOpset("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.sum(axis=1), result)

    def test_reduce_sum_any_opset_opset10(self):
        gr = _builder(opset=10)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.ReduceSumAnyOpset(
            "X", np.array([1], dtype=np.int64), keepdims=0, outputs=["Y"]
        )
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.sum(axis=1), result)

    def test_reduce_sum_any_opset_opset20(self):
        gr = _builder(opset=20)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceSumAnyOpset("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.sum(axis=1), result)


class TestReduceProdAnyOpset(ExtTestCase):
    def test_reduce_prod_any_opset_two_inputs(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceProdAnyOpset("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.5, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.prod(axis=1), result, atol=1e-6)

    def test_reduce_prod_any_opset_opset10(self):
        gr = _builder(opset=10)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.ReduceProdAnyOpset(
            "X", np.array([1], dtype=np.int64), keepdims=0, outputs=["Y"]
        )
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.5, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.prod(axis=1), result, atol=1e-6)

    def test_reduce_prod_any_opset_opset20(self):
        gr = _builder(opset=20)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceProdAnyOpset("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.5, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.prod(axis=1), result, atol=1e-6)


class TestReduceLogSumExpAnyOpset(ExtTestCase):
    def test_reduce_log_sum_exp_any_opset_two_inputs(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceLogSumExpAnyOpset("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        expected = np.log(np.sum(np.exp(x), axis=1))
        self.assertEqualArray(expected, result, atol=1e-4)

    def test_reduce_log_sum_exp_any_opset_opset10(self):
        gr = _builder(opset=10)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        out = gr.op.ReduceLogSumExpAnyOpset(
            "X", np.array([1], dtype=np.int64), keepdims=0, outputs=["Y"]
        )
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        expected = np.log(np.sum(np.exp(x), axis=1))
        self.assertEqualArray(expected, result, atol=1e-4)

    def test_reduce_log_sum_exp_any_opset_opset20(self):
        gr = _builder(opset=20)
        gr.make_tensor_input("X", TFLOAT, ("a", "b"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.ReduceLogSumExpAnyOpset("X", axes, keepdims=0, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a",), indexed=False)
        onx = gr.to_onnx()
        x = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        expected = np.log(np.sum(np.exp(x), axis=1))
        self.assertEqualArray(expected, result, atol=1e-4)


class TestSqueezeAnyOpset(ExtTestCase):
    def test_squeeze_any_opset_two_inputs(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a", "b", "c"))
        axes = gr.make_initializer("", np.array([1], dtype=np.int64))
        out = gr.op.SqueezeAnyOpset("X", axes, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("a", "c"), indexed=False)
        onx = gr.to_onnx()
        x = np.zeros((2, 1, 3), dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.squeeze(axis=1), result)

    def test_squeeze_any_opset_one_input(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, (None, None, None))
        out = gr.op.SqueezeAnyOpset("X", outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, (), indexed=False)
        onx = gr.to_onnx()
        x = np.zeros((1, 1, 1), dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(x.squeeze(), result)


class TestUnsqueezeAnyOpset(ExtTestCase):
    def test_unsqueeze_any_opset(self):
        gr = _builder()
        gr.make_tensor_input("X", TFLOAT, ("a",))
        axes = gr.make_initializer("", np.array([0], dtype=np.int64))
        out = gr.op.UnsqueezeAnyOpset("X", axes, outputs=["Y"])
        gr.make_tensor_output(out, TFLOAT, ("c", "a"), indexed=False)
        onx = gr.to_onnx()
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(np.expand_dims(x, 0), result)


class TestDFTAnyOpset(ExtTestCase):
    def _expected_dft(self, x):
        """Compute expected DFT output matching ONNX DFT op (real input, axis=1).

        x: shape (batch, signal_dim, 1) - real float input
        returns: shape (batch, signal_dim, 2) - complex output as real/imag pairs
        """
        signal = x[:, :, 0]
        dft = np.fft.fft(signal, axis=1)
        real = np.real(dft)[:, :, np.newaxis]
        imag = np.imag(dft)[:, :, np.newaxis]
        return np.concatenate([real, imag], axis=-1).astype(np.float32)

    def test_dft_any_opset_opset17(self):
        """DFTAnyOpset on opset 17 (< 20): axes numpy array is converted to axis attribute."""
        gr = GraphBuilder(17, ir_version=8)
        gr.make_tensor_input("X", TFLOAT, ("batch", "signal_dim", "one"))
        axes = np.array([1], dtype=np.int64)
        out = gr.op.DFTAnyOpset("X", "", axes, outputs=["Y"])
        gr.make_tensor_output(
            out, TFLOAT, ("batch", "signal_dim", "two"), indexed=False
        )
        onx = gr.to_onnx()
        x = np.arange(8, dtype=np.float32).reshape(1, 8, 1)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(self._expected_dft(x), result, atol=1e-5)

    def test_dft_any_opset_opset20(self):
        """DFTAnyOpset on opset 20 (>= 20): axis kwarg is converted to scalar tensor input."""
        gr = GraphBuilder(20, ir_version=9)
        gr.make_tensor_input("X", TFLOAT, ("batch", "signal_dim", "one"))
        out = gr.op.DFTAnyOpset("X", "", outputs=["Y"], axis=1)
        gr.make_tensor_output(
            out, TFLOAT, ("batch", "signal_dim", "two"), indexed=False
        )
        onx = gr.to_onnx()
        x = np.arange(8, dtype=np.float32).reshape(1, 8, 1)
        (result,) = _run(onx, {"X": x})
        self.assertEqualArray(self._expected_dft(x), result, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
