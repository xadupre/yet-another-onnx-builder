import unittest
import onnx
import onnx.helper as oh
from yobx.ext_test_case import ExtTestCase
from yobx.xshape.type_inference import infer_types

TFLOAT = onnx.TensorProto.FLOAT
TFLOAT16 = onnx.TensorProto.FLOAT16
TINT64 = onnx.TensorProto.INT64
TBOOL = onnx.TensorProto.BOOL


class TestTypeInference(ExtTestCase):
    def test_infer_types_relu(self):
        node = oh.make_node("Relu", ["X"], ["Y"])
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_add(self):
        node = oh.make_node("Add", ["X", "Y"], ["Z"])
        result = infer_types(node, [TFLOAT, TFLOAT])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_matmul(self):
        node = oh.make_node("MatMul", ["X", "Y"], ["Z"])
        result = infer_types(node, [TFLOAT, TFLOAT])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_cast(self):
        node = oh.make_node("Cast", ["X"], ["Y"], to=TINT64)
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, (TINT64,))

    def test_infer_types_cast_like(self):
        node = oh.make_node("CastLike", ["X", "target"], ["Y"])
        result = infer_types(node, [TFLOAT, TINT64])
        self.assertEqual(result, (TINT64,))

    def test_infer_types_constant_int(self):
        node = oh.make_node("Constant", [], ["C"], value_int=5)
        result = infer_types(node, [])
        self.assertEqual(result, (TINT64,))

    def test_infer_types_constant_float(self):
        node = oh.make_node("Constant", [], ["C"], value_float=3.14)
        result = infer_types(node, [])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_constant_of_shape_default(self):
        node = oh.make_node("ConstantOfShape", ["shape"], ["C"])
        result = infer_types(node, [TINT64])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_shape(self):
        node = oh.make_node("Shape", ["X"], ["S"])
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, (TINT64,))

    def test_infer_types_size(self):
        node = oh.make_node("Size", ["X"], ["S"])
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, (TINT64,))

    def test_infer_types_where(self):
        node = oh.make_node("Where", ["cond", "X", "Y"], ["Z"])
        result = infer_types(node, [TBOOL, TFLOAT, TFLOAT])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_split(self):
        node = oh.make_node("Split", ["X"], ["Y1", "Y2"])
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, [TFLOAT, TFLOAT])

    def test_infer_types_range(self):
        node = oh.make_node("Range", ["start", "end", "step"], ["Y"])
        result = infer_types(node, [TINT64, TINT64, TINT64])
        self.assertEqual(result, (TINT64,))

    def test_infer_types_eye_like_default(self):
        node = oh.make_node("EyeLike", ["X"], ["Y"])
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_eye_like_with_dtype(self):
        node = oh.make_node("EyeLike", ["X"], ["Y"], dtype=TINT64)
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, (TINT64,))

    def test_infer_types_unsupported_no_exc(self):
        node = oh.make_node("NonMaxSuppression", ["X"], ["Y"])
        result = infer_types(node, [TFLOAT], exc=False)
        self.assertEqual(result, 0)

    def test_infer_types_unsupported_exc(self):
        node = oh.make_node("NonMaxSuppression", ["X"], ["Y"])
        self.assertRaise(lambda: infer_types(node, [TFLOAT], exc=True), RuntimeError)

    def test_infer_types_output_name(self):
        node = oh.make_node("Relu", ["X"], ["Y"])
        result = infer_types(node, [TFLOAT], output_name="Y")
        self.assertEqual(result, TFLOAT)

    def test_infer_types_reshape(self):
        node = oh.make_node("Reshape", ["X", "shape"], ["Y"])
        result = infer_types(node, [TFLOAT, TINT64])
        self.assertEqual(result, (TFLOAT,))

    def test_infer_types_transpose(self):
        node = oh.make_node("Transpose", ["X"], ["Y"], perm=[1, 0])
        result = infer_types(node, [TFLOAT])
        self.assertEqual(result, (TFLOAT,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
