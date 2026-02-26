import unittest
import ml_dtypes
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.ext_test_case import ExtTestCase
from yobx.helpers.onnx_helper import onnx_dtype_name, tensor_dtype_to_np_dtype, pretty_onnx

TFLOAT = onnx.TensorProto.FLOAT


class TestOnnxHelper(ExtTestCase):
    def test_onnx_dtype_name(self):
        for k in dir(onnx.TensorProto):
            if k.upper() == k and k not in {"DESCRIPTOR", "EXTERNAL", "DEFAULT"}:
                self.assertEqual(k, onnx_dtype_name(getattr(onnx.TensorProto, k)))
        self.assertRaise(lambda: onnx_dtype_name(1000), ValueError)
        self.assertEqual(onnx_dtype_name(1000, exc=False), "UNEXPECTED")

    def test_tensor_dtype_to_np_dtype_standard(self):
        self.assertEqual(tensor_dtype_to_np_dtype(onnx.TensorProto.FLOAT), np.float32)
        self.assertEqual(tensor_dtype_to_np_dtype(onnx.TensorProto.DOUBLE), np.float64)
        self.assertEqual(tensor_dtype_to_np_dtype(onnx.TensorProto.INT32), np.int32)
        self.assertEqual(tensor_dtype_to_np_dtype(onnx.TensorProto.INT64), np.int64)
        self.assertEqual(tensor_dtype_to_np_dtype(onnx.TensorProto.BOOL), np.bool_)

    def test_tensor_dtype_to_np_dtype_float8(self):
        self.assertEqual(tensor_dtype_to_np_dtype(onnx.TensorProto.BFLOAT16), ml_dtypes.bfloat16)
        self.assertEqual(
            tensor_dtype_to_np_dtype(onnx.TensorProto.FLOAT8E4M3FN),
            ml_dtypes.float8_e4m3fn,
        )
        self.assertEqual(
            tensor_dtype_to_np_dtype(onnx.TensorProto.FLOAT8E4M3FNUZ),
            ml_dtypes.float8_e4m3fnuz,
        )
        self.assertEqual(
            tensor_dtype_to_np_dtype(onnx.TensorProto.FLOAT8E5M2), ml_dtypes.float8_e5m2
        )
        self.assertEqual(
            tensor_dtype_to_np_dtype(onnx.TensorProto.FLOAT8E5M2FNUZ),
            ml_dtypes.float8_e5m2fnuz,
        )

    def test_pretty_onnx(self):
        model = oh.make_model(
            oh.make_graph(
                [
                    oh.make_node("Add", ["X", "Y"], ["added"]),
                    oh.make_node("Concat", ["added", "X"], ["concat_out"], axis=2),
                    oh.make_node("Reshape", ["concat_out", "reshape_shape"], ["Z"]),
                ],
                "add_concat_reshape",
                [
                    oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", "d_model"]),
                    oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", "d_model"]),
                ],
                [oh.make_tensor_value_info("Z", TFLOAT, [None, None, None])],
                [
                    onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="reshape_shape"),
                ],
            ),
            opset_imports=[oh.make_opsetid("", 18)],
            ir_version=10,
        )
        text = pretty_onnx(model)
        self.assertIn("Reshape(concat_out, reshape_shape) -> Z", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
