import unittest
import onnx
from yobx.ext_test_case import ExtTestCase
from yobx.helpers.onnx_helper import onnx_dtype_name


class TestOnnxHelper(ExtTestCase):
    def test_onnx_dtype_name(self):
        for k in dir(onnx.TensorProto):
            if k.upper() == k and k not in {"DESCRIPTOR", "EXTERNAL", "DEFAULT"}:
                self.assertEqual(k, onnx_dtype_name(getattr(onnx.TensorProto, k)))
        self.assertRaise(lambda: onnx_dtype_name(1000), ValueError)
        self.assertEqual(onnx_dtype_name(1000, exc=False), "UNEXPECTED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
