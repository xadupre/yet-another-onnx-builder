import unittest
import onnx
from yobx.ext_test_case import ExtTestCase

try:
    import torch as _torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class _FakeSymbolicTensor:
    """Mock object whose str() returns a _TYPENAME key, simulating a SymbolicTensor."""

    def __init__(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return self._name


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestTorchDtypeToOnnxDtype(ExtTestCase):
    def setUp(self):
        from yobx.helpers.torch_helper import torch_dtype_to_onnx_dtype

        self.convert = torch_dtype_to_onnx_dtype

    def test_float32(self):
        self.assertEqual(self.convert(_torch.float32), onnx.TensorProto.FLOAT)

    def test_float16(self):
        self.assertEqual(self.convert(_torch.float16), onnx.TensorProto.FLOAT16)

    def test_bfloat16(self):
        self.assertEqual(self.convert(_torch.bfloat16), onnx.TensorProto.BFLOAT16)

    def test_float64(self):
        self.assertEqual(self.convert(_torch.float64), onnx.TensorProto.DOUBLE)

    def test_int64(self):
        self.assertEqual(self.convert(_torch.int64), onnx.TensorProto.INT64)

    def test_int32(self):
        self.assertEqual(self.convert(_torch.int32), onnx.TensorProto.INT32)

    def test_uint64(self):
        self.assertEqual(self.convert(_torch.uint64), onnx.TensorProto.UINT64)

    def test_uint32(self):
        self.assertEqual(self.convert(_torch.uint32), onnx.TensorProto.UINT32)

    def test_bool(self):
        self.assertEqual(self.convert(_torch.bool), onnx.TensorProto.BOOL)

    def test_symint(self):
        self.assertEqual(self.convert(_torch.SymInt), onnx.TensorProto.INT64)

    def test_int16(self):
        self.assertEqual(self.convert(_torch.int16), onnx.TensorProto.INT16)

    def test_uint16(self):
        self.assertEqual(self.convert(_torch.uint16), onnx.TensorProto.UINT16)

    def test_int8(self):
        self.assertEqual(self.convert(_torch.int8), onnx.TensorProto.INT8)

    def test_uint8(self):
        self.assertEqual(self.convert(_torch.uint8), onnx.TensorProto.UINT8)

    def test_symfloat(self):
        self.assertEqual(self.convert(_torch.SymFloat), onnx.TensorProto.FLOAT)

    def test_complex64(self):
        self.assertEqual(self.convert(_torch.complex64), onnx.TensorProto.COMPLEX64)

    def test_complex128(self):
        self.assertEqual(self.convert(_torch.complex128), onnx.TensorProto.COMPLEX128)

    def test_symbolic_tensor_float(self):
        self.assertEqual(self.convert(_FakeSymbolicTensor("FLOAT")), onnx.TensorProto.FLOAT)

    def test_symbolic_tensor_int64(self):
        self.assertEqual(self.convert(_FakeSymbolicTensor("INT64")), onnx.TensorProto.INT64)

    def test_symbolic_tensor_int32(self):
        self.assertEqual(self.convert(_FakeSymbolicTensor("INT32")), onnx.TensorProto.INT32)

    def test_symbolic_tensor_float16(self):
        self.assertEqual(self.convert(_FakeSymbolicTensor("FLOAT16")), onnx.TensorProto.FLOAT16)

    def test_symbolic_tensor_bfloat16(self):
        self.assertEqual(self.convert(_FakeSymbolicTensor("BFLOAT16")), onnx.TensorProto.BFLOAT16)

    def test_unknown_raises(self):
        with self.assertRaises(NotImplementedError):
            self.convert(_FakeSymbolicTensor("UNKNOWN"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
