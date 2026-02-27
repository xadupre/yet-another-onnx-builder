import unittest
import numpy as np
import onnx
from yobx.ext_test_case import ExtTestCase, requires_torch


class _FakeSymbolicTensor:
    """Mock object whose str() returns a _TYPENAME key, simulating a SymbolicTensor."""

    def __init__(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return self._name


@requires_torch("2.9")
class TestTorchDtypeToOnnxDtype(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch

    def setUp(self):
        from yobx.torch.torch_helper import torch_dtype_to_onnx_dtype

        self.convert = torch_dtype_to_onnx_dtype

    def test_float32(self):
        self.assertEqual(self.convert(self.torch.float32), onnx.TensorProto.FLOAT)

    def test_float16(self):
        self.assertEqual(self.convert(self.torch.float16), onnx.TensorProto.FLOAT16)

    def test_bfloat16(self):
        self.assertEqual(self.convert(self.torch.bfloat16), onnx.TensorProto.BFLOAT16)

    def test_float64(self):
        self.assertEqual(self.convert(self.torch.float64), onnx.TensorProto.DOUBLE)

    def test_int64(self):
        self.assertEqual(self.convert(self.torch.int64), onnx.TensorProto.INT64)

    def test_int32(self):
        self.assertEqual(self.convert(self.torch.int32), onnx.TensorProto.INT32)

    def test_uint64(self):
        self.assertEqual(self.convert(self.torch.uint64), onnx.TensorProto.UINT64)

    def test_uint32(self):
        self.assertEqual(self.convert(self.torch.uint32), onnx.TensorProto.UINT32)

    def test_bool(self):
        self.assertEqual(self.convert(self.torch.bool), onnx.TensorProto.BOOL)

    def test_symint(self):
        self.assertEqual(self.convert(self.torch.SymInt), onnx.TensorProto.INT64)

    def test_int16(self):
        self.assertEqual(self.convert(self.torch.int16), onnx.TensorProto.INT16)

    def test_uint16(self):
        self.assertEqual(self.convert(self.torch.uint16), onnx.TensorProto.UINT16)

    def test_int8(self):
        self.assertEqual(self.convert(self.torch.int8), onnx.TensorProto.INT8)

    def test_uint8(self):
        self.assertEqual(self.convert(self.torch.uint8), onnx.TensorProto.UINT8)

    def test_symfloat(self):
        self.assertEqual(self.convert(self.torch.SymFloat), onnx.TensorProto.FLOAT)

    def test_complex64(self):
        self.assertEqual(self.convert(self.torch.complex64), onnx.TensorProto.COMPLEX64)

    def test_complex128(self):
        self.assertEqual(self.convert(self.torch.complex128), onnx.TensorProto.COMPLEX128)

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


@requires_torch("2.9")
class TestToNumpy(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch

    def setUp(self):
        from yobx.helpers.torch_helper import to_numpy

        self.to_numpy = to_numpy

    def test_float32(self):
        t = self.torch.tensor([1.0, 2.0, 3.0], dtype=self.torch.float32)
        result = self.to_numpy(t)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqualArray(result, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_float64(self):
        t = self.torch.tensor([1.0, 2.0, 3.0], dtype=self.torch.float64)
        result = self.to_numpy(t)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)
        self.assertEqualArray(result, np.array([1.0, 2.0, 3.0], dtype=np.float64))

    def test_float16(self):
        t = self.torch.tensor([1.0, 2.0, 3.0], dtype=self.torch.float16)
        result = self.to_numpy(t)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float16)
        self.assertEqualArray(result, np.array([1.0, 2.0, 3.0], dtype=np.float16))

    def test_int64(self):
        t = self.torch.tensor([1, 2, 3], dtype=self.torch.int64)
        result = self.to_numpy(t)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.int64)
        self.assertEqualArray(result, np.array([1, 2, 3], dtype=np.int64))

    def test_int32(self):
        t = self.torch.tensor([1, 2, 3], dtype=self.torch.int32)
        result = self.to_numpy(t)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.int32)
        self.assertEqualArray(result, np.array([1, 2, 3], dtype=np.int32))

    def test_bool(self):
        t = self.torch.tensor([True, False, True], dtype=self.torch.bool)
        result = self.to_numpy(t)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.bool_)
        self.assertEqualArray(result, np.array([True, False, True], dtype=np.bool_))

    def test_bfloat16(self):
        try:
            import ml_dtypes
        except ImportError:
            self.skipTest("ml_dtypes not available")
        t = self.torch.tensor([1.0, 2.0, 3.0], dtype=self.torch.bfloat16)
        result = self.to_numpy(t)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, ml_dtypes.bfloat16)

    def test_2d_tensor(self):
        t = self.torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=self.torch.float32)
        result = self.to_numpy(t)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqualArray(result, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))


@requires_torch("2.9")
class TestOnnxDtypeToTorchDtype(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch

    def setUp(self):
        from yobx.helpers.torch_helper import onnx_dtype_to_torch_dtype

        self.convert = onnx_dtype_to_torch_dtype

    def test_float32(self):
        self.assertEqual(self.convert(onnx.TensorProto.FLOAT), self.torch.float32)

    def test_float16(self):
        self.assertEqual(self.convert(onnx.TensorProto.FLOAT16), self.torch.float16)

    def test_bfloat16(self):
        self.assertEqual(self.convert(onnx.TensorProto.BFLOAT16), self.torch.bfloat16)

    def test_float64(self):
        self.assertEqual(self.convert(onnx.TensorProto.DOUBLE), self.torch.float64)

    def test_int32(self):
        self.assertEqual(self.convert(onnx.TensorProto.INT32), self.torch.int32)

    def test_int64(self):
        self.assertEqual(self.convert(onnx.TensorProto.INT64), self.torch.int64)

    def test_uint32(self):
        self.assertEqual(self.convert(onnx.TensorProto.UINT32), self.torch.uint32)

    def test_uint64(self):
        self.assertEqual(self.convert(onnx.TensorProto.UINT64), self.torch.uint64)

    def test_bool(self):
        self.assertEqual(self.convert(onnx.TensorProto.BOOL), self.torch.bool)

    def test_int16(self):
        self.assertEqual(self.convert(onnx.TensorProto.INT16), self.torch.int16)

    def test_uint16(self):
        self.assertEqual(self.convert(onnx.TensorProto.UINT16), self.torch.uint16)

    def test_int8(self):
        self.assertEqual(self.convert(onnx.TensorProto.INT8), self.torch.int8)

    def test_uint8(self):
        self.assertEqual(self.convert(onnx.TensorProto.UINT8), self.torch.uint8)

    def test_complex64(self):
        self.assertEqual(self.convert(onnx.TensorProto.COMPLEX64), self.torch.complex64)

    def test_complex128(self):
        self.assertEqual(self.convert(onnx.TensorProto.COMPLEX128), self.torch.complex128)

    def test_unknown_raises(self):
        with self.assertRaises(NotImplementedError):
            self.convert(onnx.TensorProto.STRING)


if __name__ == "__main__":
    unittest.main(verbosity=2)
