import unittest
from onnx import TensorProto
from yobx.ext_test_case import ExtTestCase
from yobx.xbuilder._virtual_tensor import VirtualTensor

TFLOAT = TensorProto.FLOAT
TINT64 = TensorProto.INT64


class TestVirtualTensor(ExtTestCase):
    def test_init_basic(self):
        vt = VirtualTensor("x", TFLOAT, (2, 3))
        self.assertEqual(vt.name, "x")
        self.assertEqual(vt.dtype, TFLOAT)
        self.assertEqual(vt.shape, (2, 3))
        self.assertIsNone(vt.device)

    def test_init_with_device(self):
        vt = VirtualTensor("y", TINT64, (4,), device=0)
        self.assertEqual(vt.name, "y")
        self.assertEqual(vt.dtype, TINT64)
        self.assertEqual(vt.shape, (4,))
        self.assertEqual(vt.device, 0)

    def test_init_string_shape(self):
        vt = VirtualTensor("z", TFLOAT, ("batch", "seq"))
        self.assertEqual(vt.shape, ("batch", "seq"))

    def test_init_invalid_device_raises(self):
        with self.assertRaises(AssertionError):
            VirtualTensor("x", TFLOAT, (2, 3), device="cpu")

    def test_repr_without_device(self):
        vt = VirtualTensor("x", TFLOAT, (2, 3))
        r = repr(vt)
        self.assertIn("VirtualTensor", r)
        self.assertIn("'x'", r)
        self.assertIn("(2, 3)", r)
        self.assertNotIn("device", r)

    def test_repr_with_device(self):
        vt = VirtualTensor("x", TFLOAT, (2, 3), device=1)
        r = repr(vt)
        self.assertIn("VirtualTensor", r)
        self.assertIn("'x'", r)
        self.assertIn("(2, 3)", r)
        self.assertIn("device=1", r)

    def test_get_device_none(self):
        vt = VirtualTensor("x", TFLOAT, (2, 3))
        self.assertIsNone(vt.get_device())

    def test_get_device_value(self):
        vt = VirtualTensor("x", TFLOAT, (2, 3), device=2)
        self.assertEqual(vt.get_device(), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
