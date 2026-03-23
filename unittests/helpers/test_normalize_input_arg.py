"""
Unit tests for :func:`yobx.helpers.onnx_helper.normalize_input_arg`.

Verifies that all supported arg types are correctly normalized to either
a :class:`numpy.ndarray` or an :class:`onnx.ValueInfoProto`.
"""

import unittest
import numpy as np
import onnx
from yobx.ext_test_case import ExtTestCase
from yobx.helpers.onnx_helper import normalize_input_arg


class TestNormalizeInputArg(ExtTestCase):
    """Tests for :func:`normalize_input_arg`."""

    # ------------------------------------------------------------------
    # numpy.ndarray — must be returned unchanged
    # ------------------------------------------------------------------

    def test_numpy_array_returned_as_is(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = normalize_input_arg(arr)
        self.assertIs(result, arr)

    def test_numpy_array_int64(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = normalize_input_arg(arr)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.int64)

    # ------------------------------------------------------------------
    # numpy scalar types (e.g. np.float32) → ValueInfoProto
    # ------------------------------------------------------------------

    def test_numpy_scalar_type_float32(self):
        result = normalize_input_arg(np.float32)
        self.assertIsInstance(result, onnx.ValueInfoProto)
        self.assertEqual(result.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
        # No shape should be set (unranked)
        self.assertFalse(result.type.tensor_type.HasField("shape"))

    def test_numpy_scalar_type_float64(self):
        result = normalize_input_arg(np.float64)
        self.assertIsInstance(result, onnx.ValueInfoProto)
        self.assertEqual(result.type.tensor_type.elem_type, onnx.TensorProto.DOUBLE)

    def test_numpy_scalar_type_int64(self):
        result = normalize_input_arg(np.int64)
        self.assertIsInstance(result, onnx.ValueInfoProto)
        self.assertEqual(result.type.tensor_type.elem_type, onnx.TensorProto.INT64)

    def test_numpy_scalar_type_int32(self):
        result = normalize_input_arg(np.int32)
        self.assertIsInstance(result, onnx.ValueInfoProto)
        self.assertEqual(result.type.tensor_type.elem_type, onnx.TensorProto.INT32)

    def test_numpy_scalar_type_bool(self):
        result = normalize_input_arg(np.bool_)
        self.assertIsInstance(result, onnx.ValueInfoProto)
        self.assertEqual(result.type.tensor_type.elem_type, onnx.TensorProto.BOOL)

    # ------------------------------------------------------------------
    # numpy.dtype instances → ValueInfoProto
    # ------------------------------------------------------------------

    def test_numpy_dtype_float32(self):
        result = normalize_input_arg(np.dtype("float32"))
        self.assertIsInstance(result, onnx.ValueInfoProto)
        self.assertEqual(result.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)

    def test_numpy_dtype_int64(self):
        result = normalize_input_arg(np.dtype("int64"))
        self.assertIsInstance(result, onnx.ValueInfoProto)
        self.assertEqual(result.type.tensor_type.elem_type, onnx.TensorProto.INT64)

    # ------------------------------------------------------------------
    # name / idx hints for auto-generated VIP names
    # ------------------------------------------------------------------

    def test_default_name_idx_zero(self):
        result = normalize_input_arg(np.float32, idx=0)
        self.assertEqual(result.name, "X")

    def test_default_name_idx_nonzero(self):
        result = normalize_input_arg(np.float32, idx=2)
        self.assertEqual(result.name, "X2")

    def test_explicit_name(self):
        result = normalize_input_arg(np.float32, name="my_input")
        self.assertEqual(result.name, "my_input")

    # ------------------------------------------------------------------
    # onnx.ValueInfoProto — must be returned unchanged
    # ------------------------------------------------------------------

    def test_value_info_proto_returned_as_is(self):
        vip = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [None, 3])
        result = normalize_input_arg(vip)
        self.assertIs(result, vip)

    # ------------------------------------------------------------------
    # Generic array-like fallback
    # ------------------------------------------------------------------

    def test_list_fallback(self):
        result = normalize_input_arg([1.0, 2.0, 3.0])
        self.assertIsInstance(result, np.ndarray)

    def test_tuple_fallback(self):
        result = normalize_input_arg((1, 2, 3))
        self.assertIsInstance(result, np.ndarray)


if __name__ == "__main__":
    unittest.main(verbosity=2)
