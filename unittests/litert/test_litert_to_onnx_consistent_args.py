"""
Unit tests verifying that :func:`yobx.litert.to_onnx` accepts the full
range of supported arg types: numpy arrays, numpy dtypes / scalar types,
and :class:`onnx.ValueInfoProto` descriptors.

These tests use the normalize_input_arg helper directly to avoid needing
a real TFLite model file.
"""

import unittest
import numpy as np
import onnx
from yobx.ext_test_case import ExtTestCase
from yobx.helpers.onnx_helper import normalize_input_arg


class TestLiteRTNormalizeInputArg(ExtTestCase):
    """normalize_input_arg works correctly for litert-style arg processing."""

    def test_numpy_array_passthrough(self):
        arr = np.ones((1, 4), dtype=np.float32)
        result = normalize_input_arg(arr)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqualArray(arr, result)

    def test_numpy_scalar_type(self):
        result = normalize_input_arg(np.float32)
        self.assertIsInstance(result, onnx.ValueInfoProto)
        self.assertEqual(result.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)

    def test_numpy_dtype(self):
        result = normalize_input_arg(np.dtype("float32"))
        self.assertIsInstance(result, onnx.ValueInfoProto)
        self.assertEqual(result.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)

    def test_value_info_proto_passthrough(self):
        vip = onnx.helper.make_tensor_value_info("inp", onnx.TensorProto.FLOAT, [1, 4])
        result = normalize_input_arg(vip)
        self.assertIs(result, vip)

    def test_tuple_normalize(self):
        """normalize_input_arg applied to a whole args-tuple works."""
        arr = np.ones((2, 3), dtype=np.float32)
        vip = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.INT64, [None])
        args = (arr, np.float64, np.dtype("int32"), vip)
        normalized = tuple(normalize_input_arg(a, idx=i) for i, a in enumerate(args))
        self.assertIsInstance(normalized[0], np.ndarray)
        self.assertIsInstance(normalized[1], onnx.ValueInfoProto)
        self.assertEqual(
            normalized[1].type.tensor_type.elem_type, onnx.TensorProto.DOUBLE
        )
        self.assertIsInstance(normalized[2], onnx.ValueInfoProto)
        self.assertEqual(
            normalized[2].type.tensor_type.elem_type, onnx.TensorProto.INT32
        )
        self.assertIs(normalized[3], vip)


if __name__ == "__main__":
    unittest.main(verbosity=2)
