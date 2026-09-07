"""Exercises incremental native inference instead of the removed Python engine."""

import unittest
import numpy as np
from onnx_light.onnx import TensorProto, helper, numpy_helper

from yobx.xshape import NativeShapeInference


class TestNativeInferenceRuntime(unittest.TestCase):
    def test_identity_type_shape_and_rank(self):
        for shape in ((3, 4), (None, None), (), ("N", 4)):
            with self.subTest(shape=shape):
                inference = NativeShapeInference()
                inference.set_type("X", TensorProto.FLOAT)
                inference.set_shape("X", shape)
                inference.run_node(helper.make_node("Identity", ["X"], ["Y"]), cost=False)
                self.assertEqual(inference.get_shape("Y"), shape)
                self.assertEqual(inference.get_type("Y"), TensorProto.FLOAT)

    def test_shape_and_size(self):
        for op, attrs, shape, value in (
            ("Shape", {}, (3,), (2, 3, 4)),
            ("Shape", {"start": 1}, (2,), (3, 4)),
            ("Shape", {"start": 1, "end": 2}, (1,), (3,)),
            ("Size", {}, (), (24,)),
        ):
            with self.subTest(op=op, attrs=attrs):
                inference = NativeShapeInference()
                inference.set_type("X", TensorProto.FLOAT)
                inference.set_shape("X", (2, 3, 4))
                inference.run_node(helper.make_node(op, ["X"], ["Y"], **attrs), cost=False)
                self.assertEqual(inference.get_type("Y"), TensorProto.INT64)
                self.assertEqual(inference.get_shape("Y"), shape)
                self.assertEqual(inference.value_as_shape("Y"), value)

    def test_reshape_shape_literals(self):
        for source, target, expected in (
            ((2, 3), [6], (6,)),
            ((2, 3), [-1], (6,)),
            ((2, 3), [0, -1], (2, 3)),
            ((2, 3), [0, 1, -1], (2, 1, 3)),
            (("N", 3), [0, 3], ("N", 3)),
        ):
            with self.subTest(source=source, target=target):
                inference = NativeShapeInference()
                inference.set_type("X", TensorProto.FLOAT)
                inference.set_shape("X", source)
                inference.set_constant(
                    "target", numpy_helper.from_array(np.array(target, dtype=np.int64))
                )
                inference.run_node(
                    helper.make_node("Reshape", ["X", "target"], ["Y"]), cost=False
                )
                self.assertEqual(inference.get_shape("Y"), expected)

    def test_constant_tensor_types_and_ranks(self):
        for value in (
            np.array([1, 2], dtype=np.int64),
            np.array(3, dtype=np.int64),
            np.array([1, 2], dtype=np.float32),
            np.ones((2, 3), dtype=np.float64),
        ):
            with self.subTest(dtype=value.dtype, shape=value.shape):
                tensor = numpy_helper.from_array(value)
                inference = NativeShapeInference()
                inference.set_constant("X", tensor)
                self.assertEqual(inference.get_type("X"), tensor.data_type)
                self.assertEqual(inference.get_shape("X"), value.shape)
                np.testing.assert_array_equal(inference.get_constant("X"), value)
                self.assertTrue(inference.is_constant("X"))

    def test_constant_scalar_attributes(self):
        for attributes, dtype in (
            ({"value_float": 1.5}, TensorProto.FLOAT),
            ({"value_int": 3}, TensorProto.INT64),
        ):
            with self.subTest(attributes=attributes):
                inference = NativeShapeInference()
                inference.run_node(
                    helper.make_node("Constant", [], ["Y"], **attributes), cost=False
                )
                self.assertEqual(inference.get_type("Y"), dtype)
                self.assertEqual(inference.get_shape("Y"), ())

    def test_constant_of_shape(self):
        for dtype in (TensorProto.FLOAT, TensorProto.INT64):
            with self.subTest(dtype=dtype):
                inference = NativeShapeInference()
                inference.set_constant(
                    "shape", numpy_helper.from_array(np.array([2, 3], dtype=np.int64))
                )
                value = helper.make_tensor("value", dtype, [1], [0])
                inference.run_node(
                    helper.make_node("ConstantOfShape", ["shape"], ["Y"], value=value), cost=False
                )
                self.assertEqual(inference.get_shape("Y"), (2, 3))
                self.assertEqual(inference.get_type("Y"), dtype)

    def test_gather_elements(self):
        inference = NativeShapeInference()
        inference.set_type("X", TensorProto.FLOAT)
        inference.set_shape("X", (3, 4))
        inference.set_type("indices", TensorProto.INT64)
        inference.set_shape("indices", (2, 4))
        inference.run_node(
            helper.make_node("GatherElements", ["X", "indices"], ["Y"]), cost=False
        )
        self.assertEqual(inference.get_shape("Y"), (2, 4))

    def test_unavailable_literals_and_unsupported_nodes(self):
        inference = NativeShapeInference()
        self.assertFalse(inference.is_constant("unknown"))
        self.assertIsNone(inference.get_constant("unknown", exc=False))
        with self.assertRaises(KeyError):
            inference.get_constant("unknown")
        with self.assertRaises(TypeError):
            inference.set_constant("X", np.array([1]))
        with self.assertRaises(ValueError):
            inference.run_node(
                helper.make_node("Unsupported", [], ["Y"], domain="custom"), cost=False
            )

    def test_device_metadata_is_not_shape_inference(self):
        inference = NativeShapeInference()
        inference.set_device("X", 0)
        self.assertTrue(inference.has_device("X"))
        self.assertEqual(inference.get_device("X"), 0)
        self.assertFalse(inference.has_shape("X"))


if __name__ == "__main__":
    unittest.main()
