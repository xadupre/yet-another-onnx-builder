import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_torch
from yobx.xshape._shape_helper import (
    reshape_implementation_with_zero,
    all_int,
    all_float,
    all_int_or_float,
    all_int_or_str,
    is_static_shape,
    is_static_dimension,
    compatible_shapes,
    compatible_dimensions,
    _reshape_shape,
)


class TestShapeHelper(ExtTestCase):
    def test_all_int(self):
        self.assertTrue(all_int([1, 2, 3]))
        self.assertFalse(all_int([1, "a", 3]))
        self.assertFalse(all_int([1.0, 2]))
        self.assertTrue(all_int(()))

    def test_all_float(self):
        self.assertTrue(all_float([1.0, 2.5]))
        self.assertFalse(all_float([1.0, 2]))
        self.assertTrue(all_float(()))

    def test_all_int_or_float(self):
        self.assertTrue(all_int_or_float([1, 2.0, 3]))
        self.assertFalse(all_int_or_float([1, "a"]))

    def test_all_int_or_str(self):
        self.assertTrue(all_int_or_str([1, "a", 3]))
        self.assertFalse(all_int_or_str([1.0, 2]))

    def test_is_static_shape_none(self):
        self.assertFalse(is_static_shape(None))

    def test_is_static_shape_static(self):
        self.assertTrue(is_static_shape((3, 4, 5)))

    def test_is_static_shape_dynamic(self):
        self.assertFalse(is_static_shape((3, "batch", 5)))

    def test_is_static_dimension_int(self):
        self.assertTrue(is_static_dimension(5))

    def test_is_static_dimension_str(self):
        self.assertFalse(is_static_dimension("batch"))

    @requires_torch("2.0")
    def test_is_static_dimension_dim(self):
        import torch

        d = torch.export.Dim("batch", min=2, max=10)
        self.assertFalse(is_static_dimension(d))

    def test_compatible_shapes_same(self):
        self.assertTrue(compatible_shapes((1, 2), (1, 2)))

    def test_compatible_shapes_dynamic(self):
        self.assertTrue(compatible_shapes((1, 2), (1, "D2")))

    def test_compatible_shapes_different_rank(self):
        self.assertFalse(compatible_shapes((1, 2), (1, 2, 3)))

    def test_compatible_shapes_incompatible(self):
        self.assertFalse(compatible_shapes(("D2", 2), (1, "D2")))

    def test_compatible_shapes_compatible_str(self):
        self.assertTrue(compatible_shapes(("D2", 2), (2, "D2")))

    def test_compatible_dimensions_equal(self):
        self.assertTrue(compatible_dimensions(1, 1))

    def test_compatible_dimensions_different(self):
        self.assertFalse(compatible_dimensions(1, 2))

    def test_compatible_dimensions_with_str(self):
        self.assertTrue(compatible_dimensions(1, "D"))

    def test_compatible_dimensions_multiple_str(self):
        self.assertTrue(compatible_dimensions(1, "D", "DD"))

    def test_reshape_shape_no_minus_one(self):
        self.assertEqual(_reshape_shape((2, 3, 4), (6, 4)), (6, 4))

    def test_reshape_shape_minus_one(self):
        self.assertEqual(_reshape_shape((2, 3, 4), (6, -1)), (6, 4))

    def test_reshape_shape_only_minus_one(self):
        self.assertEqual(_reshape_shape((2, 3, 4), (-1,)), (24,))

    def test_reshape_implementation_with_zero_numpy(self):
        data = np.arange(24).reshape(2, 3, 4)
        result = reshape_implementation_with_zero(data, (6, 4))
        self.assertEqual(result.shape, (6, 4))

    def test_reshape_implementation_with_zero_zeros(self):
        data = np.arange(24).reshape(2, 3, 4)
        # shape=(0, 3, 4) means keep first dim from data => 2
        result = reshape_implementation_with_zero(data, (0, 3, 4))
        self.assertEqual(result.shape, (2, 3, 4))

    def test_reshape_implementation_squeeze(self):
        data = np.arange(6).reshape(1, 6)
        result = reshape_implementation_with_zero(data, ())
        self.assertEqual(result.shape, (6,))

    def test_reshape_implementation_allowzero(self):
        data = np.arange(0).reshape(0, 3)
        result = reshape_implementation_with_zero(data, (0, 3), allowzero=1)
        self.assertEqual(result.shape, (0, 3))


class TestCheckTwoShapesCompatibility(ExtTestCase):
    """Compares shape compatibility with native inferred output annotations."""

    def _infer_identity(self, input_shape, output_shape):
        from onnx_light.onnx import TensorProto, helper
        from yobx.xshape import NativeShapeInference

        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Identity", ["X"], ["Y"])],
                "compatibility",
                [helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)],
                [helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )
        inference = NativeShapeInference()
        inference.run_model(model)
        return inference

    def test_compare_both_static_compatible(self):
        """Preserves matching static output annotations."""
        self.assertTrue(compatible_shapes((1, 2), (1, 2)))
        inference = self._infer_identity((1, 2), (1, 2))
        self.assertEqual(inference.get_shape("Y"), (1, 2))

    def test_compare_both_static_incompatible(self):
        """Replaces a stale static annotation with the native inferred shape."""
        self.assertFalse(compatible_shapes((1, 2), (1, 3)))
        inference = self._infer_identity((1, 2), (1, 3))
        self.assertEqual(inference.get_shape("Y"), (1, 2))

    def test_compare_int_str_compatible(self):
        """Resolves a symbolic output annotation against a static dimension."""
        self.assertTrue(compatible_shapes((1, 2), (1, "D")))
        inference = self._infer_identity((1, 2), (1, "D"))
        self.assertEqual(inference.get_shape("Y"), (1, 2))
        self.assertIn(2, inference.get_registered_constraints()["D"])

    def test_compare_same_dynamic_name(self):
        """Preserves identical symbolic dimension names."""
        self.assertTrue(compatible_shapes(("D", 2), ("D", 2)))
        inference = self._infer_identity(("D", 2), ("D", 2))
        self.assertEqual(inference.get_shape("Y"), ("D", 2))

    def test_compare_different_dynamic_names(self):
        """Records native equality constraints between different symbolic names."""
        self.assertFalse(compatible_shapes(("D1",), ("D2",)))
        inference = self._infer_identity(("D1",), ("D2",))
        self.assertTrue(inference.context.has_constraint("D1", "D2"))

    def test_compare_rank_mismatch(self):
        """Replaces an output annotation whose rank disagrees with inference."""
        self.assertFalse(compatible_shapes((1, 2), (1, 2, 3)))
        inference = self._infer_identity((1, 2), (1, 2, 3))
        self.assertEqual(inference.get_shape("Y"), (1, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
