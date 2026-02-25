import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
