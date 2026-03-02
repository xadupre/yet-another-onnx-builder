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
    """Tests comparing compatible_shapes and _check_two_shapes_are_compatible."""

    def _make_builder(self):
        from yobx.xbuilder import GraphBuilder

        return GraphBuilder(18, ir_version=9, as_function=True)

    @requires_torch()
    def test_compare_both_static_compatible(self):
        """Both functions agree: identical static shapes are compatible."""
        self.assertTrue(compatible_shapes((1, 2), (1, 2)))
        gr = self._make_builder()
        # Should not raise
        gr._check_two_shapes_are_compatible((1, 2), (1, 2))

    @requires_torch()
    def test_compare_both_static_incompatible(self):
        """Both functions agree: different static shapes are incompatible."""
        self.assertFalse(compatible_shapes((1, 2), (1, 3)))
        gr = self._make_builder()
        self.assertRaises(
            AssertionError, gr._check_two_shapes_are_compatible, (1, 2), (1, 3)
        )

    @requires_torch()
    def test_compare_int_str_compatible(self):
        """Both functions agree: mixing static and dynamic is compatible."""
        self.assertTrue(compatible_shapes((1, 2), (1, "D")))
        gr = self._make_builder()
        # Should not raise
        gr._check_two_shapes_are_compatible((1, 2), (1, "D"))

    @requires_torch()
    def test_compare_same_dynamic_name(self):
        """Both functions agree: same dynamic name is compatible."""
        self.assertTrue(compatible_shapes(("D", 2), ("D", 2)))
        gr = self._make_builder()
        # Should not raise
        gr._check_two_shapes_are_compatible(("D", 2), ("D", 2))

    @requires_torch()
    def test_compare_different_dynamic_names(self):
        """
        Key behavioral difference: compatible_shapes returns False for two
        different string names in the same position, while
        _check_two_shapes_are_compatible registers constraints and does not raise.
        """
        self.assertFalse(compatible_shapes(("D1",), ("D2",)))
        gr = self._make_builder()
        # _check_two_shapes_are_compatible does NOT raise; it registers a constraint
        gr._check_two_shapes_are_compatible(("D1",), ("D2",))

    @requires_torch()
    def test_compare_rank_mismatch(self):
        """Both functions agree: rank mismatch is incompatible."""
        self.assertFalse(compatible_shapes((1, 2), (1, 2, 3)))
        gr = self._make_builder()
        self.assertRaises(
            AssertionError, gr._check_two_shapes_are_compatible, (1, 2), (1, 2, 3)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
