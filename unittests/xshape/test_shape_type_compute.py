import unittest
from yobx.ext_test_case import ExtTestCase
from yobx.xshape.shape_type_compute import broadcast_shape


class TestShapeTypeCompute(ExtTestCase):
    def test_broadcast_shape_equal(self):
        self.assertEqual(broadcast_shape((3, 4), (3, 4)), (3, 4))

    def test_broadcast_shape_empty_first(self):
        self.assertEqual(broadcast_shape((), (3, 4)), (3, 4))

    def test_broadcast_shape_empty_second(self):
        self.assertEqual(broadcast_shape((3, 4), ()), (3, 4))

    def test_broadcast_shape_scalar_first(self):
        self.assertEqual(broadcast_shape((1,), (3, 4)), (3, 4))

    def test_broadcast_shape_scalar_second(self):
        self.assertEqual(broadcast_shape((3, 4), (1,)), (3, 4))

    def test_broadcast_shape_extend_rank(self):
        # (4,) broadcasts to (3, 4)
        self.assertEqual(broadcast_shape((4,), (3, 4)), (3, 4))

    def test_broadcast_shape_with_ones(self):
        self.assertEqual(broadcast_shape((1, 4), (3, 1)), (3, 4))

    def test_broadcast_shape_dynamic(self):
        result = broadcast_shape((1, "seq"), ("batch", "seq"))
        self.assertEqual(result, ("batch", "seq"))

    def test_broadcast_shape_zero(self):
        result = broadcast_shape((0, 4), (3, 4))
        self.assertEqual(result, (0, 4))

    def test_broadcast_shape_dynamic_both(self):
        result = broadcast_shape(("a", "b"), ("a", "b"))
        self.assertEqual(result, ("a", "b"))

    def test_broadcast_shape_int_overrides_one(self):
        # int=5 vs int=1 => 5
        self.assertEqual(broadcast_shape((5,), (1,)), (5,))
        self.assertEqual(broadcast_shape((1,), (5,)), (5,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
