import unittest
from yobx.ext_test_case import ExtTestCase
from yobx.xexpressions.operations import (
    dim_add,
    dim_div,
    dim_max,
    dim_min,
    dim_mod,
    dim_mul,
    dim_multi_mul,
    dim_sub,
)


class TestDimOperations(ExtTestCase):
    # ------------------------------------------------------------------ dim_mul
    def test_dim_mul_int_int(self):
        self.assertEqual(dim_mul(3, 4), 12)

    def test_dim_mul_zero(self):
        self.assertEqual(dim_mul(0, 5), 0)

    def test_dim_mul_symbolic(self):
        # One or both operands symbolic → simplify_expression is called
        result = dim_mul("n", 2)
        self.assertIsInstance(result, str)
        self.assertIn("n", result)
        self.assertIn("2", result)

    def test_dim_mul_both_symbolic(self):
        result = dim_mul("a", "b")
        self.assertIsInstance(result, str)
        self.assertIn("a", result)
        self.assertIn("b", result)

    # --------------------------------------------------------------- dim_multi_mul
    def test_dim_multi_mul_all_int(self):
        self.assertEqual(dim_multi_mul(2, 3, 4), 24)

    def test_dim_multi_mul_single_int(self):
        self.assertEqual(dim_multi_mul(7), 7)

    def test_dim_multi_mul_with_symbolic(self):
        result = dim_multi_mul(2, "n", 3)
        self.assertIsInstance(result, str)
        self.assertIn("n", result)

    def test_dim_multi_mul_all_symbolic(self):
        result = dim_multi_mul("a", "b", "c")
        self.assertIsInstance(result, str)

    # ------------------------------------------------------------------ dim_add
    def test_dim_add_int_int(self):
        self.assertEqual(dim_add(3, 4), 7)

    def test_dim_add_symbolic(self):
        result = dim_add("n", 1)
        self.assertIsInstance(result, str)
        self.assertIn("n", result)
        self.assertIn("1", result)

    def test_dim_add_both_symbolic(self):
        result = dim_add("a", "b")
        self.assertIsInstance(result, str)
        self.assertIn("a", result)
        self.assertIn("b", result)

    # ------------------------------------------------------------------ dim_sub
    def test_dim_sub_int_int(self):
        self.assertEqual(dim_sub(10, 3), 7)

    def test_dim_sub_symbolic(self):
        result = dim_sub("n", 1)
        self.assertIsInstance(result, str)
        self.assertIn("n", result)

    def test_dim_sub_same_symbol(self):
        # n - n should simplify to 0
        result = dim_sub("n", "n")
        self.assertEqual(str(result), "0")

    def test_dim_sub_both_symbolic(self):
        result = dim_sub("a", "b")
        self.assertIsInstance(result, str)

    # ------------------------------------------------------------------ dim_div
    def test_dim_div_int_int_exact(self):
        self.assertEqual(dim_div(12, 4), 3)

    def test_dim_div_int_int_floor(self):
        # Uses integer floor division
        self.assertEqual(dim_div(7, 2), 3)

    def test_dim_div_symbolic(self):
        result = dim_div("2*n", 2)
        self.assertIsInstance(result, (str, int))
        # 2*n // 2 simplifies to n
        self.assertEqual(str(result), "n")

    def test_dim_div_both_symbolic(self):
        result = dim_div("a", "b")
        self.assertIsInstance(result, str)

    # ------------------------------------------------------------------ dim_mod
    def test_dim_mod_int_int(self):
        self.assertEqual(dim_mod(10, 3), 1)

    def test_dim_mod_exact(self):
        self.assertEqual(dim_mod(12, 4), 0)

    def test_dim_mod_symbolic(self):
        result = dim_mod("n", 2)
        self.assertIsInstance(result, str)
        self.assertIn("n", result)
        self.assertIn("2", result)

    def test_dim_mod_both_symbolic(self):
        result = dim_mod("a", "b")
        self.assertIsInstance(result, str)

    # ------------------------------------------------------------------ dim_max
    def test_dim_max_int_int_left_larger(self):
        self.assertEqual(dim_max(7, 3), 7)

    def test_dim_max_int_int_right_larger(self):
        self.assertEqual(dim_max(2, 9), 9)

    def test_dim_max_int_int_equal(self):
        self.assertEqual(dim_max(5, 5), 5)

    def test_dim_max_symbolic(self):
        result = dim_max("a", "b")
        self.assertIsInstance(result, str)
        # dim_max encodes max(a,b) as a^b via simplify_expression
        self.assertIn("a", result)
        self.assertIn("b", result)

    def test_dim_max_same_symbol(self):
        # max(n, n) → n
        result = dim_max("n", "n")
        self.assertEqual(str(result), "n")

    # ------------------------------------------------------------------ dim_min
    def test_dim_min_int_int_left_smaller(self):
        self.assertEqual(dim_min(2, 9), 2)

    def test_dim_min_int_int_right_smaller(self):
        self.assertEqual(dim_min(8, 3), 3)

    def test_dim_min_int_int_equal(self):
        self.assertEqual(dim_min(4, 4), 4)

    def test_dim_min_symbolic(self):
        result = dim_min("a", "b")
        self.assertIsInstance(result, str)
        self.assertIn("a", result)
        self.assertIn("b", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
