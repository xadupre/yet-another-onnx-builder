import unittest
import torch
from yobx.ext_test_case import ExtTestCase, requires_torch
from yobx.torch.new_tracing.shape import TracingInt, TracingShape, TracingBool


@requires_torch("2.0")
class TestNewTracingShapes(ExtTestCase):
    # ------------------------------------------------------------------
    # TracingInt / TracingBool
    # ------------------------------------------------------------------

    def test_tracing_dimension_repr_no_value(self):
        d = TracingInt("batch")
        self.assertIn("batch", repr(d))
        self.assertEqual(str(d), "batch")

    def test_tracing_dimension_repr_with_value(self):
        d = TracingInt(4)
        self.assertIn("4", repr(d))
        self.assertEqual(int(d), 4)

    def test_tracing_dimension_int_raises_without_value(self):
        d = TracingInt("n")
        with self.assertRaises(ValueError):
            int(d)

    def test_tracing_dimension_eq(self):
        # Concrete comparison → plain bool
        d1 = TracingInt(4)
        d2 = TracingInt(4)
        d3 = TracingInt(8)
        self.assertEqual(d1, d2)
        self.assertNotEqual(d1, d3)
        self.assertEqual(d1, 4)  # compare concrete TracingInt with plain int

        # Symbolic comparison → TracingBool
        d_sym = TracingInt("batch")
        result = d_sym == 4
        self.assertIsInstance(result, TracingBool)
        self.assertIn("batch", str(result.value))

    def test_tracing_dimension_arithmetic(self):
        # Concrete arithmetic produces concrete TracingInt
        d = TracingInt(8)
        self.assertEqual(int(d + 2), 10)
        self.assertEqual(int(d - 3), 5)
        self.assertEqual(int(d * 2), 16)
        self.assertEqual(int(d // 4), 2)
        self.assertEqual(int(2 + d), 10)
        self.assertEqual(int(2 * d), 16)

        # Symbolic arithmetic preserves the expression as a string value
        s = TracingInt("n")
        result = s + 2
        self.assertIsInstance(result, TracingInt)
        self.assertIn("n", str(result.value))

    def test_tracing_dimension_neg(self):
        from yobx.torch.new_tracing.shape import TracingInt

        d = TracingInt(5)
        negd = -d
        self.assertEqual(int(negd), -5)

    def test_tracing_dimension_hash(self):
        d = TracingInt("batch")
        self.assertIsInstance(hash(d), int)
        self.assertIn(d, {d})

    def test_tracing_bool_concrete(self):
        tb_true = TracingBool(True)
        tb_false = TracingBool(False)
        self.assertTrue(bool(tb_true))
        self.assertFalse(bool(tb_false))

    def test_tracing_bool_symbolic_raises_on_bool(self):
        tb = TracingBool("(n==4)")
        with self.assertRaises(ValueError):
            bool(tb)

    # ------------------------------------------------------------------
    # TracingInt comparison operators
    # ------------------------------------------------------------------

    def test_tracing_int_gt_concrete(self):
        """Concrete TracingInt > int returns plain bool."""
        self.assertIs(TracingInt(5) > 3, True)
        self.assertIs(TracingInt(3) > 5, False)
        self.assertIs(TracingInt(5) > 5, False)

    def test_tracing_int_ge_concrete(self):
        """Concrete TracingInt >= int returns plain bool."""
        self.assertIs(TracingInt(5) >= 5, True)
        self.assertIs(TracingInt(4) >= 5, False)

    def test_tracing_int_lt_concrete(self):
        """Concrete TracingInt < int returns plain bool."""
        self.assertIs(TracingInt(3) < 5, True)
        self.assertIs(TracingInt(5) < 3, False)

    def test_tracing_int_le_concrete(self):
        """Concrete TracingInt <= int returns plain bool."""
        self.assertIs(TracingInt(5) <= 5, True)
        self.assertIs(TracingInt(6) <= 5, False)

    def test_tracing_int_ne_concrete(self):
        """Concrete TracingInt != int returns plain bool."""
        self.assertIs(TracingInt(5) != 0, True)
        self.assertIs(TracingInt(0) != 0, False)

    def test_tracing_int_gt_symbolic(self):
        """Symbolic TracingInt > int returns TracingBool."""
        result = TracingInt("batch") > 0
        self.assertIsInstance(result, TracingBool)
        self.assertIn("batch", result.value)
        self.assertIn(">", result.value)

    def test_tracing_int_ge_symbolic(self):
        """Symbolic TracingInt >= int returns TracingBool."""
        result = TracingInt("batch") >= 1
        self.assertIsInstance(result, TracingBool)
        self.assertIn("batch", result.value)

    def test_tracing_int_lt_symbolic(self):
        """Symbolic TracingInt < int returns TracingBool."""
        result = TracingInt("n") < 10
        self.assertIsInstance(result, TracingBool)
        self.assertIn("n", result.value)

    def test_tracing_int_le_symbolic(self):
        """Symbolic TracingInt <= int returns TracingBool."""
        result = TracingInt("n") <= 10
        self.assertIsInstance(result, TracingBool)
        self.assertIn("n", result.value)

    def test_tracing_int_ne_symbolic(self):
        """Symbolic TracingInt != int returns TracingBool."""
        result = TracingInt("batch") != 0
        self.assertIsInstance(result, TracingBool)
        self.assertIn("batch", result.value)

    # ------------------------------------------------------------------
    # Conditions registry — register_condition / TracingBool.__bool__
    # ------------------------------------------------------------------

    def test_register_condition_resolves_tracing_bool(self):
        """A registered TracingBool condition resolves to True in __bool__."""
        from yobx.torch.new_tracing.shape import register_condition, clear_conditions

        clear_conditions()
        tb = TracingInt("batch") > 0
        self.assertIsInstance(tb, TracingBool)

        # Before registering, __bool__ raises.
        with self.assertRaises(ValueError):
            bool(tb)

        register_condition(tb)

        # After registering, __bool__ returns True.
        self.assertTrue(bool(tb))

        # Cleanup.
        clear_conditions()

    def test_clear_conditions_removes_all(self):
        """clear_conditions empties the registry."""
        from yobx.torch.new_tracing.shape import (
            register_condition,
            clear_conditions,
            _known_true_conditions,
        )

        clear_conditions()
        register_condition(TracingInt("x") > 0)
        self.assertGreater(len(_known_true_conditions), 0)
        clear_conditions()
        self.assertEqual(len(_known_true_conditions), 0)

    def test_unregistered_condition_still_raises(self):
        """A symbolic TracingBool not in the registry still raises ValueError."""
        from yobx.torch.new_tracing.shape import clear_conditions

        clear_conditions()
        tb = TracingInt("unknown") >= 1
        self.assertIsInstance(tb, TracingBool)
        with self.assertRaises(ValueError):
            bool(tb)

    # ------------------------------------------------------------------
    # TracingShape
    # ------------------------------------------------------------------

    def test_tracing_shape_concrete(self):

        s = TracingShape([TracingInt(4), 16])
        self.assertTrue(s.is_concrete)
        self.assertEqual(s.numel(), 64)
        self.assertEqual(s.to_torch_size(), torch.Size([4, 16]))

    def test_tracing_shape_symbolic(self):
        s = TracingShape([TracingInt("n"), 8])
        self.assertFalse(s.is_concrete)
        with self.assertRaises(ValueError):
            s.numel()
        with self.assertRaises(ValueError):
            s.to_torch_size()

    def test_tracing_shape_repr(self):
        s = TracingShape([TracingInt(2), 4])
        r = repr(s)
        self.assertIn("TracingShape", r)

    def test_tracing_shape_indexing(self):
        d = TracingInt(5)
        s = TracingShape([d, 8])
        self.assertIs(s[0], d)
        self.assertEqual(s[1], 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
