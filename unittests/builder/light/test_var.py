"""
Unit tests for :class:`yobx.builder.light.Var` focusing on ``__str__``,
``__repr__`` and all Python operator overloads.
"""

import unittest

from onnx import TensorProto

from yobx.builder.light import Var, Vars, start
from yobx.ext_test_case import ExtTestCase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _x():
    """Return a (OnnxGraph, Var) pair with a single float input ``X``."""
    gr = start()
    return gr, gr.vin("X")


def _xy():
    """Return a (OnnxGraph, Var, Var) triple with float inputs ``X`` and ``Y``."""
    gr = start()
    return gr, gr.vin("X"), gr.vin("Y")


# ---------------------------------------------------------------------------
# __str__ / __repr__
# ---------------------------------------------------------------------------


class TestVarStr(ExtTestCase):
    """Tests for ``Var.__str__`` and ``Var.__repr__``."""

    def test_str_name_only(self):
        gr = start()
        x = Var(gr, "X", elem_type=None, shape=None)
        self.assertEqual(str(x), "X")

    def test_str_with_elem_type(self):
        gr = start()
        x = gr.vin("X", elem_type=TensorProto.FLOAT)
        # TensorProto.FLOAT == 1
        self.assertEqual(str(x), f"X:{TensorProto.FLOAT}")

    def test_str_with_elem_type_and_shape(self):
        gr = start()
        x = gr.vin("X", elem_type=TensorProto.FLOAT, shape=(2, 3))
        self.assertEqual(str(x), f"X:{TensorProto.FLOAT}:[2, 3]")

    def test_str_with_shape_no_elem_type(self):
        # elem_type=None but shape provided: only shape suffix
        gr = start()
        x = Var(gr, "X", elem_type=None, shape=(4,))
        self.assertEqual(str(x), "X:[4]")

    def test_str_after_rename(self):
        gr = start()
        x = gr.vin("X")
        x.rename("Z")
        self.assertEqual(str(x), f"Z:{TensorProto.FLOAT}")

    def test_repr(self):
        gr = start()
        x = gr.vin("X")
        self.assertEqual(repr(x), "Var('X')")

    def test_repr_after_rename(self):
        gr = start()
        x = gr.vin("X")
        x.rename("NewName")
        self.assertEqual(repr(x), "Var('NewName')")

    def test_vars_repr(self):
        gr = start()
        x = gr.vin("X")
        y = gr.vin("Y")
        vs = Vars(gr, x, y)
        self.assertIn("Var('X')", repr(vs))
        self.assertIn("Var('Y')", repr(vs))


# ---------------------------------------------------------------------------
# Operator overloads
# ---------------------------------------------------------------------------


class TestVarOperators(ExtTestCase):
    """Tests for each Python operator overload on :class:`Var`."""

    def _last_op(self, gr):
        """Return the ``op_type`` of the last node added to *gr*."""
        return gr.nodes[-1].op_type

    # --- arithmetic ---

    def test_truediv(self):
        gr, x, y = _xy()
        result = x / y
        self.assertIsInstance(result, Var)
        self.assertEqual(self._last_op(gr), "Div")

    def test_matmul(self):
        gr, x, y = _xy()
        result = x @ y
        self.assertIsInstance(result, Var)
        self.assertEqual(self._last_op(gr), "MatMul")

    def test_abs(self):
        gr, x = _x()
        result = abs(x)
        self.assertIsInstance(result, Var)
        self.assertEqual(self._last_op(gr), "Abs")

    def test_mod(self):
        gr, x, y = _xy()
        result = x % y
        self.assertIsInstance(result, Var)
        self.assertEqual(self._last_op(gr), "Mod")

    def test_pow(self):
        gr, x, y = _xy()
        result = x**y
        self.assertIsInstance(result, Var)
        self.assertEqual(self._last_op(gr), "Pow")

    # --- comparison ---

    def test_eq(self):
        gr, x, y = _xy()
        result = x == y
        self.assertIsInstance(result, Var)
        self.assertEqual(self._last_op(gr), "Equal")

    def test_ne(self):
        gr, x, y = _xy()
        result = x != y
        self.assertIsInstance(result, Var)
        # __ne__ builds Equal then Not
        self.assertEqual(self._last_op(gr), "Not")

    def test_lt(self):
        gr, x, y = _xy()
        result = x < y
        self.assertIsInstance(result, Var)
        self.assertEqual(self._last_op(gr), "Less")

    def test_le(self):
        gr, x, y = _xy()
        result = x <= y
        self.assertIsInstance(result, Var)
        self.assertEqual(self._last_op(gr), "LessOrEqual")

    def test_gt(self):
        gr, x, y = _xy()
        result = x > y
        self.assertIsInstance(result, Var)
        self.assertEqual(self._last_op(gr), "Greater")

    def test_ge(self):
        gr, x, y = _xy()
        result = x >= y
        self.assertIsInstance(result, Var)
        self.assertEqual(self._last_op(gr), "GreaterOrEqual")

    # --- chaining: result can be used in a further model ---

    def test_truediv_builds_valid_model(self):
        import onnx

        gr, x, y = _xy()
        (x / y).rename("Z").vout()
        onx = gr.to_onnx()
        self.assertIsInstance(onx, onnx.ModelProto)
        self.assertEqual(onx.graph.node[0].op_type, "Div")

    def test_matmul_builds_valid_model(self):
        import onnx

        gr, x, y = _xy()
        (x @ y).rename("Z").vout()
        onx = gr.to_onnx()
        self.assertIsInstance(onx, onnx.ModelProto)
        self.assertEqual(onx.graph.node[0].op_type, "MatMul")

    def test_pow_builds_valid_model(self):
        import onnx

        gr, x, y = _xy()
        (x**y).rename("Z").vout()
        onx = gr.to_onnx()
        self.assertIsInstance(onx, onnx.ModelProto)
        self.assertEqual(onx.graph.node[0].op_type, "Pow")


if __name__ == "__main__":
    unittest.main(verbosity=2)
