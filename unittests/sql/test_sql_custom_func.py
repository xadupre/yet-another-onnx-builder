"""
Unit tests for custom Python function support in SQL-to-ONNX conversion.

These tests verify that:
* :class:`~yobx.sql.parse.FuncCallExpr` is produced by the parser.
* Custom numpy functions are correctly traced via
  :func:`~yobx.xtracing.trace_numpy_function` and embedded as ONNX nodes.
* Custom functions can be used in both ``SELECT`` and ``WHERE`` clauses.
"""

import unittest
import numpy as np

from yobx.ext_test_case import ExtTestCase, has_onnxruntime
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sql import FuncCallExpr, sql_to_onnx
from yobx.xtracing.parse import ColumnRef, SelectOp, parse_sql


def _ort_run(onx, feeds):
    """Run *onx* with onnxruntime and return outputs. Skip if ORT is not installed."""
    from onnxruntime import InferenceSession

    sess = InferenceSession(onx.proto.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


class TestFuncCallExprParsing(unittest.TestCase):
    """Tests for parsing custom function calls in SQL."""

    def test_func_call_single_arg(self):
        pq = parse_sql("SELECT my_func(a) FROM t")
        sel = pq.operations[0]
        self.assertIsInstance(sel, SelectOp)
        item = sel.items[0]
        self.assertIsInstance(item.expr, FuncCallExpr)
        self.assertEqual(item.expr.func, "my_func")
        self.assertEqual(len(item.expr.args), 1)
        self.assertIsInstance(item.expr.args[0], ColumnRef)
        self.assertEqual(item.expr.args[0].column, "a")

    def test_func_call_two_args(self):
        pq = parse_sql("SELECT add2(a, b) FROM t")
        sel = pq.operations[0]
        item = sel.items[0]
        self.assertIsInstance(item.expr, FuncCallExpr)
        self.assertEqual(item.expr.func, "add2")
        self.assertEqual(len(item.expr.args), 2)

    def test_func_call_columns_collected(self):
        pq = parse_sql("SELECT my_func(a, b) AS r FROM t")
        self.assertIn("a", pq.columns)
        self.assertIn("b", pq.columns)

    def test_func_call_str_repr(self):
        pq = parse_sql("SELECT my_func(a) FROM t")
        item = pq.operations[0].items[0]
        self.assertIn("my_func", str(item.expr))

    def test_func_call_in_where(self):
        pq = parse_sql("SELECT a FROM t WHERE norm(a) > 0")
        from yobx.xtracing.parse import FilterOp

        filter_op = next(op for op in pq.operations if isinstance(op, FilterOp))
        cond = filter_op.condition
        self.assertIsInstance(cond.left, FuncCallExpr)
        self.assertEqual(cond.left.func, "norm")


class TestCustomFunctionSelect(ExtTestCase):
    """Tests for using custom Python functions in SELECT clauses."""

    def _run(self, query, dtypes, feeds, custom_functions=None):
        onx = sql_to_onnx(query, dtypes, custom_functions=custom_functions)
        ref = ExtendedReferenceEvaluator(onx)
        ref_outputs = ref.run(None, feeds)
        if has_onnxruntime():
            ort_outputs = _ort_run(onx, feeds)
            self.assertEqual(len(ref_outputs), len(ort_outputs))
            for ref_out, ort_out in zip(ref_outputs, ort_outputs):
                np.testing.assert_allclose(ort_out, ref_out, rtol=1e-5, atol=1e-6)
        return ref_outputs

    def test_sqrt_single_column(self):
        """SELECT my_sqrt(a) should apply np.sqrt to each element."""
        dtypes = {"a": np.float32}
        a = np.array([1.0, 4.0, 9.0], dtype=np.float32)
        (out,) = self._run(
            "SELECT my_sqrt(a) AS r FROM t",
            dtypes,
            {"a": a},
            custom_functions={"my_sqrt": np.sqrt},
        )
        self.assertEqualArray(out, np.sqrt(a), atol=1e-6)

    def test_abs_column(self):
        """SELECT my_abs(a) should apply np.abs element-wise."""
        dtypes = {"a": np.float32}
        a = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
        (out,) = self._run(
            "SELECT my_abs(a) AS r FROM t", dtypes, {"a": a}, custom_functions={"my_abs": np.abs}
        )
        self.assertEqualArray(out, np.abs(a), atol=1e-6)

    def test_two_arg_function(self):
        """SELECT add2(a, b) should call a custom two-argument function."""
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        def add2(x, y):
            return x + y

        (out,) = self._run(
            "SELECT add2(a, b) AS r FROM t",
            dtypes,
            {"a": a, "b": b},
            custom_functions={"add2": add2},
        )
        self.assertEqualArray(out, a + b, atol=1e-6)

    def test_composed_custom_function(self):
        """SELECT f(a) where f uses multiple numpy ops should be traced correctly."""
        dtypes = {"a": np.float32}
        a = np.array([1.0, 4.0, 9.0], dtype=np.float32)

        def f(x):
            return np.sqrt(np.abs(x) + np.float32(1))

        (out,) = self._run("SELECT f(a) AS r FROM t", dtypes, {"a": a}, custom_functions={"f": f})
        self.assertEqualArray(out, f(a), atol=1e-6)

    def test_custom_function_with_filter(self):
        """Custom function in SELECT combined with WHERE clause."""
        dtypes = {"a": np.float32}
        a = np.array([1.0, -2.0, 9.0], dtype=np.float32)
        (out,) = self._run(
            "SELECT my_sqrt(a) AS r FROM t WHERE a > 0",
            dtypes,
            {"a": a},
            custom_functions={"my_sqrt": np.sqrt},
        )
        expected = np.sqrt(a[a > 0])
        self.assertEqualArray(out, expected, atol=1e-6)

    def test_unknown_function_raises(self):
        """Calling an unregistered function should raise KeyError at conversion time."""
        dtypes = {"a": np.float32}
        with self.assertRaises(KeyError):
            sql_to_onnx("SELECT unknown_func(a) AS r FROM t", dtypes)


class TestCustomFunctionWhere(ExtTestCase):
    """Tests for using custom Python functions in WHERE clauses."""

    def _run(self, query, dtypes, feeds, custom_functions=None):
        onx = sql_to_onnx(query, dtypes, custom_functions=custom_functions)
        ref = ExtendedReferenceEvaluator(onx)
        ref_outputs = ref.run(None, feeds)
        if has_onnxruntime():
            ort_outputs = _ort_run(onx, feeds)
            self.assertEqual(len(ref_outputs), len(ort_outputs))
            for ref_out, ort_out in zip(ref_outputs, ort_outputs):
                np.testing.assert_allclose(ort_out, ref_out, rtol=1e-5, atol=1e-6)
        return ref_outputs

    def test_func_in_where_greater(self):
        """WHERE abs(a) > 1 should filter using the traced custom function."""
        dtypes = {"a": np.float32}
        a = np.array([0.5, -2.0, 1.5, -0.1], dtype=np.float32)
        (out,) = self._run(
            "SELECT a FROM t WHERE my_abs(a) > 1",
            dtypes,
            {"a": a},
            custom_functions={"my_abs": np.abs},
        )
        expected = a[np.abs(a) > 1]
        self.assertEqualArray(out, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
