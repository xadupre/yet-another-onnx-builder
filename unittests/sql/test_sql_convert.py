"""
Unit tests for :mod:`yobx.sql.convert` — SQL-to-ONNX converter.
"""

import unittest
import numpy as np

from yobx.ext_test_case import ExtTestCase
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sql import sql_to_onnx


class TestSqlToOnnxSelect(ExtTestCase):
    """Tests for the SELECT clause conversion."""

    def _run(self, query, dtypes, feeds, *, right_dtypes=None):
        """Convert *query*, run it, return outputs."""
        onx = sql_to_onnx(query, dtypes, right_input_dtypes=right_dtypes)
        ref = ExtendedReferenceEvaluator(onx)
        return ref.run(None, feeds)

    # ------------------------------------------------------------------
    # Column pass-through
    # ------------------------------------------------------------------

    def test_select_single_column(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (out,) = self._run("SELECT a FROM t", dtypes, {"a": a})
        self.assertEqualArray(out, a)

    def test_select_multiple_columns(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        out_a, out_b = self._run("SELECT a, b FROM t", dtypes, {"a": a, "b": b})
        self.assertEqualArray(out_a, a)
        self.assertEqualArray(out_b, b)

    def test_select_integer_column(self):
        dtypes = {"x": np.int64}
        x = np.array([10, 20, 30], dtype=np.int64)
        (out,) = self._run("SELECT x FROM t", dtypes, {"x": x})
        self.assertEqualArray(out, x)

    # ------------------------------------------------------------------
    # Arithmetic expressions
    # ------------------------------------------------------------------

    def test_select_add(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = self._run("SELECT a + b AS total FROM t", dtypes, {"a": a, "b": b})
        self.assertEqualArray(total, a + b, atol=1e-6)

    def test_select_subtract(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (diff,) = self._run("SELECT a - b AS diff FROM t", dtypes, {"a": a, "b": b})
        self.assertEqualArray(diff, a - b, atol=1e-6)

    def test_select_multiply(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        (prod,) = self._run("SELECT a * b AS prod FROM t", dtypes, {"a": a, "b": b})
        self.assertEqualArray(prod, a * b, atol=1e-6)

    def test_select_divide(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        b = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        (quot,) = self._run("SELECT a / b AS quot FROM t", dtypes, {"a": a, "b": b})
        self.assertEqualArray(quot, a / b, atol=1e-6)

    def test_select_mixed_expressions(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        out_a, total = self._run(
            "SELECT a, a + b AS total FROM t", dtypes, {"a": a, "b": b}
        )
        self.assertEqualArray(out_a, a)
        self.assertEqualArray(total, a + b, atol=1e-6)

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def test_select_sum(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (s,) = self._run("SELECT SUM(a) FROM t", dtypes, {"a": a})
        self.assertEqualArray(np.array(float(s)), np.array(6.0), atol=1e-5)

    def test_select_avg(self):
        dtypes = {"a": np.float32}
        a = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        (mean,) = self._run("SELECT AVG(a) FROM t", dtypes, {"a": a})
        self.assertEqualArray(np.array(float(mean)), np.array(4.0), atol=1e-5)

    def test_select_min(self):
        dtypes = {"a": np.float32}
        a = np.array([3.0, 1.0, 2.0], dtype=np.float32)
        (mn,) = self._run("SELECT MIN(a) FROM t", dtypes, {"a": a})
        self.assertEqualArray(np.array(float(mn)), np.array(1.0), atol=1e-5)

    def test_select_max(self):
        dtypes = {"a": np.float32}
        a = np.array([3.0, 1.0, 2.0], dtype=np.float32)
        (mx,) = self._run("SELECT MAX(a) FROM t", dtypes, {"a": a})
        self.assertEqualArray(np.array(float(mx)), np.array(3.0), atol=1e-5)


class TestSqlToOnnxFilter(ExtTestCase):
    """Tests for the WHERE / filter clause conversion."""

    def _run(self, query, dtypes, feeds):
        onx = sql_to_onnx(query, dtypes)
        ref = ExtendedReferenceEvaluator(onx)
        return ref.run(None, feeds)

    def test_filter_greater(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        (out,) = self._run("SELECT a FROM t WHERE a > 0", dtypes, {"a": a})
        expected = np.array([1.0, 3.0], dtype=np.float32)
        self.assertEqualArray(out, expected)

    def test_filter_less(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        (out,) = self._run("SELECT a FROM t WHERE a < 0", dtypes, {"a": a})
        expected = np.array([-2.0], dtype=np.float32)
        self.assertEqualArray(out, expected)

    def test_filter_greater_equal(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (out,) = self._run("SELECT a FROM t WHERE a >= 2", dtypes, {"a": a})
        expected = np.array([2.0, 3.0], dtype=np.float32)
        self.assertEqualArray(out, expected)

    def test_filter_less_equal(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (out,) = self._run("SELECT a FROM t WHERE a <= 2", dtypes, {"a": a})
        expected = np.array([1.0, 2.0], dtype=np.float32)
        self.assertEqualArray(out, expected)

    def test_filter_equal(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (out,) = self._run("SELECT a FROM t WHERE a = 2", dtypes, {"a": a})
        expected = np.array([2.0], dtype=np.float32)
        self.assertEqualArray(out, expected)

    def test_filter_not_equal(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (out,) = self._run("SELECT a FROM t WHERE a <> 2", dtypes, {"a": a})
        expected = np.array([1.0, 3.0], dtype=np.float32)
        self.assertEqualArray(out, expected)

    def test_filter_and(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, -2.0, 3.0, 2.0], dtype=np.float32)
        (out,) = self._run(
            "SELECT a FROM t WHERE a > 0 AND a < 3", dtypes, {"a": a}
        )
        expected = np.array([1.0, 2.0], dtype=np.float32)
        self.assertEqualArray(out, expected)

    def test_filter_or(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        (out,) = self._run(
            "SELECT a FROM t WHERE a < 0 OR a > 2", dtypes, {"a": a}
        )
        expected = np.array([-2.0, 3.0], dtype=np.float32)
        self.assertEqualArray(out, expected)

    def test_filter_multi_column(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        out_a, out_b = self._run(
            "SELECT a, b FROM t WHERE a > 0", dtypes, {"a": a, "b": b}
        )
        self.assertEqualArray(out_a, np.array([1.0, 3.0], dtype=np.float32))
        self.assertEqualArray(out_b, np.array([10.0, 30.0], dtype=np.float32))

    def test_filter_and_select_expression(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = self._run(
            "SELECT a + b AS total FROM t WHERE a > 0", dtypes, {"a": a, "b": b}
        )
        expected = np.array([5.0, 9.0], dtype=np.float32)
        self.assertEqualArray(total, expected, atol=1e-6)


class TestSqlToOnnxGroupBy(ExtTestCase):
    """Tests for the GROUP BY clause conversion."""

    def _run(self, query, dtypes, feeds):
        onx = sql_to_onnx(query, dtypes)
        ref = ExtendedReferenceEvaluator(onx)
        return ref.run(None, feeds)

    def test_group_by_sum(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 1.0], dtype=np.float32)
        (s,) = self._run("SELECT SUM(a) FROM t GROUP BY b", dtypes, {"a": a, "b": b})
        # SUM over all = 6.0 (no per-group reduction yet)
        self.assertEqualArray(np.array(float(s)), np.array(6.0), atol=1e-5)

    def test_group_by_with_filter(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 1.0, 2.0], dtype=np.float32)
        (s,) = self._run(
            "SELECT SUM(a) FROM t WHERE a > 0 GROUP BY b",
            dtypes,
            {"a": a, "b": b},
        )
        # After filter (a>0): a=[1.0, 3.0], SUM = 4.0
        self.assertEqualArray(np.array(float(s)), np.array(4.0), atol=1e-5)


class TestSqlToOnnxReturnedModel(ExtTestCase):
    """Tests verifying properties of the returned ONNX model."""

    def test_returns_model_proto(self):
        from onnx import ModelProto

        dtypes = {"a": np.float32}
        onx = sql_to_onnx("SELECT a FROM t", dtypes)
        self.assertIsInstance(onx, ModelProto)

    def test_inputs_one_per_column(self):
        dtypes = {"a": np.float32, "b": np.float32, "c": np.float32}
        onx = sql_to_onnx("SELECT a, b, c FROM t", dtypes)
        input_names = [inp.name for inp in onx.graph.input]
        self.assertIn("a", input_names)
        self.assertIn("b", input_names)
        self.assertIn("c", input_names)
        self.assertEqual(len(input_names), 3)

    def test_where_adds_filter_nodes(self):
        dtypes = {"a": np.float32}
        onx = sql_to_onnx("SELECT a FROM t WHERE a > 0", dtypes)
        op_types = {n.op_type for n in onx.graph.node}
        self.assertIn("Compress", op_types)

    def test_aggregation_adds_reduce_nodes(self):
        dtypes = {"a": np.float32}
        onx = sql_to_onnx("SELECT SUM(a) FROM t", dtypes)
        op_types = {n.op_type for n in onx.graph.node}
        self.assertIn("ReduceSum", op_types)


if __name__ == "__main__":
    unittest.main()
