"""
Unit tests for :mod:`yobx.sql.convert` — SQL-to-ONNX converter.
"""

import unittest
import numpy as np

from yobx.ext_test_case import ExtTestCase, has_onnxruntime
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sql import sql_to_onnx, sql_to_onnx_graph


def _ort_run(onx, feeds):
    """Run *onx* with onnxruntime and return outputs. Skip if ORT is not installed."""
    from onnxruntime import InferenceSession

    sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


class TestSqlToOnnxSelect(ExtTestCase):
    """Tests for the SELECT clause conversion."""

    def _run(self, query, dtypes, feeds, *, right_dtypes=None):
        """Convert *query*, run with reference evaluator and (when available) ORT."""
        onx = sql_to_onnx(query, dtypes, right_input_dtypes=right_dtypes)
        ref = ExtendedReferenceEvaluator(onx)
        ref_outputs = ref.run(None, feeds)
        if has_onnxruntime():
            ort_outputs = _ort_run(onx, feeds)
            self.assertEqual(len(ref_outputs), len(ort_outputs))
            for ref_out, ort_out in zip(ref_outputs, ort_outputs):
                np.testing.assert_allclose(ort_out, ref_out, rtol=1e-5, atol=1e-6)
        return ref_outputs

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
        out_a, total = self._run("SELECT a, a + b AS total FROM t", dtypes, {"a": a, "b": b})
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
        ref_outputs = ref.run(None, feeds)
        if has_onnxruntime():
            ort_outputs = _ort_run(onx, feeds)
            self.assertEqual(len(ref_outputs), len(ort_outputs))
            for ref_out, ort_out in zip(ref_outputs, ort_outputs):
                np.testing.assert_allclose(ort_out, ref_out, rtol=1e-5, atol=1e-6)
        return ref_outputs

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
        (out,) = self._run("SELECT a FROM t WHERE a > 0 AND a < 3", dtypes, {"a": a})
        expected = np.array([1.0, 2.0], dtype=np.float32)
        self.assertEqualArray(out, expected)

    def test_filter_or(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        (out,) = self._run("SELECT a FROM t WHERE a < 0 OR a > 2", dtypes, {"a": a})
        expected = np.array([-2.0, 3.0], dtype=np.float32)
        self.assertEqualArray(out, expected)

    def test_filter_multi_column(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        out_a, out_b = self._run("SELECT a, b FROM t WHERE a > 0", dtypes, {"a": a, "b": b})
        self.assertEqualArray(out_a, np.array([1.0, 3.0], dtype=np.float32))
        self.assertEqualArray(out_b, np.array([10.0, 30.0], dtype=np.float32))

    def test_filter_and_select_expression(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = self._run("SELECT a + b AS total FROM t WHERE a > 0", dtypes, {"a": a, "b": b})
        expected = np.array([5.0, 9.0], dtype=np.float32)
        self.assertEqualArray(total, expected, atol=1e-6)


class TestSqlToOnnxGroupBy(ExtTestCase):
    """Tests for the GROUP BY clause conversion."""

    def _run(self, query, dtypes, feeds):
        onx = sql_to_onnx(query, dtypes)
        ref = ExtendedReferenceEvaluator(onx)
        ref_outputs = ref.run(None, feeds)
        if has_onnxruntime():
            ort_outputs = _ort_run(onx, feeds)
            self.assertEqual(len(ref_outputs), len(ort_outputs))
            for ref_out, ort_out in zip(ref_outputs, ort_outputs):
                np.testing.assert_allclose(ort_out, ref_out, rtol=1e-5, atol=1e-6)
        return ref_outputs

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
        (s,) = self._run("SELECT SUM(a) FROM t WHERE a > 0 GROUP BY b", dtypes, {"a": a, "b": b})
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


class TestSqlToOnnxGraph(ExtTestCase):
    """Tests for :func:`~yobx.sql.sql_to_onnx_graph`."""

    def _build_and_run(self, query, dtypes, feeds, *, right_dtypes=None):
        """Call sql_to_onnx_graph, finalise the model, run with reference evaluator."""
        from yobx.xbuilder import GraphBuilder

        g = GraphBuilder(18, ir_version=10)
        out_names = sql_to_onnx_graph(
            g,
            None,
            [],
            query,
            dtypes,
            right_input_dtypes=right_dtypes,
        )
        onx, _ = g.to_onnx(return_optimize_report=True)
        ref = ExtendedReferenceEvaluator(onx)
        return out_names, ref.run(None, feeds)

    def test_returns_list_of_output_names(self):
        dtypes = {"a": np.float32, "b": np.float32}
        from yobx.xbuilder import GraphBuilder

        g = GraphBuilder(18, ir_version=10)
        out_names = sql_to_onnx_graph(g, None, [], "SELECT a, b FROM t", dtypes)
        self.assertIsInstance(out_names, list)
        self.assertEqual(len(out_names), 2)

    def test_simple_select(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out_names, (out,) = self._build_and_run("SELECT a FROM t", dtypes, {"a": a})
        self.assertEqual(len(out_names), 1)
        self.assertEqualArray(out, a)

    def test_arithmetic_expression(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        _, (total,) = self._build_and_run(
            "SELECT a + b AS total FROM t", dtypes, {"a": a, "b": b}
        )
        self.assertEqualArray(total, a + b, atol=1e-6)

    def test_where_filter(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        _, (out,) = self._build_and_run("SELECT a FROM t WHERE a > 0", dtypes, {"a": a})
        self.assertEqualArray(out, np.array([1.0, 3.0], dtype=np.float32))

    def test_existing_input_reused(self):
        """sql_to_onnx_graph must not duplicate an input already in the builder."""
        from onnx import TensorProto
        from yobx.xbuilder import GraphBuilder

        g = GraphBuilder(18, ir_version=10)
        g.make_tensor_input("a", TensorProto.FLOAT, ("N",))
        sql_to_onnx_graph(g, None, [], "SELECT a FROM t", {"a": np.float32})
        onx, _ = g.to_onnx(return_optimize_report=True)
        # Exactly one input named "a"
        input_names = [inp.name for inp in onx.graph.input]
        self.assertEqual(input_names.count("a"), 1)

    def test_result_matches_sql_to_onnx(self):
        """sql_to_onnx_graph result must be numerically identical to sql_to_onnx."""
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        feeds = {"a": a, "b": b}
        query = "SELECT a + b AS total FROM t WHERE a > 0"

        onx_high = sql_to_onnx(query, dtypes)
        (high_out,) = ExtendedReferenceEvaluator(onx_high).run(None, feeds)

        _, (low_out,) = self._build_and_run(query, dtypes, feeds)
        self.assertEqualArray(high_out, low_out, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
