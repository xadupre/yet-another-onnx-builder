"""
Unit tests for :mod:`yobx.sql.sql_convert` — SQL-to-ONNX converter.
"""

import unittest
import numpy as np

from yobx.ext_test_case import ExtTestCase, has_onnxruntime
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sql import sql_to_onnx, sql_to_onnx_graph


def _ort_run(onx, feeds):
    """Run *onx* with onnxruntime and return outputs. Skip if ORT is not installed."""
    from onnxruntime import InferenceSession
    from yobx.container import ExportArtifact

    proto = onx.proto if isinstance(onx, ExportArtifact) else onx
    sess = InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
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
        # True GROUP BY: group b=1 → SUM([1,3])=4, group b=2 → SUM([2])=2
        expected = np.array([4.0, 2.0], dtype=np.float32)
        self.assertEqualArray(np.sort(s), np.sort(expected), atol=1e-5)

    def test_group_by_with_filter(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 1.0, 2.0], dtype=np.float32)
        (s,) = self._run("SELECT SUM(a) FROM t WHERE a > 0 GROUP BY b", dtypes, {"a": a, "b": b})
        # After filter (a>0): a=[1.0, 3.0], b=[1.0, 2.0]
        # group b=1 → SUM([1])=1, group b=2 → SUM([3])=3
        expected = np.array([1.0, 3.0], dtype=np.float32)
        self.assertEqualArray(np.sort(s), np.sort(expected), atol=1e-5)

    def test_group_by_avg(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 1.0, 2.0], dtype=np.float32)
        (avg,) = self._run("SELECT AVG(a) FROM t GROUP BY b", dtypes, {"a": a, "b": b})
        # group b=1 → AVG([1,3])=2, group b=2 → AVG([2,4])=3
        expected = np.array([2.0, 3.0], dtype=np.float32)
        self.assertEqualArray(np.sort(avg), np.sort(expected), atol=1e-5)

    def test_group_by_min(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, 5.0, 2.0, 4.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 1.0, 2.0, 1.0], dtype=np.float32)
        (mn,) = self._run("SELECT MIN(a) FROM t GROUP BY b", dtypes, {"a": a, "b": b})
        # group b=1 → MIN([1,2,3])=1, group b=2 → MIN([5,4])=4
        expected = np.array([1.0, 4.0], dtype=np.float32)
        self.assertEqualArray(np.sort(mn), np.sort(expected), atol=1e-5)

    def test_group_by_max(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, 5.0, 2.0, 4.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 1.0, 2.0, 1.0], dtype=np.float32)
        (mx,) = self._run("SELECT MAX(a) FROM t GROUP BY b", dtypes, {"a": a, "b": b})
        # group b=1 → MAX([1,2,3])=3, group b=2 → MAX([5,4])=5
        expected = np.array([3.0, 5.0], dtype=np.float32)
        self.assertEqualArray(np.sort(mx), np.sort(expected), atol=1e-5)

    def test_group_by_key_in_select(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 1.0], dtype=np.float32)
        b_out, s = self._run("SELECT b, SUM(a) FROM t GROUP BY b", dtypes, {"a": a, "b": b})
        # Unique b values (sorted): [1.0, 2.0]
        # group b=1 → SUM([1,3])=4, group b=2 → SUM([2])=2
        order = np.argsort(b_out)
        self.assertEqualArray(b_out[order], np.array([1.0, 2.0], dtype=np.float32), atol=1e-5)
        self.assertEqualArray(s[order], np.array([4.0, 2.0], dtype=np.float32), atol=1e-5)

    def test_group_by_two_columns(self):
        dtypes = {"a": np.float32, "b": np.float32, "c": np.float32}
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 1.0, 2.0, 1.0], dtype=np.float32)
        c = np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
        # Groups: (b=1,c=1)→[1,5]→sum=6, (b=1,c=2)→[3]→sum=3, (b=2,c=1)→[2,4]→sum=6
        (s,) = self._run("SELECT SUM(a) FROM t GROUP BY b, c", dtypes, {"a": a, "b": b, "c": c})
        expected = np.array([6.0, 3.0, 6.0], dtype=np.float32)
        self.assertEqualArray(np.sort(s), np.sort(expected), atol=1e-5)

    def test_group_by_three_columns(self):
        dtypes = {"a": np.float32, "b": np.float32, "c": np.float32, "d": np.float32}
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
        c = np.array([1.0, 1.0, 1.0, 2.0], dtype=np.float32)
        d = np.array([1.0, 2.0, 1.0, 1.0], dtype=np.float32)
        # Groups: (1,1,1)→[1]→1, (1,1,2)→[2]→2, (2,1,1)→[3]→3, (2,2,1)→[4]→4
        (s,) = self._run(
            "SELECT SUM(a) FROM t GROUP BY b, c, d", dtypes, {"a": a, "b": b, "c": c, "d": d}
        )
        expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        self.assertEqualArray(np.sort(s), np.sort(expected), atol=1e-5)

    def test_group_by_two_columns_key_in_select(self):
        dtypes = {"a": np.float32, "b": np.float32, "c": np.float32}
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 1.0, 2.0, 1.0], dtype=np.float32)
        c = np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32)
        b_out, c_out, s = self._run(
            "SELECT b, c, SUM(a) FROM t GROUP BY b, c", dtypes, {"a": a, "b": b, "c": c}
        )
        # Sort output by (b, c) for stable comparison
        order = np.lexsort((c_out, b_out))
        self.assertEqualArray(
            b_out[order], np.array([1.0, 1.0, 2.0], dtype=np.float32), atol=1e-5
        )
        self.assertEqualArray(
            c_out[order], np.array([1.0, 2.0, 1.0], dtype=np.float32), atol=1e-5
        )
        self.assertEqualArray(s[order], np.array([6.0, 3.0, 6.0], dtype=np.float32), atol=1e-5)


class TestSqlToOnnxReturnedModel(ExtTestCase):
    """Tests verifying properties of the returned ONNX model."""

    def test_returns_model_proto(self):
        from yobx.container import ExportArtifact

        dtypes = {"a": np.float32}
        onx = sql_to_onnx("SELECT a FROM t", dtypes)
        self.assertIsInstance(onx, ExportArtifact)
        # The underlying proto is accessible via the attribute:
        from onnx import ModelProto

        self.assertIsInstance(onx.proto, ModelProto)

    def test_inputs_one_per_column(self):
        dtypes = {"a": np.float32, "b": np.float32, "c": np.float32}
        onx = sql_to_onnx("SELECT a, b, c FROM t", dtypes)
        input_names = [inp.name for inp in onx.proto.graph.input]
        self.assertIn("a", input_names)
        self.assertIn("b", input_names)
        self.assertIn("c", input_names)
        self.assertEqual(len(input_names), 3)

    def test_where_adds_filter_nodes(self):
        dtypes = {"a": np.float32}
        onx = sql_to_onnx("SELECT a FROM t WHERE a > 0", dtypes)
        op_types = {n.op_type for n in onx.proto.graph.node}
        self.assertIn("Compress", op_types)

    def test_aggregation_adds_reduce_nodes(self):
        dtypes = {"a": np.float32}
        onx = sql_to_onnx("SELECT SUM(a) FROM t", dtypes)
        op_types = {n.op_type for n in onx.proto.graph.node}
        self.assertIn("ReduceSum", op_types)

    def test_builder_cls_used(self):
        """builder_cls should be instantiated instead of the default GraphBuilder."""
        from yobx.container import ExportArtifact
        from yobx.xbuilder import GraphBuilder

        instantiated = []

        class TrackingBuilder(GraphBuilder):
            def __init__(self, *args, **kwargs):
                instantiated.append(True)
                super().__init__(*args, **kwargs)

        dtypes = {"a": np.float32}
        onx = sql_to_onnx("SELECT a FROM t", dtypes, builder_cls=TrackingBuilder)
        self.assertIsInstance(onx, ExportArtifact)
        self.assertEqual(len(instantiated), 1, "TrackingBuilder was not instantiated")


class TestSqlToOnnxGraph(ExtTestCase):
    """Tests for :func:`~yobx.sql.sql_to_onnx_graph`."""

    def _build_and_run(self, query, dtypes, feeds, *, right_dtypes=None):
        """Call sql_to_onnx_graph, finalise the model, run with reference evaluator."""
        from yobx.xbuilder import GraphBuilder

        g = GraphBuilder(18, ir_version=10)
        out_names = sql_to_onnx_graph(g, None, [], query, dtypes, right_input_dtypes=right_dtypes)
        onx = g.to_onnx(return_optimize_report=True)
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
        onx = g.to_onnx(return_optimize_report=True)
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

    def test_output_names_from_alias(self):
        """Output tensor names should reflect SELECT aliases when they don't conflict."""
        from yobx.xbuilder import GraphBuilder

        g = GraphBuilder(18, ir_version=10)
        dtypes = {"a": np.float32, "b": np.float32}
        out_names = sql_to_onnx_graph(g, None, [], "SELECT a + b AS total FROM t", dtypes)
        # "total" is the alias and doesn't collide with any input
        self.assertEqual(out_names, ["total"])

    def test_output_names_fallback_on_collision(self):
        """When the alias conflicts with an input name, fall back to indexed name."""
        from yobx.xbuilder import GraphBuilder

        g = GraphBuilder(18, ir_version=10)
        dtypes = {"a": np.float32}
        out_names = sql_to_onnx_graph(g, None, [], "SELECT a FROM t", dtypes)
        # "a" is already registered as an input, so must fall back to "output_0"
        self.assertEqual(out_names, ["output_0"])


class TestSqlToOnnxSubquery(ExtTestCase):
    """Tests for subquery (``SELECT … FROM (SELECT …)``) conversion."""

    def _run(self, query, dtypes, feeds):
        """Convert *query*, run with reference evaluator and (when available) ORT."""
        onx = sql_to_onnx(query, dtypes)
        ref = ExtendedReferenceEvaluator(onx)
        ref_outputs = ref.run(None, feeds)
        if has_onnxruntime():
            ort_outputs = _ort_run(onx, feeds)
            self.assertEqual(len(ref_outputs), len(ort_outputs))
            for ref_out, ort_out in zip(ref_outputs, ort_outputs):
                np.testing.assert_allclose(ort_out, ref_out, rtol=1e-5, atol=1e-6)
        return ref_outputs

    def test_subquery_passthrough_single_column(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (out,) = self._run("SELECT a FROM (SELECT a FROM t)", dtypes, {"a": a})
        self.assertEqualArray(out, a)

    def test_subquery_passthrough_multiple_columns(self):
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        out_a, out_b = self._run(
            "SELECT a, b FROM (SELECT a, b FROM t)", dtypes, {"a": a, "b": b}
        )
        self.assertEqualArray(out_a, a)
        self.assertEqualArray(out_b, b)

    def test_subquery_inner_expression(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (out,) = self._run("SELECT a FROM (SELECT a + 1 AS a FROM t)", dtypes, {"a": a})
        self.assertEqualArray(out, a + 1, atol=1e-6)

    def test_subquery_inner_where(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        (out,) = self._run("SELECT a FROM (SELECT a FROM t WHERE a > 0)", dtypes, {"a": a})
        self.assertEqualArray(out, np.array([1.0, 3.0], dtype=np.float32))

    def test_subquery_outer_where(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        (out,) = self._run(
            "SELECT a FROM (SELECT a + 1 AS a FROM t) WHERE a > 2", dtypes, {"a": a}
        )
        # Inner: a+1 = [2, 3, 4, 5]; outer WHERE a > 2 => [3, 4, 5]
        self.assertEqualArray(out, np.array([3.0, 4.0, 5.0], dtype=np.float32))

    def test_subquery_with_alias(self):
        dtypes = {"a": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (out,) = self._run("SELECT a FROM (SELECT a FROM t) AS sub", dtypes, {"a": a})
        self.assertEqualArray(out, a)

    def test_subquery_inner_and_outer_where(self):
        dtypes = {"a": np.float32}
        a = np.array([-1.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        (out,) = self._run(
            "SELECT a FROM (SELECT a FROM t WHERE a > 0) WHERE a < 4", dtypes, {"a": a}
        )
        # Inner WHERE: [1, 2, 3, 4]; outer WHERE a < 4: [1, 2, 3]
        self.assertEqualArray(out, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_subquery_column_rename(self):
        """Outer query can reference the inner alias."""
        dtypes = {"a": np.float32, "b": np.float32}
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (out,) = self._run(
            "SELECT total FROM (SELECT a + b AS total FROM t)", dtypes, {"a": a, "b": b}
        )
        self.assertEqualArray(out, a + b, atol=1e-6)


class TestSqlToOnnxJoin(ExtTestCase):
    """Tests for ``JOIN … ON …`` conversion — single and multi-column keys."""

    def _run(self, query, dtypes, feeds, *, right_dtypes=None):
        onx = sql_to_onnx(query, dtypes, right_input_dtypes=right_dtypes)
        ref = ExtendedReferenceEvaluator(onx)
        ref_outputs = ref.run(None, feeds)
        if has_onnxruntime():
            ort_outputs = _ort_run(onx, feeds)
            self.assertEqual(len(ref_outputs), len(ort_outputs))
            for ref_out, ort_out in zip(ref_outputs, ort_outputs):
                np.testing.assert_allclose(ort_out, ref_out, rtol=1e-5, atol=1e-6)
        return ref_outputs

    def test_single_key_join_different_names(self):
        """Basic inner join on a single key with distinct column names on each side."""
        sql = "SELECT a.x, b.y FROM a JOIN b ON a.id = b.rid"
        left_dtypes = {"id": np.int64, "x": np.float32}
        right_dtypes = {"rid": np.int64, "y": np.float32}
        # Left: id=[1,2,3], x=[10,20,30]
        # Right: rid=[2,3], y=[200,300]
        # Match: id=2 (x=20,y=200) and id=3 (x=30,y=300)
        feeds = {
            "id": np.array([1, 2, 3], dtype=np.int64),
            "x": np.array([10.0, 20.0, 30.0], dtype=np.float32),
            "rid": np.array([2, 3], dtype=np.int64),
            "y": np.array([200.0, 300.0], dtype=np.float32),
        }
        x_out, y_out = self._run(sql, left_dtypes, feeds, right_dtypes=right_dtypes)
        np.testing.assert_allclose(x_out, np.array([20.0, 30.0], dtype=np.float32))
        np.testing.assert_allclose(y_out, np.array([200.0, 300.0], dtype=np.float32))

    def test_single_key_join_same_name(self):
        """Inner join where the key column has the same name on both sides.

        The right-side key is renamed to ``id_right`` in the ONNX model to
        avoid clashing with the left-side ``id`` input.
        """
        sql = "SELECT a.x, b.y FROM a JOIN b ON a.id = b.id"
        left_dtypes = {"id": np.int64, "x": np.float32}
        right_dtypes = {"id": np.int64, "y": np.float32}
        onx = sql_to_onnx(sql, left_dtypes, right_input_dtypes=right_dtypes)
        self.assertIn("id_right", onx.input_names)
        ref = ExtendedReferenceEvaluator(onx)
        # Left: id=[1,2,3], Right: id=[2,3]
        feeds = {
            "id": np.array([1, 2, 3], dtype=np.int64),
            "x": np.array([10.0, 20.0, 30.0], dtype=np.float32),
            "id_right": np.array([2, 3], dtype=np.int64),
            "y": np.array([200.0, 300.0], dtype=np.float32),
        }
        x_out, y_out = ref.run(None, feeds)
        np.testing.assert_allclose(x_out, np.array([20.0, 30.0], dtype=np.float32))
        np.testing.assert_allclose(y_out, np.array([200.0, 300.0], dtype=np.float32))

    def test_multi_column_join_two_keys(self):
        """Inner join on two AND-chained key columns with different names."""
        sql = "SELECT a.x, b.y FROM a JOIN b ON a.company_id = b.cid AND a.dept_id = b.did"
        left_dtypes = {"company_id": np.int64, "dept_id": np.int64, "x": np.float32}
        right_dtypes = {"cid": np.int64, "did": np.int64, "y": np.float32}
        # Left: (1,10,1.0), (2,20,2.0), (3,30,3.0)
        # Right: (2,20,200.0), (3,30,300.0), (4,40,400.0)
        # Matches: (2,20) → x=2.0, y=200.0 and (3,30) → x=3.0, y=300.0
        feeds = {
            "company_id": np.array([1, 2, 3], dtype=np.int64),
            "dept_id": np.array([10, 20, 30], dtype=np.int64),
            "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "cid": np.array([2, 3, 4], dtype=np.int64),
            "did": np.array([20, 30, 40], dtype=np.int64),
            "y": np.array([200.0, 300.0, 400.0], dtype=np.float32),
        }
        x_out, y_out = self._run(sql, left_dtypes, feeds, right_dtypes=right_dtypes)
        np.testing.assert_allclose(x_out, np.array([2.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(y_out, np.array([200.0, 300.0], dtype=np.float32))

    def test_multi_column_join_no_match_row_excluded(self):
        """A row that matches only one of two key columns is excluded."""
        sql = "SELECT a.x, b.y FROM a JOIN b ON a.k1 = b.rk1 AND a.k2 = b.rk2"
        left_dtypes = {"k1": np.int64, "k2": np.int64, "x": np.float32}
        right_dtypes = {"rk1": np.int64, "rk2": np.int64, "y": np.float32}
        # Left: (1,10), (2,20), (3,30)
        # Right: (2,99), (3,30)   — row (2,99) matches k1=2 but not k2=20 → excluded
        feeds = {
            "k1": np.array([1, 2, 3], dtype=np.int64),
            "k2": np.array([10, 20, 30], dtype=np.int64),
            "x": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "rk1": np.array([2, 3], dtype=np.int64),
            "rk2": np.array([99, 30], dtype=np.int64),
            "y": np.array([99.0, 300.0], dtype=np.float32),
        }
        x_out, y_out = self._run(sql, left_dtypes, feeds, right_dtypes=right_dtypes)
        # Only (3,30) matches both keys
        np.testing.assert_allclose(x_out, np.array([3.0], dtype=np.float32))
        np.testing.assert_allclose(y_out, np.array([300.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
