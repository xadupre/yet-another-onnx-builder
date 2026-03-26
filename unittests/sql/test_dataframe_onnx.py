import unittest

import numpy as np

from yobx.ext_test_case import ExtTestCase, has_onnxruntime
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sql import dataframe_to_onnx
from yobx.xtracing.dataframe_trace import _to_ast
from yobx.xtracing.parse import Literal


def _ort_run(onx, feeds):
    from onnxruntime import InferenceSession
    from yobx.container import ExportArtifact

    proto = onx.proto if isinstance(onx, ExportArtifact) else onx
    sess = InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(func, dtypes, feeds):
    """Trace *func*, convert to ONNX, run with reference evaluator."""
    artifact = dataframe_to_onnx(func, dtypes)
    ref = ExtendedReferenceEvaluator(artifact)
    ref_outputs = ref.run(None, feeds)
    if has_onnxruntime():
        ort_outputs = _ort_run(artifact, feeds)
        assert len(ref_outputs) == len(ort_outputs)
        for ro, oo in zip(ref_outputs, ort_outputs):
            np.testing.assert_allclose(oo, ro, rtol=1e-5, atol=1e-6)
    return ref_outputs


# ---------------------------------------------------------------------------
# _to_ast helper
# ---------------------------------------------------------------------------


class TestToAst(ExtTestCase):
    def test_int(self):
        node = _to_ast(42)
        self.assertIsInstance(node, Literal)
        self.assertEqual(node.value, 42)

    def test_float(self):
        node = _to_ast(1.5)
        self.assertIsInstance(node, Literal)
        self.assertAlmostEqual(node.value, 1.5)

    def test_bool(self):
        node = _to_ast(True)
        self.assertIsInstance(node, Literal)
        self.assertEqual(node.value, True)

    def test_unsupported_raises(self):
        with self.assertRaises(TypeError):
            _to_ast("bad_type")


class TestDataframeToOnnx(ExtTestCase):
    def test_passthrough_single_column(self):
        def transform(df):
            return df.select([df["a"].alias("out_a")])

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (out,) = _run(transform, {"a": np.float32}, {"a": a})
        self.assertEqualArray(out, a)

    def test_passthrough_all_columns_implicit(self):
        """When only a filter is applied, all source columns pass through."""

        def transform(df):
            return df.filter(df["a"] > 1.0)

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        out_a, out_b = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        self.assertEqualArray(out_a, np.array([2.0, 3.0], dtype=np.float32))
        self.assertEqualArray(out_b, np.array([5.0, 6.0], dtype=np.float32))

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def test_select_add(self):
        def transform(df):
            return df.select([(df["a"] + df["b"]).alias("total")])

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)

    def test_select_sub(self):
        def transform(df):
            return df.select([(df["a"] - df["b"]).alias("diff")])

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (diff,) = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(diff, a - b)

    def test_select_mul(self):
        def transform(df):
            return df.select([(df["a"] * df["b"]).alias("product")])

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (product,) = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(product, a * b)

    def test_select_div(self):
        def transform(df):
            return df.select([(df["a"] / df["b"]).alias("ratio")])

        a = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (ratio,) = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(ratio, a / b)

    def test_select_scalar_add(self):
        def transform(df):
            return df.select([(df["a"] + 1.0).alias("inc")])

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (inc,) = _run(transform, {"a": np.float32}, {"a": a})
        np.testing.assert_allclose(inc, a + 1.0, rtol=1e-5)

    def test_select_multiple_columns(self):
        def transform(df):
            return df.select([(df["a"] + df["b"]).alias("s"), (df["a"] - df["b"]).alias("d")])

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        s, d = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(s, a + b)
        np.testing.assert_allclose(d, a - b)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def test_filter_gt(self):
        def transform(df):
            df = df.filter(df["a"] > 0)
            return df.select([(df["a"] + df["b"]).alias("total")])

        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(total, np.array([5.0, 9.0], dtype=np.float32))

    def test_filter_and(self):
        def transform(df):
            return df.filter((df["a"] > 1.0) & (df["b"] < 6.0))

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        out_a, out_b = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(out_a, np.array([2.0], dtype=np.float32))
        np.testing.assert_allclose(out_b, np.array([5.0], dtype=np.float32))

    def test_filter_or(self):
        def transform(df):
            return df.filter((df["a"] < 1.5) | (df["a"] > 2.5))

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        out_a, _out_b = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(out_a, np.array([1.0, 3.0], dtype=np.float32))

    def test_filter_via_getitem(self):
        """df[condition] should be equivalent to df.filter(condition)."""

        def transform(df):
            return df[df["a"] > 0].select([(df["a"] + df["b"]).alias("total")])

        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(total, np.array([5.0, 9.0], dtype=np.float32))

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def test_agg_sum(self):
        def transform(df):
            return df.select([df["v"].sum().alias("total")])

        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        (total,) = _run(transform, {"v": np.float32}, {"v": v})
        np.testing.assert_allclose(total, np.array([10.0], dtype=np.float32))

    def test_agg_mean(self):
        def transform(df):
            return df.select([df["v"].mean().alias("avg")])

        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        (avg,) = _run(transform, {"v": np.float32}, {"v": v})
        np.testing.assert_allclose(avg, np.array([2.5], dtype=np.float32))

    def test_agg_min(self):
        def transform(df):
            return df.select([df["v"].min().alias("mn")])

        v = np.array([3.0, 1.0, 2.0], dtype=np.float32)
        (mn,) = _run(transform, {"v": np.float32}, {"v": v})
        np.testing.assert_allclose(mn, np.array([1.0], dtype=np.float32))

    def test_agg_max(self):
        def transform(df):
            return df.select([df["v"].max().alias("mx")])

        v = np.array([3.0, 1.0, 2.0], dtype=np.float32)
        (mx,) = _run(transform, {"v": np.float32}, {"v": v})
        np.testing.assert_allclose(mx, np.array([3.0], dtype=np.float32))

    # ------------------------------------------------------------------
    # Float64 inputs
    # ------------------------------------------------------------------

    def test_float64_add(self):
        def transform(df):
            return df.select([(df["a"] + df["b"]).alias("total")])

        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        (total,) = _run(transform, {"a": np.float64, "b": np.float64}, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)

    # ------------------------------------------------------------------
    # dict-style select
    # ------------------------------------------------------------------

    def test_select_dict_style(self):
        def transform(df):
            return df.select({"total": df["a"] + df["b"]})

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)

    # ------------------------------------------------------------------
    # Public imports
    # ------------------------------------------------------------------

    def test_imported_from_sql_package(self):
        from yobx.sql import dataframe_to_onnx as dtonnx  # noqa: F401

        self.assertTrue(callable(dtonnx))

    def test_trace_dataframe_imported_from_sql_package(self):
        from yobx.sql import trace_dataframe as td  # noqa: F401

        self.assertTrue(callable(td))

    def test_imported_from_xtracing_package(self):
        from yobx.sql import dataframe_to_onnx as dtonnx  # noqa: F401

        self.assertTrue(callable(dtonnx))

    def test_trace_dataframe_imported_from_xtracing_package(self):
        from yobx.xtracing import trace_dataframe as td  # noqa: F401

        self.assertTrue(callable(td))

    def test_traced_classes_imported_from_xtracing(self):
        from yobx.xtracing import (  # noqa: F401
            TracedCondition,
            TracedDataFrame,
            TracedGroupBy,
            TracedSeries,
        )

        self.assertTrue(issubclass(TracedDataFrame, object))
        self.assertTrue(issubclass(TracedSeries, object))
        self.assertTrue(issubclass(TracedCondition, object))
        self.assertTrue(issubclass(TracedGroupBy, object))

    # ------------------------------------------------------------------
    # pipe — calling other functions that process a dataframe
    # ------------------------------------------------------------------

    def test_pipe_single(self):
        """A function using .pipe() to call another dataframe function."""

        def preprocess(df):
            return df.filter(df["a"] > 0)

        def transform(df):
            return df.pipe(preprocess).select([(df["a"] + df["b"]).alias("total")])

        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(total, np.array([5.0, 9.0], dtype=np.float32))

    def test_pipe_chained(self):
        """Multiple .pipe() calls chaining two sub-functions."""

        def preprocess(df):
            return df.filter(df["a"] > 0)

        def project(df):
            return df.select([(df["a"] + df["b"]).alias("total")])

        def pipeline(df):
            return df.pipe(preprocess).pipe(project)

        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run(pipeline, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(total, np.array([5.0, 9.0], dtype=np.float32))

    def test_pipe_with_extra_args(self):
        """pipe() forwards extra positional and keyword arguments to the function."""

        def scale(df, factor):
            return df.assign(a=(df["a"] * factor).alias("a"))

        def pipeline(df):
            df2 = df.pipe(scale, factor=2.0)
            return df2.select([(df2["a"] + df2["b"]).alias("total")])

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run(pipeline, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(total, a * 2.0 + b)


def _run_multi(func, dtypes_list, feeds):
    """Trace *func* with a list of dtype dicts, convert to ONNX, run."""
    artifact = dataframe_to_onnx(func, dtypes_list)
    ref = ExtendedReferenceEvaluator(artifact)
    ref_outputs = ref.run(None, feeds)
    if has_onnxruntime():
        ort_outputs = _ort_run(artifact, feeds)
        assert len(ref_outputs) == len(ort_outputs)
        for ro, oo in zip(ref_outputs, ort_outputs):
            np.testing.assert_allclose(oo, ro, rtol=1e-5, atol=1e-6)
    return ref_outputs


class TestMultiDataframe(ExtTestCase):
    """Tests for functions that accept multiple :class:`TracedDataFrame` arguments."""

    # ------------------------------------------------------------------
    # Basic multi-frame column access (no join)
    # ------------------------------------------------------------------

    def test_two_frames_add(self):
        """Columns from two independent frames combined with +."""

        def transform(df1, df2):
            return df1.select([(df1["a"] + df2["b"]).alias("total")])

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run_multi(transform, [{"a": np.float32}, {"b": np.float32}], {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)

    def test_two_frames_filter_and_select(self):
        """Filter on df1 column, select combined expression."""

        def transform(df1, df2):
            filtered = df1.filter(df1["a"] > 0)
            return filtered.select([(filtered["a"] + df2["b"]).alias("total")])

        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run_multi(transform, [{"a": np.float32}, {"b": np.float32}], {"a": a, "b": b})
        np.testing.assert_allclose(total, np.array([5.0, 9.0], dtype=np.float32))

    def test_three_frames(self):
        """Three independent frames, columns summed together."""

        def transform(df1, df2, df3):
            return df1.select([(df1["a"] + df2["b"] + df3["c"]).alias("total")])

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        c = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        (total,) = _run_multi(
            transform,
            [{"a": np.float32}, {"b": np.float32}, {"c": np.float32}],
            {"a": a, "b": b, "c": c},
        )
        np.testing.assert_allclose(total, a + b + c)

    def test_join_two_frames(self):
        """Inner join on different key column names."""

        def transform(df1, df2):
            return df1.join(df2, left_key="cid", right_key="id")

        cid = np.array([1, 2, 3], dtype=np.int64)
        a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        id_ = np.array([2, 3], dtype=np.int64)
        b = np.array([200.0, 300.0], dtype=np.float32)

        dtypes1 = {"cid": np.int64, "a": np.float32}
        dtypes2 = {"id": np.int64, "b": np.float32}
        artifact = dataframe_to_onnx(transform, [dtypes1, dtypes2])
        ref = ExtendedReferenceEvaluator(artifact)
        cid_out, a_out, _id_out, b_out = ref.run(None, {"cid": cid, "a": a, "id": id_, "b": b})
        # Rows where cid matches id: (cid=2,a=20), (cid=3,a=30)
        np.testing.assert_array_equal(cid_out, np.array([2, 3], dtype=np.int64))
        np.testing.assert_allclose(a_out, np.array([20.0, 30.0], dtype=np.float32))
        np.testing.assert_allclose(b_out, np.array([200.0, 300.0], dtype=np.float32))

    def test_join_with_select(self):
        """Join two frames then select expressions involving both sides."""

        def transform(df1, df2):
            joined = df1.join(df2, left_key="cid", right_key="id")
            return joined.select([(joined["a"] + joined["b"]).alias("sum_ab")])

        cid = np.array([1, 2, 3], dtype=np.int64)
        a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        id_ = np.array([2, 3], dtype=np.int64)
        b = np.array([200.0, 300.0], dtype=np.float32)

        dtypes1 = {"cid": np.int64, "a": np.float32}
        dtypes2 = {"id": np.int64, "b": np.float32}
        artifact = dataframe_to_onnx(transform, [dtypes1, dtypes2])
        ref = ExtendedReferenceEvaluator(artifact)
        (sum_ab,) = ref.run(None, {"cid": cid, "a": a, "id": id_, "b": b})
        np.testing.assert_allclose(sum_ab, np.array([220.0, 330.0], dtype=np.float32))

    # ------------------------------------------------------------------
    # to_onnx dispatcher (callable path)
    # ------------------------------------------------------------------

    def test_to_onnx_multi_frame(self):
        """to_onnx() correctly dispatches multi-frame callables."""
        from yobx.sql import to_onnx

        def transform(df1, df2):
            return df1.select([(df1["a"] + df2["b"]).alias("total")])

        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        artifact = to_onnx(transform, [{"a": np.float32}, {"b": np.float32}])
        ref = ExtendedReferenceEvaluator(artifact)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)


# ---------------------------------------------------------------------------
# parsed_query_to_onnx
# ---------------------------------------------------------------------------


class TestParsedQueryToOnnx(ExtTestCase):
    """Tests for :func:`~yobx.sql.sql_convert.parsed_query_to_onnx`."""

    def test_basic_select_add(self):
        from yobx.sql import parse_sql
        from yobx.sql.sql_convert import parsed_query_to_onnx

        pq = parse_sql("SELECT a + b AS total FROM t")
        dtypes = {"a": np.float32, "b": np.float32}
        artifact = parsed_query_to_onnx(pq, dtypes)
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)

    def test_with_filter(self):
        from yobx.sql import parse_sql
        from yobx.sql.sql_convert import parsed_query_to_onnx

        pq = parse_sql("SELECT a + b AS total FROM t WHERE a > 0")
        dtypes = {"a": np.float32, "b": np.float32}
        artifact = parsed_query_to_onnx(pq, dtypes)
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, np.array([5.0, 9.0], dtype=np.float32))

    def test_no_input_dtypes_when_traced(self):
        """parsed_query_to_onnx works without input_dtypes for tracer-produced queries."""
        from yobx.xtracing.dataframe_trace import trace_dataframe
        from yobx.sql.sql_convert import parsed_query_to_onnx

        def transform(df):
            return df.select([(df["a"] + df["b"]).alias("total")])

        pq = trace_dataframe(transform, {"a": np.float32, "b": np.float32})
        # No input_dtypes argument — dtype info comes from ColumnRef.dtype
        artifact = parsed_query_to_onnx(pq)
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)

    def test_imported_from_sql_package(self):
        from yobx.sql import parsed_query_to_onnx  # noqa: F401

        self.assertTrue(callable(parsed_query_to_onnx))

    def test_parsed_query_to_onnx_graph_imported(self):
        from yobx.sql import parsed_query_to_onnx_graph  # noqa: F401

        self.assertTrue(callable(parsed_query_to_onnx_graph))


# ---------------------------------------------------------------------------
# DataFrame element-wise arithmetic
# ---------------------------------------------------------------------------


class TestDataframeArithmetic(ExtTestCase):
    def test_df_add_scalar_onnx(self):
        """df + scalar: all columns increased by scalar."""

        def transform(df):
            return df + 1.0

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        out_a, out_b = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(out_a, a + 1.0, rtol=1e-5)
        np.testing.assert_allclose(out_b, b + 1.0, rtol=1e-5)

    def test_df_sub_scalar_onnx(self):
        """df - scalar: all columns decreased by scalar."""

        def transform(df):
            return df - 1.0

        a = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        (out_a,) = _run(transform, {"a": np.float32}, {"a": a})
        np.testing.assert_allclose(out_a, a - 1.0, rtol=1e-5)

    def test_df_mul_scalar_onnx(self):
        """df * scalar: all columns multiplied by scalar."""

        def transform(df):
            return df * 2.0

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (out_a,) = _run(transform, {"a": np.float32}, {"a": a})
        np.testing.assert_allclose(out_a, a * 2.0, rtol=1e-5)

    def test_df_div_scalar_onnx(self):
        """df / scalar: all columns divided by scalar."""

        def transform(df):
            return df / 4.0

        a = np.array([4.0, 8.0, 12.0], dtype=np.float32)
        (out_a,) = _run(transform, {"a": np.float32}, {"a": a})
        np.testing.assert_allclose(out_a, a / 4.0, rtol=1e-5)

    def test_df_radd_scalar_onnx(self):
        """scalar + df: all columns added to scalar."""

        def transform(df):
            return 1.0 + df

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (out_a,) = _run(transform, {"a": np.float32}, {"a": a})
        np.testing.assert_allclose(out_a, 1.0 + a, rtol=1e-5)

    def test_df_rsub_scalar_onnx(self):
        """scalar - df: scalar minus each column element."""

        def transform(df):
            return 10.0 - df

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (out_a,) = _run(transform, {"a": np.float32}, {"a": a})
        np.testing.assert_allclose(out_a, 10.0 - a, rtol=1e-5)

    def test_df_rmul_scalar_onnx(self):
        """scalar * df."""

        def transform(df):
            return 3.0 * df

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (out_a,) = _run(transform, {"a": np.float32}, {"a": a})
        np.testing.assert_allclose(out_a, 3.0 * a, rtol=1e-5)

    def test_df_rtruediv_scalar_onnx(self):
        """scalar / df."""

        def transform(df):
            return 1.0 / df

        a = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        (out_a,) = _run(transform, {"a": np.float32}, {"a": a})
        np.testing.assert_allclose(out_a, 1.0 / a, rtol=1e-5)

    def test_df_add_then_filter(self):
        """df + 1 followed by filter uses computed column values."""

        def transform(df):
            df2 = df + 1.0
            return df2.filter(df2["a"] > 2.0)

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (out_a,) = _run(transform, {"a": np.float32}, {"a": a})
        # After +1: [2, 3, 4]. Filter > 2: [3, 4].
        np.testing.assert_allclose(out_a, np.array([3.0, 4.0], dtype=np.float32), rtol=1e-5)

    def test_df_add_then_select(self):
        """df + 1 followed by select uses the post-arithmetic column expressions."""

        def transform(df):
            df2 = df + 1.0
            return df2.select([(df2["a"] + df2["b"]).alias("total")])

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        # (a+1) + (b+1)
        np.testing.assert_allclose(total, (a + 1.0) + (b + 1.0), rtol=1e-5)

    def test_df_add_dataframe_onnx(self):
        """df1 + df2 adds matching columns element-wise (same column names)."""

        def transform(df1, df2):
            return df1 + df2

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # Both frames reference column "a", so there is a single ONNX input "a".
        # The result is a + a.
        (out_a,) = _run_multi(transform, [{"a": np.float32}, {"a": np.float32}], {"a": a})
        np.testing.assert_allclose(out_a, a + a, rtol=1e-5)

    def test_df_add_scalar_two_columns_onnx(self):
        """df + scalar with two columns: both columns are incremented."""

        def transform(df):
            return df + 1.0

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        out_a, out_b = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(out_a, a + 1.0, rtol=1e-5)
        np.testing.assert_allclose(out_b, b + 1.0, rtol=1e-5)

    def test_df_mul_scalar_two_columns_onnx(self):
        """df * scalar with two columns: both columns are scaled."""

        def transform(df):
            return df * 2.0

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        out_a, out_b = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(out_a, a * 2.0, rtol=1e-5)
        np.testing.assert_allclose(out_b, b * 2.0, rtol=1e-5)

    def test_df_sub_scalar_two_columns_onnx(self):
        """df - scalar with two columns: scalar subtracted from both columns."""

        def transform(df):
            return df - 1.0

        a = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        out_a, out_b = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(out_a, a - 1.0, rtol=1e-5)
        np.testing.assert_allclose(out_b, b - 1.0, rtol=1e-5)

    def test_df_chained_arith_two_columns_onnx(self):
        """(df + 1) * 2 applied to a two-column frame."""

        def transform(df):
            return (df + 1.0) * 2.0

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        out_a, out_b = _run(transform, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(out_a, (a + 1.0) * 2.0, rtol=1e-5)
        np.testing.assert_allclose(out_b, (b + 1.0) * 2.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# DataFrame as input_dtypes
# ---------------------------------------------------------------------------


class TestInputDtypesAsDataFrame(ExtTestCase):
    """Tests that a pandas DataFrame can be passed as *input_dtypes* to all entry points."""

    @classmethod
    def _make_df(cls):
        """Return a small two-column pandas DataFrame with float32 columns."""
        try:
            import pandas as pd
        except ImportError:
            return None
        return pd.DataFrame(
            {
                "a": np.array([1.0, -2.0, 3.0], dtype=np.float32),
                "b": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            }
        )

    def test_sql_to_onnx_with_dataframe(self):
        """sql_to_onnx accepts a DataFrame instead of a dtype dict."""
        from yobx.sql import sql_to_onnx

        df = self._make_df()
        if df is None:
            self.skipTest("pandas not available")

        artifact = sql_to_onnx("SELECT a + b AS total FROM t", df)
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)

    def test_sql_to_onnx_with_dataframe_filter(self):
        """sql_to_onnx with DataFrame input_dtypes and a WHERE clause."""
        from yobx.sql import sql_to_onnx

        df = self._make_df()
        if df is None:
            self.skipTest("pandas not available")

        artifact = sql_to_onnx("SELECT a + b AS total FROM t WHERE a > 0", df)
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, np.array([5.0, 9.0], dtype=np.float32))

    def test_dataframe_to_onnx_with_dataframe(self):
        """dataframe_to_onnx accepts a DataFrame instead of a dtype dict."""
        df = self._make_df()
        if df is None:
            self.skipTest("pandas not available")

        def transform(traced_df):
            return traced_df.select([(traced_df["a"] + traced_df["b"]).alias("total")])

        artifact = dataframe_to_onnx(transform, df)
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)

    def test_to_onnx_sql_with_dataframe(self):
        """to_onnx(SQL string, DataFrame) works as expected."""
        from yobx.sql import to_onnx

        df = self._make_df()
        if df is None:
            self.skipTest("pandas not available")

        artifact = to_onnx("SELECT a + b AS total FROM t WHERE a > 0", df)
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, np.array([5.0, 9.0], dtype=np.float32))

    def test_to_onnx_callable_with_dataframe(self):
        """to_onnx(callable, DataFrame) works as expected."""
        from yobx.sql import to_onnx

        df = self._make_df()
        if df is None:
            self.skipTest("pandas not available")

        def transform(traced_df):
            traced_df = traced_df.filter(traced_df["a"] > 0)
            return traced_df.select([(traced_df["a"] + traced_df["b"]).alias("total")])

        artifact = to_onnx(transform, df)
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, np.array([5.0, 9.0], dtype=np.float32))

    def test_to_onnx_callable_with_list_of_dataframes(self):
        """to_onnx(callable, [DataFrame, DataFrame]) works for multi-frame transforms."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not available")

        from yobx.sql import to_onnx

        df1 = pd.DataFrame({"a": np.array([1.0, 2.0], dtype=np.float32)})
        df2 = pd.DataFrame({"b": np.array([3.0, 4.0], dtype=np.float32)})

        def transform(traced_df1, traced_df2):
            return traced_df1.select([(traced_df1["a"] + traced_df2["b"]).alias("total")])

        artifact = to_onnx(transform, [df1, df2])
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)


if __name__ == "__main__":
    unittest.main()
