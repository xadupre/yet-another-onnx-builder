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

    def test_join_multi_column_different_key_names(self):
        """Multi-column join where key columns have different names on each side."""

        def transform(df1, df2):
            return df1.join(df2, left_key=["company_id", "dept_id"], right_key=["cid", "did"])

        # Left: company_id=[1,2,3], dept_id=[10,20,30], a=[1,2,3]
        # Right: cid=[2,3,4], did=[20,30,40], b=[200,300,400]
        # Matching pairs: (2,20) and (3,30).
        company_id = np.array([1, 2, 3], dtype=np.int64)
        dept_id = np.array([10, 20, 30], dtype=np.int64)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cid = np.array([2, 3, 4], dtype=np.int64)
        did = np.array([20, 30, 40], dtype=np.int64)
        b = np.array([200.0, 300.0, 400.0], dtype=np.float32)

        dtypes1 = {"company_id": np.int64, "dept_id": np.int64, "a": np.float32}
        dtypes2 = {"cid": np.int64, "did": np.int64, "b": np.float32}
        artifact = dataframe_to_onnx(transform, [dtypes1, dtypes2])
        ref = ExtendedReferenceEvaluator(artifact)
        feeds = {
            "company_id": company_id,
            "dept_id": dept_id,
            "a": a,
            "cid": cid,
            "did": did,
            "b": b,
        }
        results = ref.run(None, feeds)
        # Output columns: company_id, dept_id, a, cid, did, b
        company_out, dept_out, a_out, _cid_out, _did_out, b_out = results
        np.testing.assert_array_equal(company_out, np.array([2, 3], dtype=np.int64))
        np.testing.assert_array_equal(dept_out, np.array([20, 30], dtype=np.int64))
        np.testing.assert_allclose(a_out, np.array([2.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(b_out, np.array([200.0, 300.0], dtype=np.float32))

    def test_join_multi_column_same_key_names(self):
        """Multi-column join where key columns share names across both frames."""

        def transform(df1, df2):
            return df1.join(df2, left_key=["k1", "k2"], right_key=["k1", "k2"])

        # Left: k1=[1,2,3], k2=[10,20,30], a=[1,2,3]
        # Right: k1=[2,3,4], k2=[20,30,40], b=[200,300,400]
        # Matching pairs: (k1=2,k2=20) and (k1=3,k2=30).
        k1_l = np.array([1, 2, 3], dtype=np.int64)
        k2_l = np.array([10, 20, 30], dtype=np.int64)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        k1_r = np.array([2, 3, 4], dtype=np.int64)
        k2_r = np.array([20, 30, 40], dtype=np.int64)
        b = np.array([200.0, 300.0, 400.0], dtype=np.float32)

        dtypes1 = {"k1": np.int64, "k2": np.int64, "a": np.float32}
        dtypes2 = {"k1": np.int64, "k2": np.int64, "b": np.float32}
        artifact = dataframe_to_onnx(transform, [dtypes1, dtypes2])
        # The right-side key columns are renamed to "k1_right" / "k2_right" in
        # the ONNX model to avoid clashing with the left-side inputs.
        self.assertEqual(
            sorted(artifact.input_names), ["a", "b", "k1", "k1_right", "k2", "k2_right"]
        )
        ref = ExtendedReferenceEvaluator(artifact)
        feeds = {"k1": k1_l, "k2": k2_l, "a": a, "k1_right": k1_r, "k2_right": k2_r, "b": b}
        k1_out, k2_out, a_out, b_out = ref.run(None, feeds)
        np.testing.assert_array_equal(k1_out, np.array([2, 3], dtype=np.int64))
        np.testing.assert_array_equal(k2_out, np.array([20, 30], dtype=np.int64))
        np.testing.assert_allclose(a_out, np.array([2.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(b_out, np.array([200.0, 300.0], dtype=np.float32))

    def test_join_multi_column_with_select(self):
        """Multi-column join followed by a SELECT expression."""

        def transform(df1, df2):
            joined = df1.join(df2, left_key=["company_id", "dept_id"], right_key=["cid", "did"])
            return joined.select([(joined["a"] + joined["b"]).alias("total")])

        company_id = np.array([1, 2, 3], dtype=np.int64)
        dept_id = np.array([10, 20, 30], dtype=np.int64)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cid = np.array([2, 3, 4], dtype=np.int64)
        did = np.array([20, 30, 40], dtype=np.int64)
        b = np.array([200.0, 300.0, 400.0], dtype=np.float32)

        dtypes1 = {"company_id": np.int64, "dept_id": np.int64, "a": np.float32}
        dtypes2 = {"cid": np.int64, "did": np.int64, "b": np.float32}
        artifact = dataframe_to_onnx(transform, [dtypes1, dtypes2])
        ref = ExtendedReferenceEvaluator(artifact)
        feeds = {
            "company_id": company_id,
            "dept_id": dept_id,
            "a": a,
            "cid": cid,
            "did": did,
            "b": b,
        }
        (total,) = ref.run(None, feeds)
        np.testing.assert_allclose(total, np.array([202.0, 303.0], dtype=np.float32))

    def test_join_multi_column_mismatch_raises(self):
        """Passing left_key and right_key lists of different lengths raises ValueError."""

        def transform(df1, df2):
            return df1.join(df2, left_key=["k1", "k2"], right_key=["k1"])

        with self.assertRaises(ValueError):
            dataframe_to_onnx(
                transform,
                [
                    {"k1": np.int64, "k2": np.int64, "a": np.float32},
                    {"k1": np.int64, "b": np.float32},
                ],
            )

    def test_join_three_consecutive(self):
        """Three consecutive inner joins across four input frames."""

        def transform(orders, customers, products, warehouses):
            j1 = orders.join(customers, left_key="customer_id", right_key="cid")
            j2 = j1.join(products, left_key="product_id", right_key="pid")
            j3 = j2.join(warehouses, left_key="warehouse_id", right_key="wid")
            return j3.select(
                [j3["order_id"], j3["discount"], j3["unit_price"], j3["shipping_cost"]]
            )

        dtypes_orders = {
            "order_id": np.int64,
            "customer_id": np.int64,
            "product_id": np.int64,
            "warehouse_id": np.int64,
        }
        dtypes_customers = {"cid": np.int64, "discount": np.float32}
        dtypes_products = {"pid": np.int64, "unit_price": np.float32}
        dtypes_warehouses = {"wid": np.int64, "shipping_cost": np.float32}

        artifact = dataframe_to_onnx(
            transform, [dtypes_orders, dtypes_customers, dtypes_products, dtypes_warehouses]
        )
        ref = ExtendedReferenceEvaluator(artifact)

        order_id = np.array([1, 2, 3, 4], dtype=np.int64)
        customer_id = np.array([10, 20, 10, 30], dtype=np.int64)
        product_id = np.array([100, 200, 300, 100], dtype=np.int64)
        warehouse_id = np.array([1000, 2000, 1000, 2000], dtype=np.int64)

        cid = np.array([10, 20, 30], dtype=np.int64)
        discount = np.array([0.1, 0.2, 0.0], dtype=np.float32)

        pid = np.array([100, 200, 300], dtype=np.int64)
        unit_price = np.array([50.0, 80.0, 60.0], dtype=np.float32)

        wid = np.array([1000, 2000], dtype=np.int64)
        shipping_cost = np.array([5.0, 8.0], dtype=np.float32)

        feeds = {
            "order_id": order_id,
            "customer_id": customer_id,
            "product_id": product_id,
            "warehouse_id": warehouse_id,
            "cid": cid,
            "discount": discount,
            "pid": pid,
            "unit_price": unit_price,
            "wid": wid,
            "shipping_cost": shipping_cost,
        }
        order_id_out, discount_out, unit_price_out, shipping_cost_out = ref.run(None, feeds)
        # All four orders must appear in the result (all match every dimension table).
        np.testing.assert_array_equal(order_id_out, np.array([1, 2, 3, 4]))
        np.testing.assert_allclose(discount_out, np.array([0.1, 0.2, 0.1, 0.0], dtype=np.float32))
        np.testing.assert_allclose(
            unit_price_out, np.array([50.0, 80.0, 60.0, 50.0], dtype=np.float32)
        )
        np.testing.assert_allclose(
            shipping_cost_out, np.array([5.0, 8.0, 5.0, 8.0], dtype=np.float32)
        )

    def test_join_three_consecutive_with_select(self):
        """Three consecutive joins followed by a computed SELECT expression."""

        def transform(orders, customers, products, warehouses):
            j1 = orders.join(customers, left_key="customer_id", right_key="cid")
            j2 = j1.join(products, left_key="product_id", right_key="pid")
            j3 = j2.join(warehouses, left_key="warehouse_id", right_key="wid")
            # total = qty * unit_price * (1 - discount) + shipping_cost
            total = (
                j3["qty"] * j3["unit_price"] * (1.0 - j3["discount"]) + j3["shipping_cost"]
            ).alias("total")
            return j3.select([total])

        dtypes_orders = {
            "order_id": np.int64,
            "customer_id": np.int64,
            "product_id": np.int64,
            "warehouse_id": np.int64,
            "qty": np.float32,
        }
        dtypes_customers = {"cid": np.int64, "discount": np.float32}
        dtypes_products = {"pid": np.int64, "unit_price": np.float32}
        dtypes_warehouses = {"wid": np.int64, "shipping_cost": np.float32}

        artifact = dataframe_to_onnx(
            transform, [dtypes_orders, dtypes_customers, dtypes_products, dtypes_warehouses]
        )
        ref = ExtendedReferenceEvaluator(artifact)

        order_id = np.array([1, 2, 3, 4], dtype=np.int64)
        customer_id = np.array([10, 20, 10, 30], dtype=np.int64)
        product_id = np.array([100, 200, 300, 100], dtype=np.int64)
        warehouse_id = np.array([1000, 2000, 1000, 2000], dtype=np.int64)
        qty = np.array([2.0, 1.0, 3.0, 1.0], dtype=np.float32)

        cid = np.array([10, 20, 30], dtype=np.int64)
        discount = np.array([0.1, 0.2, 0.0], dtype=np.float32)

        pid = np.array([100, 200, 300], dtype=np.int64)
        unit_price = np.array([50.0, 80.0, 60.0], dtype=np.float32)

        wid = np.array([1000, 2000], dtype=np.int64)
        shipping_cost = np.array([5.0, 8.0], dtype=np.float32)

        feeds = {
            "order_id": order_id,
            "customer_id": customer_id,
            "product_id": product_id,
            "warehouse_id": warehouse_id,
            "qty": qty,
            "cid": cid,
            "discount": discount,
            "pid": pid,
            "unit_price": unit_price,
            "wid": wid,
            "shipping_cost": shipping_cost,
        }
        (total_out,) = ref.run(None, feeds)
        # Manually verified totals:
        # order 1: 2*50*(1-0.1)+5  = 2*50*0.9+5  = 95
        # order 2: 1*80*(1-0.2)+8  = 1*80*0.8+8  = 72
        # order 3: 3*60*(1-0.1)+5  = 3*60*0.9+5  = 167
        # order 4: 1*50*(1-0.0)+8  = 1*50*1.0+8  = 58
        expected = np.array([95.0, 72.0, 167.0, 58.0], dtype=np.float32)
        np.testing.assert_allclose(total_out, expected, rtol=1e-5)

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

    def test_to_onnx_multi_frame_tuple_of_dataframes(self):
        """to_onnx() accepts a tuple of pandas DataFrames for multi-frame callables."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not available")
        from yobx.sql import to_onnx

        df1 = pd.DataFrame({"a": np.array([1.0, 2.0], dtype=np.float32)})
        df2 = pd.DataFrame({"b": np.array([3.0, 4.0], dtype=np.float32)})

        def transform(traced_df1, traced_df2):
            return traced_df1.select([(traced_df1["a"] + traced_df2["b"]).alias("total")])

        artifact = to_onnx(transform, (df1, df2))
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)

    def test_dataframe_to_onnx_tuple_of_dataframes(self):
        """dataframe_to_onnx() accepts a tuple of pandas DataFrames."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not available")

        df1 = pd.DataFrame({"a": np.array([1.0, 2.0], dtype=np.float32)})
        df2 = pd.DataFrame({"b": np.array([3.0, 4.0], dtype=np.float32)})

        def transform(traced_df1, traced_df2):
            return traced_df1.select([(traced_df1["a"] + traced_df2["b"]).alias("total")])

        artifact = dataframe_to_onnx(transform, (df1, df2))
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)

    def test_to_onnx_three_frames_tuple_of_dataframes(self):
        """to_onnx() accepts a tuple of three pandas DataFrames."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not available")
        from yobx.sql import to_onnx

        df1 = pd.DataFrame({"a": np.array([1.0, 2.0], dtype=np.float32)})
        df2 = pd.DataFrame({"b": np.array([3.0, 4.0], dtype=np.float32)})
        df3 = pd.DataFrame({"c": np.array([10.0, 20.0], dtype=np.float32)})

        def transform(traced_df1, traced_df2, traced_df3):
            return traced_df1.select(
                [(traced_df1["a"] + traced_df2["b"] + traced_df3["c"]).alias("total")]
            )

        artifact = to_onnx(transform, (df1, df2, df3))
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        c = np.array([10.0, 20.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b, "c": c})
        np.testing.assert_allclose(total, a + b + c)

    def test_to_onnx_numpy_single_array_in_tuple(self):
        """to_onnx(f, (arr,)) dispatches to trace_numpy_to_onnx."""
        from yobx.sql import to_onnx

        def my_func(x):
            return np.sqrt(np.abs(x) + 1)

        x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        artifact = to_onnx(my_func, (x,))
        ref = ExtendedReferenceEvaluator(artifact)
        (result,) = ref.run(None, {"X": x})
        np.testing.assert_allclose(result, my_func(x), rtol=1e-5)

    def test_to_onnx_numpy_single_array(self):
        """to_onnx(f, arr) dispatches to trace_numpy_to_onnx."""
        from yobx.sql import to_onnx

        def my_func(x):
            return x + 1.0

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        artifact = to_onnx(my_func, x)
        ref = ExtendedReferenceEvaluator(artifact)
        (result,) = ref.run(None, {"X": x})
        np.testing.assert_allclose(result, x + 1.0)

    def test_to_onnx_numpy_two_arrays_in_tuple(self):
        """to_onnx(f, (arr1, arr2)) dispatches to trace_numpy_to_onnx."""
        from yobx.sql import to_onnx

        def my_func(x, y):
            return x + y

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        artifact = to_onnx(my_func, (x, y))
        ref = ExtendedReferenceEvaluator(artifact)
        (result,) = ref.run(None, {"X0": x, "X1": y})
        np.testing.assert_allclose(result, x + y)


# ---------------------------------------------------------------------------
# parsed_query_to_onnx
# ---------------------------------------------------------------------------


class TestParsedQueryToOnnx(ExtTestCase):
    """Tests for :func:`~yobx.sql.sql_convert.parsed_query_to_onnx`."""

    def test_basic_select_add(self):
        from yobx.xtracing.dataframe_trace import trace_dataframe
        from yobx.sql.sql_convert import parsed_query_to_onnx

        def transform(df):
            return df.select([(df["a"] + df["b"]).alias("total")])

        pq = trace_dataframe(transform, {"a": np.float32, "b": np.float32})
        artifact = parsed_query_to_onnx(pq)
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)

    def test_with_filter(self):
        from yobx.xtracing.dataframe_trace import trace_dataframe
        from yobx.sql.sql_convert import parsed_query_to_onnx

        def transform(df):
            df = df.filter(df["a"] > 0)
            return df.select([(df["a"] + df["b"]).alias("total")])

        pq = trace_dataframe(transform, {"a": np.float32, "b": np.float32})
        artifact = parsed_query_to_onnx(pq)
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
# Multiple output dataframes
# ---------------------------------------------------------------------------


class TestMultipleOutputDataframes(ExtTestCase):
    """Tests for functions that return multiple :class:`TracedDataFrame` outputs."""

    def test_trace_dataframe_tuple_output(self):
        """trace_dataframe returns a list of ParsedQuery when func returns a tuple."""
        from yobx.xtracing.dataframe_trace import trace_dataframe
        from yobx.xtracing.parse import ParsedQuery

        def transform(df):
            out1 = df.select([(df["a"] + df["b"]).alias("sum_ab")])
            out2 = df.select([(df["a"] - df["b"]).alias("diff_ab")])
            return out1, out2

        result = trace_dataframe(transform, {"a": np.float32, "b": np.float32})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for pq in result:
            self.assertIsInstance(pq, ParsedQuery)

    def test_trace_dataframe_list_output(self):
        """trace_dataframe returns a list of ParsedQuery when func returns a list."""
        from yobx.xtracing.dataframe_trace import trace_dataframe
        from yobx.xtracing.parse import ParsedQuery

        def transform(df):
            return [
                df.select([(df["a"] * 2).alias("double_a")]),
                df.select([(df["b"] * 3).alias("triple_b")]),
            ]

        result = trace_dataframe(transform, {"a": np.float32, "b": np.float32})
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for pq in result:
            self.assertIsInstance(pq, ParsedQuery)

    def test_trace_dataframe_bad_tuple_item_raises(self):
        """trace_dataframe raises TypeError when tuple contains non-TracedDataFrame."""
        from yobx.xtracing.dataframe_trace import trace_dataframe

        def bad_transform(df):
            return df.select([(df["a"] + df["b"]).alias("sum")]), 42

        with self.assertRaises(TypeError):
            trace_dataframe(bad_transform, {"a": np.float32, "b": np.float32})

    def test_dataframe_to_onnx_two_outputs(self):
        """dataframe_to_onnx handles a function that returns two dataframes."""
        from yobx.sql import dataframe_to_onnx

        def transform(df):
            out1 = df.select([(df["a"] + df["b"]).alias("sum_ab")])
            out2 = df.select([(df["a"] - df["b"]).alias("diff_ab")])
            return out1, out2

        artifact = dataframe_to_onnx(transform, {"a": np.float32, "b": np.float32})
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        results = ref.run(None, {"a": a, "b": b})
        self.assertEqual(len(results), 2)
        np.testing.assert_allclose(results[0], a + b)
        np.testing.assert_allclose(results[1], a - b)

    def test_dataframe_to_onnx_three_outputs(self):
        """dataframe_to_onnx handles a function that returns three dataframes."""
        from yobx.sql import dataframe_to_onnx

        def transform(df):
            out1 = df.select([(df["a"] + df["b"]).alias("sum_ab")])
            out2 = df.select([(df["a"] - df["b"]).alias("diff_ab")])
            out3 = df.select([(df["a"] * df["b"]).alias("prod_ab")])
            return out1, out2, out3

        artifact = dataframe_to_onnx(transform, {"a": np.float32, "b": np.float32})
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        results = ref.run(None, {"a": a, "b": b})
        self.assertEqual(len(results), 3)
        np.testing.assert_allclose(results[0], a + b)
        np.testing.assert_allclose(results[1], a - b)
        np.testing.assert_allclose(results[2], a * b)

    def test_dataframe_to_onnx_two_outputs_with_filter(self):
        """Multiple output dataframes — one with a filter applied."""
        from yobx.sql import dataframe_to_onnx

        def transform(df):
            out1 = df.filter(df["a"] > 0).select([(df["a"] + df["b"]).alias("sum_pos")])
            out2 = df.select([(df["a"] * df["b"]).alias("prod_ab")])
            return out1, out2

        artifact = dataframe_to_onnx(transform, {"a": np.float32, "b": np.float32})
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        results = ref.run(None, {"a": a, "b": b})
        self.assertEqual(len(results), 2)
        np.testing.assert_allclose(results[0], np.array([5.0, 9.0], dtype=np.float32))
        np.testing.assert_allclose(results[1], a * b)

    def test_dataframe_to_onnx_multi_input_multi_output(self):
        """Multiple input frames and multiple output frames."""
        from yobx.sql import dataframe_to_onnx

        def transform(df1, df2):
            out1 = df1.select([(df1["a"] + df2["b"]).alias("sum_ab")])
            out2 = df1.select([(df1["a"] - df2["b"]).alias("diff_ab")])
            return out1, out2

        artifact = dataframe_to_onnx(transform, [{"a": np.float32}, {"b": np.float32}])
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        results = ref.run(None, {"a": a, "b": b})
        self.assertEqual(len(results), 2)
        np.testing.assert_allclose(results[0], a + b)
        np.testing.assert_allclose(results[1], a - b)

    def test_to_onnx_multiple_outputs(self):
        """to_onnx() dispatches callable returning multiple dataframes correctly."""
        from yobx.sql import to_onnx

        def transform(df):
            out1 = df.select([(df["x"] + 1).alias("x_plus_1")])
            out2 = df.select([(df["x"] * 2).alias("x_times_2")])
            return out1, out2

        artifact = to_onnx(transform, {"x": np.float32})
        ref = ExtendedReferenceEvaluator(artifact)
        x = np.array([5.0, 6.0], dtype=np.float32)
        results = ref.run(None, {"x": x})
        self.assertEqual(len(results), 2)
        np.testing.assert_allclose(results[0], x + 1)
        np.testing.assert_allclose(results[1], x * 2)

    def test_parsed_query_to_onnx_list_of_queries(self):
        """parsed_query_to_onnx accepts a list of ParsedQuery objects."""
        from yobx.xtracing.dataframe_trace import trace_dataframe
        from yobx.sql.sql_convert import parsed_query_to_onnx

        def transform(df):
            out1 = df.select([(df["a"] + df["b"]).alias("sum_ab")])
            out2 = df.select([(df["a"] - df["b"]).alias("diff_ab")])
            return out1, out2

        pqs = trace_dataframe(transform, {"a": np.float32, "b": np.float32})
        self.assertIsInstance(pqs, list)
        artifact = parsed_query_to_onnx(pqs)
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        results = ref.run(None, {"a": a, "b": b})
        self.assertEqual(len(results), 2)
        np.testing.assert_allclose(results[0], a + b)
        np.testing.assert_allclose(results[1], a - b)

    def test_trace_dataframe_empty_tuple_raises(self):
        """trace_dataframe raises ValueError when func returns an empty tuple."""
        from yobx.xtracing.dataframe_trace import trace_dataframe

        def empty_transform(df):
            return ()

        with self.assertRaises(ValueError):
            trace_dataframe(empty_transform, {"a": np.float32})

    def test_parsed_query_to_onnx_empty_list_raises(self):
        """parsed_query_to_onnx raises ValueError when given an empty list."""
        from yobx.sql.sql_convert import parsed_query_to_onnx

        with self.assertRaises(ValueError):
            parsed_query_to_onnx([])


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

    def test_df_sub_dataframe_onnx(self):
        """df1 - df2 subtracts matching columns element-wise (same column names)."""

        def transform(df1, df2):
            return df1 - df2

        a = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out_a, out_b = _run_multi(
            transform,
            [{"a": np.float32, "b": np.float32}, {"a": np.float32, "b": np.float32}],
            {"a": a, "b": b},
        )
        np.testing.assert_allclose(out_a, a - a, rtol=1e-5)
        np.testing.assert_allclose(out_b, b - b, rtol=1e-5)

    def test_df_mul_dataframe_onnx(self):
        """df1 * df2 multiplies matching columns element-wise (same column names)."""

        def transform(df1, df2):
            return df1 * df2

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (out_a,) = _run_multi(transform, [{"a": np.float32}, {"a": np.float32}], {"a": a})
        np.testing.assert_allclose(out_a, a * a, rtol=1e-5)

    def test_df_div_dataframe_onnx(self):
        """df1 / df2 divides matching columns element-wise (same column names)."""

        def transform(df1, df2):
            return df1 / df2

        a = np.array([2.0, 4.0, 8.0], dtype=np.float32)
        (out_a,) = _run_multi(transform, [{"a": np.float32}, {"a": np.float32}], {"a": a})
        np.testing.assert_allclose(out_a, a / a, rtol=1e-5)

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

    def test_to_onnx_callable_with_tuple_of_dataframe(self):
        """to_onnx(callable, (DataFrame,)) works — the primary form requested in the issue."""
        from yobx.sql import to_onnx

        df = self._make_df()
        if df is None:
            self.skipTest("pandas not available")

        def transform(traced_df):
            traced_df = traced_df.filter(traced_df["a"] > 0)
            return traced_df.select([(traced_df["a"] + traced_df["b"]).alias("total")])

        # tuple wrapping a single DataFrame — should behave identically to passing df directly
        artifact = to_onnx(transform, (df,))
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, np.array([5.0, 9.0], dtype=np.float32))

    def test_to_onnx_callable_with_tuple_of_two_dataframes(self):
        """to_onnx(callable, (df1, df2)) works for multi-frame transforms."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not available")

        from yobx.sql import to_onnx

        df1 = pd.DataFrame({"a": np.array([1.0, 2.0], dtype=np.float32)})
        df2 = pd.DataFrame({"b": np.array([3.0, 4.0], dtype=np.float32)})

        def transform(traced_df1, traced_df2):
            return traced_df1.select([(traced_df1["a"] + traced_df2["b"]).alias("total")])

        artifact = to_onnx(transform, (df1, df2))
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)


if __name__ == "__main__":
    unittest.main()
