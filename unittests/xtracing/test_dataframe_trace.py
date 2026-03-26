import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase
from yobx.sql import (
    FilterOp,
    GroupByOp,
    ParsedQuery,
    SelectOp,
    TracedCondition,
    TracedDataFrame,
    TracedGroupBy,
    TracedSeries,
    trace_dataframe,
)
from yobx.xtracing.dataframe_trace import _to_ast
from yobx.xtracing.parse import AggExpr, BinaryExpr, ColumnRef, Condition, Literal


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

    def test_traced_series(self):
        s = TracedSeries(ColumnRef("a"))
        node = _to_ast(s)
        self.assertIsInstance(node, ColumnRef)
        self.assertEqual(node.column, "a")

    def test_unsupported_raises(self):
        with self.assertRaises(TypeError):
            _to_ast("bad_type")


# ---------------------------------------------------------------------------
# TracedSeries
# ---------------------------------------------------------------------------


class TestTracedSeriesArithmetic(ExtTestCase):
    def setUp(self):
        self.a = TracedSeries(ColumnRef("a"))
        self.b = TracedSeries(ColumnRef("b"))

    def test_add(self):
        result = self.a + self.b
        self.assertIsInstance(result, TracedSeries)
        self.assertIsInstance(result._expr, BinaryExpr)
        self.assertEqual(result._expr.op, "+")

    def test_radd(self):
        result = 2.0 + self.a
        self.assertIsInstance(result, TracedSeries)
        self.assertIsInstance(result._expr.left, Literal)

    def test_sub(self):
        result = self.a - self.b
        self.assertEqual(result._expr.op, "-")

    def test_rsub(self):
        result = 1.0 - self.a
        self.assertIsInstance(result._expr.left, Literal)
        self.assertEqual(result._expr.op, "-")

    def test_mul(self):
        result = self.a * self.b
        self.assertEqual(result._expr.op, "*")

    def test_rmul(self):
        result = 3 * self.a
        self.assertIsInstance(result._expr.left, Literal)

    def test_div(self):
        result = self.a / self.b
        self.assertEqual(result._expr.op, "/")

    def test_rdiv(self):
        result = 6.0 / self.a
        self.assertIsInstance(result._expr.left, Literal)
        self.assertEqual(result._expr.op, "/")

    def test_alias(self):
        result = self.a.alias("out")
        self.assertEqual(result._alias, "out")
        self.assertIsInstance(result._expr, ColumnRef)

    def test_to_select_item(self):
        item = self.a.alias("out").to_select_item()
        self.assertEqual(item.alias, "out")
        self.assertIsInstance(item.expr, ColumnRef)

    def test_repr(self):
        s = self.a.alias("out")
        r = repr(s)
        self.assertIn("TracedSeries", r)
        self.assertIn("out", r)

    def test_repr_no_alias(self):
        r = repr(self.a)
        self.assertIn("TracedSeries", r)


class TestTracedSeriesComparison(ExtTestCase):
    def setUp(self):
        self.a = TracedSeries(ColumnRef("a"))

    def _check_condition(self, cond, op):
        self.assertIsInstance(cond, TracedCondition)
        self.assertIsInstance(cond._condition, Condition)
        self.assertEqual(cond._condition.op, op)

    def test_gt(self):
        self._check_condition(self.a > 0, ">")

    def test_lt(self):
        self._check_condition(self.a < 0, "<")

    def test_ge(self):
        self._check_condition(self.a >= 0, ">=")

    def test_le(self):
        self._check_condition(self.a <= 0, "<=")

    def test_eq(self):
        self._check_condition(self.a == 1, "=")

    def test_ne(self):
        self._check_condition(self.a != 1, "<>")


class TestTracedSeriesAggregations(ExtTestCase):
    def setUp(self):
        self.v = TracedSeries(ColumnRef("v"))

    def _check_agg(self, series, func_name):
        self.assertIsInstance(series, TracedSeries)
        self.assertIsInstance(series._expr, AggExpr)
        self.assertEqual(series._expr.func, func_name)

    def test_sum(self):
        self._check_agg(self.v.sum(), "sum")

    def test_mean(self):
        self._check_agg(self.v.mean(), "avg")

    def test_min(self):
        self._check_agg(self.v.min(), "min")

    def test_max(self):
        self._check_agg(self.v.max(), "max")

    def test_count(self):
        self._check_agg(self.v.count(), "count")


# ---------------------------------------------------------------------------
# TracedCondition
# ---------------------------------------------------------------------------


class TestTracedCondition(ExtTestCase):
    def setUp(self):
        a = TracedSeries(ColumnRef("a"))
        b = TracedSeries(ColumnRef("b"))
        self.ca = a > 0
        self.cb = b < 10

    def test_and(self):
        combined = self.ca & self.cb
        self.assertIsInstance(combined, TracedCondition)
        self.assertEqual(combined._condition.op, "and")

    def test_or(self):
        combined = self.ca | self.cb
        self.assertIsInstance(combined, TracedCondition)
        self.assertEqual(combined._condition.op, "or")

    def test_bool_raises(self):
        with self.assertRaises(TypeError):
            bool(self.ca)

    def test_repr(self):
        r = repr(self.ca)
        self.assertIn("TracedCondition", r)


# ---------------------------------------------------------------------------
# TracedDataFrame construction
# ---------------------------------------------------------------------------


class TestTracedDataFrameConstruction(ExtTestCase):
    def _make(self, cols=("a", "b")):
        columns = {c: TracedSeries(ColumnRef(c)) for c in cols}
        return TracedDataFrame(columns)

    def test_columns_property(self):
        df = self._make(["a", "b"])
        self.assertEqual(df.columns, ["a", "b"])

    def test_getitem_str(self):
        df = self._make(["a"])
        s = df["a"]
        self.assertIsInstance(s, TracedSeries)

    def test_getitem_str_missing_raises(self):
        df = self._make(["a"])
        with self.assertRaises(KeyError):
            _ = df["z"]

    def test_getattr_col(self):
        df = self._make(["a"])
        s = df.a
        self.assertIsInstance(s, TracedSeries)

    def test_getattr_missing_raises(self):
        df = self._make(["a"])
        with self.assertRaises(AttributeError):
            _ = df.z

    def test_getitem_list(self):
        df = self._make(["a", "b"])
        sub = df[["a"]]
        self.assertIsInstance(sub, TracedDataFrame)
        self.assertEqual(sub.columns, ["a"])

    def test_getitem_condition(self):
        df = self._make(["a"])
        filtered = df[df["a"] > 0]
        self.assertIsInstance(filtered, TracedDataFrame)
        self.assertTrue(any(isinstance(op, FilterOp) for op in filtered._ops))

    def test_repr(self):
        df = self._make(["a", "b"])
        r = repr(df)
        self.assertIn("TracedDataFrame", r)


# ---------------------------------------------------------------------------
# TracedDataFrame.filter
# ---------------------------------------------------------------------------


class TestTracedDataFrameFilter(ExtTestCase):
    def _make(self):
        columns = {c: TracedSeries(ColumnRef(c)) for c in ("a", "b")}
        return TracedDataFrame(columns, source_columns=["a", "b"])

    def test_filter_records_op(self):
        df = self._make()
        result = df.filter(df["a"] > 0)
        self.assertEqual(len(result._ops), 1)
        self.assertIsInstance(result._ops[0], FilterOp)

    def test_filter_chaining(self):
        df = self._make()
        result = df.filter(df["a"] > 0).filter(df["b"] < 10)
        filter_ops = [op for op in result._ops if isinstance(op, FilterOp)]
        self.assertEqual(len(filter_ops), 2)


# ---------------------------------------------------------------------------
# TracedDataFrame.select
# ---------------------------------------------------------------------------


class TestTracedDataFrameSelect(ExtTestCase):
    def _make(self):
        columns = {c: TracedSeries(ColumnRef(c)) for c in ("a", "b")}
        return TracedDataFrame(columns, source_columns=["a", "b"])

    def test_select_list_of_strings(self):
        df = self._make()
        result = df.select(["a"])
        self.assertEqual(result.columns, ["a"])
        select_ops = [op for op in result._ops if isinstance(op, SelectOp)]
        self.assertEqual(len(select_ops), 1)

    def test_select_list_of_series(self):
        df = self._make()
        result = df.select([(df["a"] + df["b"]).alias("total")])
        self.assertEqual(result.columns, ["total"])

    def test_select_dict(self):
        df = self._make()
        result = df.select({"total": df["a"] + df["b"]})
        self.assertEqual(result.columns, ["total"])

    def test_select_unknown_str_raises(self):
        df = self._make()
        with self.assertRaises(KeyError):
            df.select(["z"])

    def test_select_bad_type_raises(self):
        df = self._make()
        with self.assertRaises(TypeError):
            df.select([42])  # type: ignore[list-item]

    def test_select_bad_exprs_raises(self):
        df = self._make()
        with self.assertRaises(TypeError):
            df.select(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TracedDataFrame.assign
# ---------------------------------------------------------------------------


class TestTracedDataFrameAssign(ExtTestCase):
    def _make(self):
        columns = {c: TracedSeries(ColumnRef(c)) for c in ("a", "b")}
        return TracedDataFrame(columns, source_columns=["a", "b"])

    def test_assign_adds_column(self):
        df = self._make()
        result = df.assign(total=df["a"] + df["b"])
        self.assertIn("total", result.columns)
        self.assertIn("a", result.columns)

    def test_assign_bad_value_raises(self):
        df = self._make()
        with self.assertRaises(TypeError):
            df.assign(bad=123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TracedDataFrame.pipe
# ---------------------------------------------------------------------------


class TestTracedDataFramePipe(ExtTestCase):
    def _make(self):
        columns = {c: TracedSeries(ColumnRef(c)) for c in ("a", "b")}
        return TracedDataFrame(columns, source_columns=["a", "b"])

    def test_pipe_returns_traced_dataframe(self):
        df = self._make()

        def identity(d):
            return d

        result = df.pipe(identity)
        self.assertIsInstance(result, TracedDataFrame)

    def test_pipe_applies_function(self):
        df = self._make()

        def add_col(d):
            return d.assign(total=d["a"] + d["b"])

        result = df.pipe(add_col)
        self.assertIn("total", result.columns)

    def test_pipe_chaining(self):
        df = self._make()

        def step1(d):
            return d.assign(total=d["a"] + d["b"])

        def step2(d):
            return d.select(["total"])

        result = df.pipe(step1).pipe(step2)
        self.assertEqual(result.columns, ["total"])

    def test_pipe_with_extra_args(self):
        df = self._make()

        def scale(d, factor):
            return d.assign(a=(d["a"] * factor).alias("a"))

        result = df.pipe(scale, factor=2.0)
        self.assertIn("a", result.columns)


# ---------------------------------------------------------------------------
# TracedDataFrame.copy
# ---------------------------------------------------------------------------


class TestTracedDataFrameCopy(ExtTestCase):
    def _make(self):
        columns = {c: TracedSeries(ColumnRef(c)) for c in ("a", "b")}
        return TracedDataFrame(columns, source_columns=["a", "b"])

    def test_copy_returns_traced_dataframe(self):
        df = self._make()
        result = df.copy()
        self.assertIsInstance(result, TracedDataFrame)

    def test_copy_is_new_object(self):
        df = self._make()
        result = df.copy()
        self.assertIsNot(result, df)

    def test_copy_preserves_columns(self):
        df = self._make()
        result = df.copy()
        self.assertEqual(result.columns, df.columns)

    def test_copy_deep_false_preserves_columns(self):
        df = self._make()
        result = df.copy(deep=False)
        self.assertEqual(result.columns, df.columns)

    def test_copy_does_not_share_ops_list(self):
        df = self._make()
        result = df.copy()
        result._ops.append("sentinel")
        self.assertNotIn("sentinel", df._ops)

    def test_copy_in_pipeline(self):
        """Verify that .copy() inside a traced function does not break tracing."""
        import numpy as np
        from yobx.sql import dataframe_to_onnx

        def transform(df):
            df2 = df.copy()
            return df2.assign(total=(df2["a"] + df2["b"]).alias("total")).select(["total"])

        dtypes = {"a": np.float32, "b": np.float32}
        artifact = dataframe_to_onnx(transform, dtypes)
        self.assertIsNotNone(artifact)


# ---------------------------------------------------------------------------
# TracedDataFrame.groupby / TracedGroupBy.agg
# ---------------------------------------------------------------------------


class TestTracedGroupBy(ExtTestCase):
    def _make(self):
        columns = {c: TracedSeries(ColumnRef(c)) for c in ("k", "v")}
        return TracedDataFrame(columns, source_columns=["k", "v"])

    def test_groupby_returns_traced_groupby(self):
        df = self._make()
        gb = df.groupby("k")
        self.assertIsInstance(gb, TracedGroupBy)
        self.assertEqual(gb._by, ["k"])

    def test_groupby_list(self):
        df = self._make()
        gb = df.groupby(["k"])
        self.assertEqual(gb._by, ["k"])

    def test_agg_list(self):
        df = self._make()
        result = df.groupby("k").agg([df["v"].sum().alias("total")])
        self.assertIsInstance(result, TracedDataFrame)
        group_ops = [op for op in result._ops if isinstance(op, GroupByOp)]
        select_ops = [op for op in result._ops if isinstance(op, SelectOp)]
        self.assertEqual(len(group_ops), 1)
        self.assertEqual(len(select_ops), 1)
        self.assertEqual(group_ops[0].columns, ["k"])

    def test_agg_dict(self):
        df = self._make()
        result = df.groupby("k").agg({"total": df["v"].sum()})
        self.assertEqual(result.columns, ["total"])


# ---------------------------------------------------------------------------
# TracedDataFrame.to_parsed_query
# ---------------------------------------------------------------------------


class TestToParsedQuery(ExtTestCase):
    def _make(self):
        columns = {c: TracedSeries(ColumnRef(c)) for c in ("a", "b")}
        return TracedDataFrame(columns, source_columns=["a", "b"])

    def test_no_select_appends_passthrough(self):
        df = self._make()
        pq = df.to_parsed_query()
        self.assertIsInstance(pq, ParsedQuery)
        select_ops = [op for op in pq.operations if isinstance(op, SelectOp)]
        self.assertEqual(len(select_ops), 1)

    def test_with_filter_and_select(self):
        df = self._make()
        result = df.filter(df["a"] > 0).select([(df["a"] + df["b"]).alias("total")])
        pq = result.to_parsed_query()
        filter_ops = [op for op in pq.operations if isinstance(op, FilterOp)]
        select_ops = [op for op in pq.operations if isinstance(op, SelectOp)]
        self.assertEqual(len(filter_ops), 1)
        self.assertEqual(len(select_ops), 1)

    def test_columns_collected(self):
        df = self._make()
        result = df.select([(df["a"] + df["b"]).alias("total")])
        pq = result.to_parsed_query()
        col_names = [c.column for c in pq.columns]
        self.assertIn("a", col_names)
        self.assertIn("b", col_names)


# ---------------------------------------------------------------------------
# trace_dataframe
# ---------------------------------------------------------------------------


class TestTraceDataframe(ExtTestCase):
    def test_returns_parsed_query(self):
        def transform(df):
            return df.select([(df["a"] + df["b"]).alias("total")])

        pq = trace_dataframe(transform, {"a": np.float32, "b": np.float32})
        self.assertIsInstance(pq, ParsedQuery)

    def test_non_dataframe_return_raises(self):
        def bad(df):
            return 42  # type: ignore[return-value]

        with self.assertRaises(TypeError):
            trace_dataframe(bad, {"a": np.float32})

    # ------------------------------------------------------------------
    # Public imports
    # ------------------------------------------------------------------

    def test_imported_from_sql_package(self):
        from yobx.sql import dataframe_to_onnx as dtonnx  # noqa: F401

        self.assertTrue(callable(dtonnx))

    def test_trace_dataframe_imported_from_sql_package(self):
        from yobx.sql import trace_dataframe as td  # noqa: F401

        self.assertTrue(callable(td))

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


class TestDataframeArithmetic(ExtTestCase):
    """Tests for element-wise arithmetic operators on :class:`TracedDataFrame`."""

    def test_add_scalar_returns_dataframe(self):
        df = TracedDataFrame({"a": TracedSeries(ColumnRef("a"))})
        result = df + 1
        self.assertIsInstance(result, TracedDataFrame)
        self.assertIn("a", result.columns)
        expr = result._columns["a"]._expr
        self.assertIsInstance(expr, BinaryExpr)
        self.assertEqual(expr.op, "+")

    def test_sub_scalar_returns_dataframe(self):
        df = TracedDataFrame({"a": TracedSeries(ColumnRef("a"))})
        result = df - 1
        self.assertIsInstance(result, TracedDataFrame)
        expr = result._columns["a"]._expr
        self.assertIsInstance(expr, BinaryExpr)
        self.assertEqual(expr.op, "-")

    def test_mul_scalar_returns_dataframe(self):
        df = TracedDataFrame({"a": TracedSeries(ColumnRef("a"))})
        result = df * 2
        expr = result._columns["a"]._expr
        self.assertIsInstance(expr, BinaryExpr)
        self.assertEqual(expr.op, "*")

    def test_div_scalar_returns_dataframe(self):
        df = TracedDataFrame({"a": TracedSeries(ColumnRef("a"))})
        result = df / 2
        expr = result._columns["a"]._expr
        self.assertIsInstance(expr, BinaryExpr)
        self.assertEqual(expr.op, "/")

    def test_radd_scalar_reversed(self):
        df = TracedDataFrame({"a": TracedSeries(ColumnRef("a"))})
        result = 1 + df
        expr = result._columns["a"]._expr
        self.assertIsInstance(expr, BinaryExpr)
        self.assertEqual(expr.op, "+")
        self.assertIsInstance(expr.left, Literal)

    def test_rsub_scalar_reversed(self):
        df = TracedDataFrame({"a": TracedSeries(ColumnRef("a"))})
        result = 10 - df
        expr = result._columns["a"]._expr
        self.assertIsInstance(expr, BinaryExpr)
        self.assertEqual(expr.op, "-")
        self.assertIsInstance(expr.left, Literal)

    def test_rmul_scalar_reversed(self):
        df = TracedDataFrame({"a": TracedSeries(ColumnRef("a"))})
        result = 3 * df
        expr = result._columns["a"]._expr
        self.assertIsInstance(expr, BinaryExpr)
        self.assertEqual(expr.op, "*")
        self.assertIsInstance(expr.left, Literal)

    def test_rtruediv_scalar_reversed(self):
        df = TracedDataFrame({"a": TracedSeries(ColumnRef("a"))})
        result = 1.0 / df
        expr = result._columns["a"]._expr
        self.assertIsInstance(expr, BinaryExpr)
        self.assertEqual(expr.op, "/")
        self.assertIsInstance(expr.left, Literal)

    def test_add_dataframe_column_wise(self):
        df1 = TracedDataFrame({"a": TracedSeries(ColumnRef("a"))})
        df2 = TracedDataFrame({"a": TracedSeries(ColumnRef("b"))})
        result = df1 + df2
        expr = result._columns["a"]._expr
        self.assertIsInstance(expr, BinaryExpr)
        self.assertEqual(expr.op, "+")

    def test_add_dataframe_missing_column_raises(self):
        df1 = TracedDataFrame({"a": TracedSeries(ColumnRef("a"))})
        df2 = TracedDataFrame({"b": TracedSeries(ColumnRef("b"))})
        with self.assertRaises(KeyError):
            _ = df1 + df2

    def test_ops_list_unchanged(self):
        """Arithmetic should not add SelectOp to the ops list."""
        df = TracedDataFrame({"a": TracedSeries(ColumnRef("a"))})
        result = df + 1
        self.assertEqual(len(result._ops), 0)

    def test_source_columns_preserved(self):
        df = TracedDataFrame({"a": TracedSeries(ColumnRef("a"))}, source_columns=["a"])
        result = df + 1
        self.assertEqual(result._source_columns, ["a"])


# ---------------------------------------------------------------------------
# ColumnRef dtype field
# ---------------------------------------------------------------------------


class TestColumnRefDtype(ExtTestCase):
    def test_dtype_defaults_to_zero(self):
        cr = ColumnRef("a")
        self.assertEqual(cr.dtype, 0)

    def test_dtype_stored_as_tensor_proto_int(self):
        from onnx import TensorProto

        cr = ColumnRef("a", dtype=TensorProto.FLOAT)
        self.assertEqual(cr.dtype, TensorProto.FLOAT)

    def test_dtype_int64(self):
        from onnx import TensorProto

        cr = ColumnRef("a", dtype=TensorProto.INT64)
        self.assertEqual(cr.dtype, TensorProto.INT64)

    def test_dtype_independent_of_table(self):
        from onnx import TensorProto

        cr = ColumnRef("col", table="tbl", dtype=TensorProto.INT32)
        self.assertEqual(cr.table, "tbl")
        self.assertEqual(cr.dtype, TensorProto.INT32)

    def test_trace_dataframe_propagates_dtype(self):
        """ColumnRef nodes produced by trace_dataframe carry the TensorProto dtype."""
        from onnx import TensorProto
        from yobx.xtracing.parse import SelectOp

        def transform(df):
            return df.select([df["a"].alias("a")])

        pq = trace_dataframe(transform, {"a": np.float32, "b": np.int64})
        select_op = next(op for op in pq.operations if isinstance(op, SelectOp))
        col_ref = select_op.items[0].expr
        self.assertIsInstance(col_ref, ColumnRef)
        self.assertEqual(col_ref.column, "a")
        self.assertEqual(col_ref.dtype, TensorProto.FLOAT)

    def test_trace_dataframe_multi_input_propagates_dtype(self):
        """Dtypes are propagated for multi-input trace_dataframe calls."""
        from onnx import TensorProto
        from yobx.xtracing.parse import SelectOp

        def transform(df1, df2):
            return df1.select([df1["x"].alias("x")])

        pq = trace_dataframe(transform, [{"x": np.float64}, {"y": np.int32}])
        select_op = next(op for op in pq.operations if isinstance(op, SelectOp))
        col_ref = select_op.items[0].expr
        self.assertIsInstance(col_ref, ColumnRef)
        self.assertEqual(col_ref.dtype, TensorProto.DOUBLE)


if __name__ == "__main__":
    unittest.main()
