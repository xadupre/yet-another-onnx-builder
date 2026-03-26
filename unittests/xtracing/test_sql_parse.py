"""
Unit tests for :mod:`yobx.sql.parse`.
"""

import unittest

from yobx.xtracing.parse import (
    AggExpr,
    BinaryExpr,
    ColumnRef,
    Condition,
    FilterOp,
    GroupByOp,
    JoinOp,
    SelectOp,
    parse_sql,
)


class TestParseSqlSelect(unittest.TestCase):
    """Tests for the SELECT clause parsing."""

    def test_select_single_column(self):
        pq = parse_sql("SELECT a FROM t")
        self.assertEqual(pq.from_table, "t")
        self.assertEqual(len(pq.operations), 1)
        sel = pq.operations[0]
        self.assertIsInstance(sel, SelectOp)
        self.assertEqual(len(sel.items), 1)
        self.assertIsInstance(sel.items[0].expr, ColumnRef)
        self.assertEqual(sel.items[0].expr.column, "a")

    def test_select_multiple_columns(self):
        pq = parse_sql("SELECT a, b, c FROM t")
        sel = pq.operations[0]
        self.assertIsInstance(sel, SelectOp)
        self.assertEqual(len(sel.items), 3)
        names = [item.expr.column for item in sel.items]
        self.assertEqual(names, ["a", "b", "c"])

    def test_select_with_alias(self):
        pq = parse_sql("SELECT a AS alpha FROM t")
        sel = pq.operations[0]
        self.assertEqual(sel.items[0].alias, "alpha")
        self.assertEqual(sel.items[0].output_name(), "alpha")

    def test_select_arithmetic_expression(self):
        pq = parse_sql("SELECT a + b AS total FROM t")
        sel = pq.operations[0]
        item = sel.items[0]
        self.assertIsInstance(item.expr, BinaryExpr)
        self.assertEqual(item.expr.op, "+")
        self.assertEqual(item.alias, "total")

    def test_select_distinct(self):
        pq = parse_sql("SELECT DISTINCT a FROM t")
        sel = pq.operations[0]
        self.assertTrue(sel.distinct)

    def test_select_case_insensitive(self):
        pq = parse_sql("select A, B from T")
        sel = pq.operations[0]
        names = [item.expr.column for item in sel.items]
        self.assertEqual(names, ["a", "b"])
        self.assertEqual(pq.from_table, "t")

    def test_select_aggregation_sum(self):
        pq = parse_sql("SELECT SUM(a) FROM t")
        sel = pq.operations[0]
        item = sel.items[0]
        self.assertIsInstance(item.expr, AggExpr)
        self.assertEqual(item.expr.func, "sum")

    def test_select_aggregation_count_star(self):
        pq = parse_sql("SELECT COUNT(*) FROM t")
        sel = pq.operations[0]
        item = sel.items[0]
        self.assertIsInstance(item.expr, AggExpr)
        self.assertEqual(item.expr.func, "count")
        self.assertEqual(item.expr.arg, "*")

    def test_select_aggregation_avg(self):
        pq = parse_sql("SELECT AVG(score) FROM t")
        sel = pq.operations[0]
        item = sel.items[0]
        self.assertIsInstance(item.expr, AggExpr)
        self.assertEqual(item.expr.func, "avg")

    def test_select_item_output_name_column(self):
        pq = parse_sql("SELECT revenue FROM t")
        item = pq.operations[0].items[0]
        self.assertEqual(item.output_name(), "revenue")

    def test_select_item_output_name_agg(self):
        pq = parse_sql("SELECT SUM(x) FROM t")
        item = pq.operations[0].items[0]
        self.assertEqual(item.output_name(), "sum_x")


class TestParseSqlFilter(unittest.TestCase):
    """Tests for the WHERE clause parsing."""

    def test_where_simple_comparison(self):
        pq = parse_sql("SELECT a FROM t WHERE a > 0")
        # operations: FilterOp, SelectOp
        self.assertEqual(len(pq.operations), 2)
        filt = pq.operations[0]
        self.assertIsInstance(filt, FilterOp)
        cond = filt.condition
        self.assertIsInstance(cond, Condition)
        self.assertEqual(cond.op, ">")

    def test_where_equal(self):
        pq = parse_sql("SELECT a FROM t WHERE b = 1")
        filt = pq.operations[0]
        self.assertEqual(filt.condition.op, "=")

    def test_where_not_equal(self):
        pq = parse_sql("SELECT a FROM t WHERE b <> 0")
        filt = pq.operations[0]
        self.assertEqual(filt.condition.op, "<>")

    def test_where_and(self):
        pq = parse_sql("SELECT a FROM t WHERE a > 0 AND b < 10")
        filt = pq.operations[0]
        cond = filt.condition
        self.assertIsInstance(cond, Condition)
        self.assertEqual(cond.op, "and")

    def test_where_or(self):
        pq = parse_sql("SELECT a FROM t WHERE a > 0 OR b < 0")
        filt = pq.operations[0]
        cond = filt.condition
        self.assertEqual(cond.op, "or")

    def test_where_less_equal(self):
        pq = parse_sql("SELECT a FROM t WHERE a <= 5")
        filt = pq.operations[0]
        self.assertEqual(filt.condition.op, "<=")

    def test_where_greater_equal(self):
        pq = parse_sql("SELECT a FROM t WHERE a >= 5")
        filt = pq.operations[0]
        self.assertEqual(filt.condition.op, ">=")

    def test_columns_collected_from_where(self):
        pq = parse_sql("SELECT a FROM t WHERE b > 0")
        col_names = [c.column for c in pq.columns]
        self.assertIn("b", col_names)


class TestParseSqlGroupBy(unittest.TestCase):
    """Tests for the GROUP BY clause parsing."""

    def test_group_by_single_column(self):
        pq = parse_sql("SELECT a, SUM(b) FROM t GROUP BY a")
        ops = pq.operations
        # SelectOp is always last; GroupByOp before it
        self.assertIsInstance(ops[-1], SelectOp)
        gb = next(op for op in ops if isinstance(op, GroupByOp))
        self.assertEqual(gb.columns, ["a"])

    def test_group_by_multiple_columns(self):
        pq = parse_sql("SELECT a, b, SUM(c) FROM t GROUP BY a, b")
        gb = next(op for op in pq.operations if isinstance(op, GroupByOp))
        self.assertEqual(gb.columns, ["a", "b"])

    def test_group_by_with_filter(self):
        pq = parse_sql("SELECT a, SUM(b) FROM t WHERE a > 0 GROUP BY a")
        types = [type(op).__name__ for op in pq.operations]
        self.assertIn("FilterOp", types)
        self.assertIn("GroupByOp", types)
        self.assertEqual(types[-1], "SelectOp")
        # FilterOp must appear before GroupByOp
        self.assertLess(types.index("FilterOp"), types.index("GroupByOp"))


class TestParseSqlJoin(unittest.TestCase):
    """Tests for the JOIN clause parsing."""

    def test_inner_join(self):
        pq = parse_sql("SELECT a.x, b.y FROM a INNER JOIN b ON a.id = b.id")
        join = next(op for op in pq.operations if isinstance(op, JoinOp))
        self.assertEqual(join.right_table, "b")
        self.assertEqual(join.left_key, "id")
        self.assertEqual(join.right_key, "id")
        self.assertEqual(join.join_type, "inner")

    def test_join_without_inner_keyword(self):
        pq = parse_sql("SELECT a, b FROM t1 JOIN t2 ON t1.k = t2.k")
        join = next(op for op in pq.operations if isinstance(op, JoinOp))
        self.assertIsNotNone(join)
        self.assertEqual(join.join_type, "inner")

    def test_left_join(self):
        pq = parse_sql("SELECT a FROM t1 LEFT JOIN t2 ON t1.k = t2.k")
        join = next(op for op in pq.operations if isinstance(op, JoinOp))
        self.assertEqual(join.join_type, "left")

    def test_join_column_collected(self):
        pq = parse_sql("SELECT x FROM t1 JOIN t2 ON t1.k = t2.k")
        col_names = [c.column for c in pq.columns]
        self.assertIn("k", col_names)

    # ------------------------------------------------------------------
    # Multi-column (AND) join parsing
    # ------------------------------------------------------------------

    def test_multi_column_join_two_keys(self):
        """ON clause with two AND-chained key pairs."""
        pq = parse_sql(
            "SELECT a.x, b.y FROM a JOIN b ON a.company_id = b.cid AND a.dept_id = b.did"
        )
        join = next(op for op in pq.operations if isinstance(op, JoinOp))
        self.assertEqual(join.left_keys, ["company_id", "dept_id"])
        self.assertEqual(join.right_keys, ["cid", "did"])

    def test_multi_column_join_three_keys(self):
        """ON clause with three AND-chained key pairs."""
        pq = parse_sql("SELECT a.x FROM a JOIN b ON a.k1 = b.k1 AND a.k2 = b.k2 AND a.k3 = b.k3")
        join = next(op for op in pq.operations if isinstance(op, JoinOp))
        self.assertEqual(join.left_keys, ["k1", "k2", "k3"])
        self.assertEqual(join.right_keys, ["k1", "k2", "k3"])

    def test_multi_column_join_single_key_lists(self):
        """Single-key ON clause: left_keys/right_keys are one-element lists."""
        pq = parse_sql("SELECT x FROM t1 JOIN t2 ON t1.id = t2.id")
        join = next(op for op in pq.operations if isinstance(op, JoinOp))
        self.assertEqual(join.left_keys, ["id"])
        self.assertEqual(join.right_keys, ["id"])
        # Backward-compat properties still work.
        self.assertEqual(join.left_key, "id")
        self.assertEqual(join.right_key, "id")

    def test_multi_column_join_columns_collected(self):
        """All key columns from a multi-column ON appear in pq.columns."""
        pq = parse_sql(
            "SELECT x FROM t1 JOIN t2 ON t1.company_id = t2.cid AND t1.dept_id = t2.did"
        )
        col_names = [c.column for c in pq.columns]
        for col in ("company_id", "dept_id", "cid", "did"):
            self.assertIn(col, col_names)


class TestParsedQueryColumns(unittest.TestCase):
    """Tests for the column collection logic."""

    def test_columns_from_select(self):
        pq = parse_sql("SELECT a, b FROM t")
        col_names = [c.column for c in pq.columns]
        self.assertIn("a", col_names)
        self.assertIn("b", col_names)

    def test_columns_deduped(self):
        pq = parse_sql("SELECT a, a FROM t")
        col_names = [c.column for c in pq.columns]
        self.assertEqual(col_names.count("a"), 1)

    def test_columns_from_expression(self):
        pq = parse_sql("SELECT a + b AS total FROM t")
        col_names = [c.column for c in pq.columns]
        self.assertIn("a", col_names)
        self.assertIn("b", col_names)

    def test_columns_are_column_refs(self):
        pq = parse_sql("SELECT a, b FROM t")
        for col in pq.columns:
            self.assertIsInstance(col, ColumnRef)

    def test_columns_dtype_zero_from_sql_parser(self):
        # SQL string parser does not set dtype; ColumnRef.dtype should be 0
        pq = parse_sql("SELECT a, b FROM t")
        for col in pq.columns:
            self.assertEqual(col.dtype, 0)


class TestParseSqlSubquery(unittest.TestCase):
    """Tests for subquery (``FROM (SELECT …)``) parsing."""

    def test_subquery_sets_subquery_field(self):
        pq = parse_sql("SELECT a FROM (SELECT a FROM t)")
        self.assertIsNotNone(pq.subquery)

    def test_subquery_inner_from_table(self):
        pq = parse_sql("SELECT a FROM (SELECT a FROM t)")
        self.assertEqual(pq.subquery.from_table, "t")

    def test_subquery_alias_stored_in_from_table(self):
        pq = parse_sql("SELECT a FROM (SELECT a FROM t) AS sub")
        self.assertEqual(pq.from_table, "sub")

    def test_subquery_no_alias_empty_from_table(self):
        pq = parse_sql("SELECT a FROM (SELECT a FROM t)")
        self.assertEqual(pq.from_table, "")

    def test_subquery_inner_ops(self):
        pq = parse_sql("SELECT a FROM (SELECT a FROM t WHERE a > 0)")
        inner_ops = [type(op).__name__ for op in pq.subquery.operations]
        self.assertIn("FilterOp", inner_ops)
        self.assertIn("SelectOp", inner_ops)

    def test_subquery_outer_ops(self):
        pq = parse_sql("SELECT a FROM (SELECT a FROM t) WHERE a < 5")
        outer_ops = [type(op).__name__ for op in pq.operations]
        self.assertIn("FilterOp", outer_ops)
        self.assertIn("SelectOp", outer_ops)

    def test_subquery_inner_expression(self):
        pq = parse_sql("SELECT a FROM (SELECT a + 1 AS a FROM t)")
        inner_select = next(
            (op for op in pq.subquery.operations if isinstance(op, SelectOp)), None
        )
        self.assertIsNotNone(inner_select)
        self.assertEqual(inner_select.items[0].alias, "a")
        self.assertIsInstance(inner_select.items[0].expr, BinaryExpr)

    def test_subquery_no_alias_keyword_not_consumed(self):
        """Alias should not be required; keywords must not be consumed as alias."""
        pq = parse_sql("SELECT a FROM (SELECT a FROM t) WHERE a > 0")
        # The WHERE follows without AS alias
        outer_ops = [type(op).__name__ for op in pq.operations]
        self.assertIn("FilterOp", outer_ops)

    def test_plain_query_subquery_is_none(self):
        pq = parse_sql("SELECT a FROM t")
        self.assertIsNone(pq.subquery)


class TestParsedQueryRepr(unittest.TestCase):
    """Tests for ParsedQuery.__repr__."""

    def test_repr_starts_with_parsed_query(self):
        pq = parse_sql("SELECT a FROM t")
        r = repr(pq)
        self.assertIn("ParsedQuery", r)

    def test_repr_each_operation_on_own_line(self):
        pq = parse_sql("SELECT a FROM t WHERE a > 0")
        r = repr(pq)
        lines = r.splitlines()
        # At least one line per operation plus the header and footer
        self.assertGreater(len(lines), 2)

    def test_repr_contains_from_table(self):
        pq = parse_sql("SELECT a FROM t")
        r = repr(pq)
        self.assertIn("from_table='t'", r)

    def test_repr_contains_columns(self):
        pq = parse_sql("SELECT a, b FROM t")
        r = repr(pq)
        self.assertIn("columns=", r)

    def test_repr_with_subquery_contains_subquery(self):
        pq = parse_sql("SELECT a FROM (SELECT a FROM t)")
        r = repr(pq)
        self.assertIn("subquery=", r)


if __name__ == "__main__":
    unittest.main()
