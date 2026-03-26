"""Tests for pivot_table tracing layer (TracedDataFrame API and PivotTableOp)."""

import unittest

import numpy as np

from yobx.ext_test_case import ExtTestCase
from yobx.sql import PivotTableOp
from yobx.xtracing.parse import ColumnRef


# ---------------------------------------------------------------------------
# TracedDataFrame.pivot_table — API / tracing tests
# ---------------------------------------------------------------------------


class TestPivotTableTracing(ExtTestCase):
    """Unit tests for the tracing layer (no ONNX execution)."""

    def _make_df(self, *cols):
        from yobx.xtracing.dataframe_trace import TracedDataFrame, TracedSeries

        columns = {ColumnRef(c): TracedSeries(ColumnRef(c)) for c in cols}
        return TracedDataFrame(columns, source_columns=list(cols))

    def test_returns_traced_dataframe(self):
        from yobx.xtracing.dataframe_trace import TracedDataFrame

        df = self._make_df("k", "cat", "v")
        result = df.pivot_table(
            values="v", index="k", columns="cat", column_values=["X", "Y"]
        )
        self.assertIsInstance(result, TracedDataFrame)

    def test_output_columns_single_index(self):
        df = self._make_df("k", "cat", "v")
        result = df.pivot_table(
            values="v", index="k", columns="cat", column_values=["X", "Y"]
        )
        # Output columns: index "k" + "v_X", "v_Y"
        self.assertEqual(result.columns, ["k", "v_X", "v_Y"])

    def test_output_columns_multi_index(self):
        df = self._make_df("k1", "k2", "cat", "v")
        result = df.pivot_table(
            values="v", index=["k1", "k2"], columns="cat", column_values=["A"]
        )
        self.assertEqual(result.columns, ["k1", "k2", "v_A"])

    def test_output_columns_multi_values(self):
        """Multiple values columns produce separate output columns per (val, cv) pair."""
        df = self._make_df("k", "cat", "v1", "v2")
        result = df.pivot_table(
            values=["v1", "v2"], index="k", columns="cat", column_values=["X", "Y"]
        )
        # Expected: k, v1_X, v1_Y, v2_X, v2_Y
        self.assertEqual(result.columns, ["k", "v1_X", "v1_Y", "v2_X", "v2_Y"])

    def test_output_columns_multi_category_columns(self):
        """Multi-column category uses tuple column_values → <val>_<cv1>_<cv2> names."""
        df = self._make_df("k", "cat1", "cat2", "v")
        result = df.pivot_table(
            values="v", index="k", columns=["cat1", "cat2"],
            column_values=[("A", "X"), ("B", "Y")]
        )
        self.assertEqual(result.columns, ["k", "v_A_X", "v_B_Y"])

    def test_ops_recorded(self):
        df = self._make_df("k", "cat", "v")
        result = df.pivot_table(
            values="v", index="k", columns="cat", column_values=["X"]
        )
        pivot_ops = [op for op in result._ops if isinstance(op, PivotTableOp)]
        self.assertEqual(len(pivot_ops), 1)
        op = pivot_ops[0]
        self.assertEqual(op.index, ["k"])
        self.assertEqual(op.columns, "cat")
        self.assertEqual(op.values, "v")
        self.assertEqual(op.column_values, ["X"])

    def test_ops_recorded_multi_values(self):
        """PivotTableOp stores multiple values columns as a list."""
        df = self._make_df("k", "cat", "v1", "v2")
        result = df.pivot_table(
            values=["v1", "v2"], index="k", columns="cat", column_values=["X"]
        )
        pivot_ops = [op for op in result._ops if isinstance(op, PivotTableOp)]
        self.assertEqual(len(pivot_ops), 1)
        op = pivot_ops[0]
        self.assertEqual(op.values, ["v1", "v2"])

    def test_ops_recorded_multi_columns(self):
        """PivotTableOp stores multiple category columns as a list."""
        df = self._make_df("k", "cat1", "cat2", "v")
        result = df.pivot_table(
            values="v", index="k", columns=["cat1", "cat2"],
            column_values=[("A", "X")]
        )
        pivot_ops = [op for op in result._ops if isinstance(op, PivotTableOp)]
        self.assertEqual(len(pivot_ops), 1)
        op = pivot_ops[0]
        self.assertEqual(op.columns, ["cat1", "cat2"])

    def test_aggfunc_stored(self):
        df = self._make_df("k", "cat", "v")
        for agg in ("sum", "mean", "min", "max", "count"):
            result = df.pivot_table(
                values="v", index="k", columns="cat", aggfunc=agg, column_values=["X"]
            )
            ops = [op for op in result._ops if isinstance(op, PivotTableOp)]
            self.assertEqual(ops[0].aggfunc, agg)

    def test_fill_value_stored(self):
        df = self._make_df("k", "cat", "v")
        result = df.pivot_table(
            values="v", index="k", columns="cat", fill_value=-1.0, column_values=["X"]
        )
        ops = [op for op in result._ops if isinstance(op, PivotTableOp)]
        self.assertAlmostEqual(ops[0].fill_value, -1.0)

    def test_missing_column_values_raises(self):
        df = self._make_df("k", "cat", "v")
        with self.assertRaises(ValueError):
            df.pivot_table(values="v", index="k", columns="cat")  # no column_values

    def test_unknown_index_col_raises(self):
        df = self._make_df("k", "cat", "v")
        with self.assertRaises(KeyError):
            df.pivot_table(
                values="v", index="missing", columns="cat", column_values=["X"]
            )

    def test_no_select_op_added(self):
        """to_parsed_query() must NOT append a fallback SelectOp after PivotTableOp."""
        from yobx.xtracing.parse import SelectOp

        df = self._make_df("k", "cat", "v")
        result = df.pivot_table(
            values="v", index="k", columns="cat", column_values=["X"]
        )
        pq = result.to_parsed_query()
        select_ops = [op for op in pq.operations if isinstance(op, SelectOp)]
        self.assertEqual(len(select_ops), 0, "No SelectOp should be added after PivotTableOp")

    def test_parsed_query_columns_include_input_cols(self):
        """The ParsedQuery.columns list must include the input (not output) columns."""
        from yobx.xtracing.dataframe_trace import TracedDataFrame, TracedSeries

        k_ref = ColumnRef("k", dtype=7)  # int64
        cat_ref = ColumnRef("cat", dtype=8)  # string
        v_ref = ColumnRef("v", dtype=1)  # float32
        columns = {k_ref: TracedSeries(k_ref), cat_ref: TracedSeries(cat_ref), v_ref: TracedSeries(v_ref)}
        df = TracedDataFrame(columns, source_columns=["k", "cat", "v"])
        result = df.pivot_table(
            values="v", index="k", columns="cat", column_values=["X"]
        )
        pq = result.to_parsed_query()
        col_names = [c.column for c in pq.columns]
        self.assertIn("k", col_names)
        self.assertIn("cat", col_names)
        self.assertIn("v", col_names)
        # Output columns (v_X) should NOT be in input list
        self.assertNotIn("v_X", col_names)


# ---------------------------------------------------------------------------
# Exported API test
# ---------------------------------------------------------------------------


class TestPivotTableExport(ExtTestCase):
    """Verify that PivotTableOp is importable from yobx.sql."""

    def test_importable(self):
        from yobx.sql import PivotTableOp as PT  # noqa: F401

        self.assertIs(PT, PivotTableOp)

    def test_pivot_table_op_fields(self):
        """Basic PivotTableOp construction."""
        k_ref = ColumnRef("k", dtype=7)
        cat_ref = ColumnRef("cat", dtype=8)
        v_ref = ColumnRef("v", dtype=1)
        op = PivotTableOp(
            index_refs=[k_ref],
            columns_refs=[cat_ref],
            values_refs=[v_ref],
            aggfunc="sum",
            column_values=["X", "Y"],
            fill_value=0.0,
        )
        self.assertEqual(op.index, ["k"])
        self.assertEqual(op.columns, "cat")
        self.assertEqual(op.values, "v")
        self.assertEqual(op.column_values, ["X", "Y"])

    def test_pivot_table_op_multi_fields(self):
        """PivotTableOp with multiple values and columns refs."""
        k_ref = ColumnRef("k", dtype=7)
        cat1_ref = ColumnRef("cat1", dtype=8)
        cat2_ref = ColumnRef("cat2", dtype=8)
        v1_ref = ColumnRef("v1", dtype=1)
        v2_ref = ColumnRef("v2", dtype=1)
        op = PivotTableOp(
            index_refs=[k_ref],
            columns_refs=[cat1_ref, cat2_ref],
            values_refs=[v1_ref, v2_ref],
            aggfunc="sum",
            column_values=[("A", "X"), ("B", "Y")],
            fill_value=0.0,
        )
        self.assertEqual(op.columns, ["cat1", "cat2"])
        self.assertEqual(op.values, ["v1", "v2"])


if __name__ == "__main__":
    unittest.main()
