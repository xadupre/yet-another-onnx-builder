"""Tests for pivot_table support in the DataFrame tracer and ONNX converter."""

import unittest

import numpy as np

from yobx.ext_test_case import ExtTestCase, has_onnxruntime
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sql import PivotTableOp, dataframe_to_onnx
from yobx.xtracing.parse import ColumnRef


def _run(func, dtypes, feeds):
    """Trace *func*, convert to ONNX and run with both the reference evaluator
    (and, when available, OnnxRuntime) returning the reference-evaluator outputs.
    """
    from yobx.container import ExportArtifact

    artifact = dataframe_to_onnx(func, dtypes)
    ref = ExtendedReferenceEvaluator(artifact)
    ref_outputs = ref.run(None, feeds)
    if has_onnxruntime():
        from onnxruntime import InferenceSession

        proto = artifact.proto if isinstance(artifact, ExportArtifact) else artifact
        sess = InferenceSession(
            proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_outputs = sess.run(None, feeds)
        assert len(ref_outputs) == len(ort_outputs)
        for ro, oo in zip(ref_outputs, ort_outputs):
            np.testing.assert_allclose(oo, ro, rtol=1e-5, atol=1e-6)
    return ref_outputs


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
# ONNX execution tests — SUM
# ---------------------------------------------------------------------------


class TestPivotTableSum(ExtTestCase):
    """End-to-end tests for pivot_table(aggfunc='sum')."""

    def _feeds(self):
        return {
            "k": np.array([1, 1, 2, 2, 1], dtype=np.int64),
            "cat": np.array(["X", "Y", "X", "Y", "X"], dtype=object),
            "v": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        }

    def _dtypes(self):
        return {"k": np.int64, "cat": object, "v": np.float32}

    def test_sum_basic(self):
        def transform(df):
            return df.pivot_table(
                values="v", index="k", columns="cat", aggfunc="sum",
                column_values=["X", "Y"]
            )

        feeds = self._feeds()
        k_out, vX, vY = _run(transform, self._dtypes(), feeds)
        order = np.argsort(k_out)
        # k=1: X=1+5=6, Y=2;  k=2: X=3, Y=4
        np.testing.assert_array_equal(k_out[order], [1, 2])
        np.testing.assert_allclose(vX[order], [6.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(vY[order], [2.0, 4.0], atol=1e-5)

    def test_sum_fill_value(self):
        """fill_value is used for (index, column) combinations with no rows."""
        def transform(df):
            return df.pivot_table(
                values="v", index="k", columns="cat", aggfunc="sum",
                fill_value=-99.0, column_values=["X", "Z"]
            )

        feeds = {
            "k": np.array([1, 2, 2], dtype=np.int64),
            "cat": np.array(["X", "X", "Z"], dtype=object),
            "v": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        }
        k_out, vX, vZ = _run(transform, self._dtypes(), feeds)
        order = np.argsort(k_out)
        # k=1: X=10, Z=-99 (no Z rows);  k=2: X=20, Z=30
        np.testing.assert_allclose(vX[order], [10.0, 20.0], atol=1e-5)
        np.testing.assert_allclose(vZ[order], [-99.0, 30.0], atol=1e-5)

    def test_sum_int_column_values(self):
        """Pivot on integer column values (not strings)."""
        def transform(df):
            return df.pivot_table(
                values="v", index="k", columns="cat", aggfunc="sum",
                column_values=[10, 20]
            )

        feeds = {
            "k": np.array([1, 1, 2, 2], dtype=np.int64),
            "cat": np.array([10, 20, 10, 20], dtype=np.int64),
            "v": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        }
        dtypes = {"k": np.int64, "cat": np.int64, "v": np.float32}
        k_out, v10, v20 = _run(transform, dtypes, feeds)
        order = np.argsort(k_out)
        np.testing.assert_allclose(v10[order], [1.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(v20[order], [2.0, 4.0], atol=1e-5)

    def test_sum_with_filter(self):
        """filter() applied before pivot_table()."""
        def transform(df):
            df2 = df.filter(df["v"] > 2.0)
            return df2.pivot_table(
                values="v", index="k", columns="cat", aggfunc="sum",
                column_values=["X"]
            )

        feeds = {
            "k": np.array([1, 1, 2], dtype=np.int64),
            "cat": np.array(["X", "X", "X"], dtype=object),
            "v": np.array([1.0, 3.0, 5.0], dtype=np.float32),
        }
        k_out, vX = _run(transform, self._dtypes(), feeds)
        order = np.argsort(k_out)
        # After filter (v>2): rows 1 (k=1,v=3) and 2 (k=2,v=5)
        np.testing.assert_allclose(vX[order], [3.0, 5.0], atol=1e-5)


# ---------------------------------------------------------------------------
# ONNX execution tests — MEAN, MIN, MAX, COUNT
# ---------------------------------------------------------------------------


class TestPivotTableAggFuncs(ExtTestCase):
    """Tests for mean / min / max / count aggregation functions."""

    def _feeds(self):
        return {
            "k": np.array([1, 1, 2, 2, 1], dtype=np.int64),
            "cat": np.array(["X", "Y", "X", "Y", "X"], dtype=object),
            "v": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        }

    def _dtypes(self):
        return {"k": np.int64, "cat": object, "v": np.float32}

    def test_mean(self):
        def transform(df):
            return df.pivot_table(
                values="v", index="k", columns="cat", aggfunc="mean",
                column_values=["X", "Y"]
            )

        k_out, vX, vY = _run(transform, self._dtypes(), self._feeds())
        order = np.argsort(k_out)
        # k=1: X=mean(1,5)=3, Y=mean(2)=2;  k=2: X=mean(3)=3, Y=mean(4)=4
        np.testing.assert_allclose(vX[order], [3.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(vY[order], [2.0, 4.0], atol=1e-5)

    def test_mean_fill_value(self):
        """Groups with no matching rows receive fill_value for mean."""
        def transform(df):
            return df.pivot_table(
                values="v", index="k", columns="cat", aggfunc="mean",
                fill_value=-1.0, column_values=["X", "Z"]
            )

        feeds = {
            "k": np.array([1, 2], dtype=np.int64),
            "cat": np.array(["X", "X"], dtype=object),
            "v": np.array([4.0, 8.0], dtype=np.float32),
        }
        k_out, vX, vZ = _run(transform, self._dtypes(), feeds)
        order = np.argsort(k_out)
        np.testing.assert_allclose(vX[order], [4.0, 8.0], atol=1e-5)
        np.testing.assert_allclose(vZ[order], [-1.0, -1.0], atol=1e-5)

    def test_min(self):
        def transform(df):
            return df.pivot_table(
                values="v", index="k", columns="cat", aggfunc="min",
                column_values=["X", "Y"]
            )

        k_out, vX, vY = _run(transform, self._dtypes(), self._feeds())
        order = np.argsort(k_out)
        # k=1: X=min(1,5)=1, Y=min(2)=2;  k=2: X=min(3)=3, Y=min(4)=4
        np.testing.assert_allclose(vX[order], [1.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(vY[order], [2.0, 4.0], atol=1e-5)

    def test_min_fill_value(self):
        """Groups with no matching rows get fill_value for min."""
        def transform(df):
            return df.pivot_table(
                values="v", index="k", columns="cat", aggfunc="min",
                fill_value=-99.0, column_values=["X", "Y"]
            )

        feeds = {
            "k": np.array([1, 2], dtype=np.int64),
            "cat": np.array(["X", "X"], dtype=object),  # no Y rows
            "v": np.array([3.0, 7.0], dtype=np.float32),
        }
        k_out, vX, vY = _run(transform, self._dtypes(), feeds)
        order = np.argsort(k_out)
        np.testing.assert_allclose(vX[order], [3.0, 7.0], atol=1e-5)
        np.testing.assert_allclose(vY[order], [-99.0, -99.0], atol=1e-5)

    def test_max(self):
        def transform(df):
            return df.pivot_table(
                values="v", index="k", columns="cat", aggfunc="max",
                column_values=["X", "Y"]
            )

        k_out, vX, vY = _run(transform, self._dtypes(), self._feeds())
        order = np.argsort(k_out)
        # k=1: X=max(1,5)=5, Y=max(2)=2;  k=2: X=max(3)=3, Y=max(4)=4
        np.testing.assert_allclose(vX[order], [5.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(vY[order], [2.0, 4.0], atol=1e-5)

    def test_count(self):
        def transform(df):
            return df.pivot_table(
                values="v", index="k", columns="cat", aggfunc="count",
                column_values=["X", "Y"]
            )

        k_out, vX, vY = _run(transform, self._dtypes(), self._feeds())
        order = np.argsort(k_out)
        # k=1: X=2, Y=1;  k=2: X=1, Y=1
        np.testing.assert_array_equal(vX[order], [2, 1])
        np.testing.assert_array_equal(vY[order], [1, 1])

    def test_invalid_aggfunc_raises(self):
        def transform(df):
            return df.pivot_table(
                values="v", index="k", columns="cat", aggfunc="median",
                column_values=["X"]
            )

        with self.assertRaises(ValueError):
            _run(transform, {"k": np.int64, "cat": object, "v": np.float32}, {
                "k": np.array([1], dtype=np.int64),
                "cat": np.array(["X"], dtype=object),
                "v": np.array([1.0], dtype=np.float32),
            })


# ---------------------------------------------------------------------------
# ONNX execution tests — multi-column index
# ---------------------------------------------------------------------------


class TestPivotTableMultiIndex(ExtTestCase):
    """Tests for pivot_table with multiple index columns."""

    def test_two_column_index_sum(self):
        def transform(df):
            return df.pivot_table(
                values="v", index=["k1", "k2"], columns="cat", aggfunc="sum",
                column_values=["X", "Y"]
            )

        feeds = {
            "k1": np.array([1, 1, 2, 2, 1], dtype=np.int64),
            "k2": np.array([1, 1, 2, 2, 2], dtype=np.int64),
            "cat": np.array(["X", "Y", "X", "Y", "X"], dtype=object),
            "v": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        }
        dtypes = {"k1": np.int64, "k2": np.int64, "cat": object, "v": np.float32}
        k1_out, k2_out, vX, vY = _run(transform, dtypes, feeds)
        # Unique (k1,k2) combos: (1,1), (1,2), (2,2)
        self.assertEqual(len(k1_out), 3)
        self.assertEqual(len(vX), 3)
        # Verify total sum is preserved
        np.testing.assert_allclose(vX.sum() + vY.sum(), feeds["v"].sum(), atol=1e-5)


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
            columns_ref=cat_ref,
            values_ref=v_ref,
            aggfunc="sum",
            column_values=["X", "Y"],
            fill_value=0.0,
        )
        self.assertEqual(op.index, ["k"])
        self.assertEqual(op.columns, "cat")
        self.assertEqual(op.values, "v")
        self.assertEqual(op.column_values, ["X", "Y"])


if __name__ == "__main__":
    unittest.main()
