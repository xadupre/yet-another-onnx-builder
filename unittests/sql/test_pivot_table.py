"""ONNX execution tests for pivot_table support (DataFrame tracer → ONNX)."""

import unittest

import numpy as np

from yobx.ext_test_case import ExtTestCase, has_onnxruntime
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sql import dataframe_to_onnx


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
# ONNX execution tests — multiple values columns
# ---------------------------------------------------------------------------


class TestPivotTableMultiValues(ExtTestCase):
    """Tests for pivot_table with multiple values columns."""

    def test_two_values_columns_sum(self):
        """Two independent values columns are each pivoted and output separately."""
        def transform(df):
            return df.pivot_table(
                values=["v1", "v2"], index="k", columns="cat", aggfunc="sum",
                column_values=["X", "Y"]
            )

        feeds = {
            "k": np.array([1, 1, 2, 2], dtype=np.int64),
            "cat": np.array(["X", "Y", "X", "Y"], dtype=object),
            "v1": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "v2": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        }
        dtypes = {"k": np.int64, "cat": object, "v1": np.float32, "v2": np.float32}
        # Output order: k, v1_X, v1_Y, v2_X, v2_Y
        k_out, v1X, v1Y, v2X, v2Y = _run(transform, dtypes, feeds)
        order = np.argsort(k_out)
        # k=1: v1_X=1, v1_Y=2, v2_X=10, v2_Y=20
        # k=2: v1_X=3, v1_Y=4, v2_X=30, v2_Y=40
        np.testing.assert_allclose(v1X[order], [1.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(v1Y[order], [2.0, 4.0], atol=1e-5)
        np.testing.assert_allclose(v2X[order], [10.0, 30.0], atol=1e-5)
        np.testing.assert_allclose(v2Y[order], [20.0, 40.0], atol=1e-5)

    def test_two_values_columns_mean(self):
        """mean aggfunc works independently for each values column."""
        def transform(df):
            return df.pivot_table(
                values=["v1", "v2"], index="k", columns="cat", aggfunc="mean",
                column_values=["X"]
            )

        feeds = {
            "k": np.array([1, 1, 2], dtype=np.int64),
            "cat": np.array(["X", "X", "X"], dtype=object),
            "v1": np.array([2.0, 4.0, 6.0], dtype=np.float32),
            "v2": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        }
        dtypes = {"k": np.int64, "cat": object, "v1": np.float32, "v2": np.float32}
        k_out, v1X, v2X = _run(transform, dtypes, feeds)
        order = np.argsort(k_out)
        # k=1: v1_X=mean(2,4)=3, v2_X=mean(10,20)=15  k=2: v1_X=6, v2_X=30
        np.testing.assert_allclose(v1X[order], [3.0, 6.0], atol=1e-5)
        np.testing.assert_allclose(v2X[order], [15.0, 30.0], atol=1e-5)


# ---------------------------------------------------------------------------
# ONNX execution tests — multiple category columns
# ---------------------------------------------------------------------------


class TestPivotTableMultiColumns(ExtTestCase):
    """Tests for pivot_table with multiple category columns."""

    def test_two_category_columns_sum(self):
        """Compound category key (cat1, cat2) filters rows by AND of equalities."""
        def transform(df):
            return df.pivot_table(
                values="v", index="k", columns=["cat1", "cat2"], aggfunc="sum",
                column_values=[("A", "X"), ("A", "Y"), ("B", "X")]
            )

        feeds = {
            "k": np.array([1, 1, 1, 2, 2], dtype=np.int64),
            "cat1": np.array(["A", "A", "B", "A", "B"], dtype=object),
            "cat2": np.array(["X", "Y", "X", "X", "X"], dtype=object),
            "v": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        }
        dtypes = {"k": np.int64, "cat1": object, "cat2": object, "v": np.float32}
        # Outputs: k, v_A_X, v_A_Y, v_B_X
        k_out, vAX, vAY, vBX = _run(transform, dtypes, feeds)
        order = np.argsort(k_out)
        # k=1: A_X=1, A_Y=2, B_X=3;  k=2: A_X=4, A_Y=0, B_X=5
        np.testing.assert_allclose(vAX[order], [1.0, 4.0], atol=1e-5)
        np.testing.assert_allclose(vAY[order], [2.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(vBX[order], [3.0, 5.0], atol=1e-5)

    def test_two_category_columns_fill_value(self):
        """fill_value applied to (index, compound-category) combinations with no rows."""
        def transform(df):
            return df.pivot_table(
                values="v", index="k", columns=["cat1", "cat2"], aggfunc="sum",
                fill_value=-1.0, column_values=[("A", "X"), ("B", "Y")]
            )

        feeds = {
            "k": np.array([1, 2], dtype=np.int64),
            "cat1": np.array(["A", "A"], dtype=object),
            "cat2": np.array(["X", "X"], dtype=object),
            "v": np.array([7.0, 8.0], dtype=np.float32),
        }
        dtypes = {"k": np.int64, "cat1": object, "cat2": object, "v": np.float32}
        k_out, vAX, vBY = _run(transform, dtypes, feeds)
        order = np.argsort(k_out)
        np.testing.assert_allclose(vAX[order], [7.0, 8.0], atol=1e-5)
        # No rows match (B, Y) for either group → fill_value
        np.testing.assert_allclose(vBY[order], [-1.0, -1.0], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
