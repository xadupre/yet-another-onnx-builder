"""
Unit tests for the new parameters added to all SQL to_onnx functions:
  - large_model / external_threshold (sql_to_onnx, parsed_query_to_onnx,
    lazyframe_to_onnx, dataframe_to_onnx, trace_numpy_to_onnx, to_onnx)
  - dynamic_shapes / input_names (trace_numpy_to_onnx, to_onnx)
  - return_optimize_report (all entry points)
"""

import unittest
import numpy as np

from yobx.ext_test_case import ExtTestCase
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sql import dataframe_to_onnx, sql_to_onnx, to_onnx
from yobx.sql.convert import trace_numpy_to_onnx
from yobx.sql.sql_convert import parsed_query_to_onnx
from yobx.xtracing import trace_dataframe

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_transform(df):
    return df.select([(df["a"] + df["b"]).alias("total")])


def _simple_numpy(X):
    return X + np.float32(1)


# ---------------------------------------------------------------------------
# sql_to_onnx — large_model / external_threshold
# ---------------------------------------------------------------------------


class TestSqlToOnnxLargeModel(ExtTestCase):

    def test_large_model_false_container_is_none(self):
        dtypes = {"a": np.float32}
        art = sql_to_onnx("SELECT a FROM t", dtypes, large_model=False)
        self.assertIsNone(art.container)

    def test_large_model_true_sets_container(self):
        from yobx.container import ExtendedModelContainer

        dtypes = {"a": np.float32}
        art = sql_to_onnx("SELECT a FROM t", dtypes, large_model=True)
        self.assertIsNotNone(art.container)
        self.assertIsInstance(art.container, ExtendedModelContainer)

    def test_large_model_result_is_still_runnable(self):
        dtypes = {"a": np.float32, "b": np.float32}
        art = sql_to_onnx("SELECT a + b AS total FROM t", dtypes, large_model=True)
        # When large_model=True, art.proto is None; use get_proto() instead.
        ref = ExtendedReferenceEvaluator(art.get_proto())
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        (out,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(out, a + b, rtol=1e-6)

    def test_external_threshold_accepted(self):
        # Just verify no error is raised with a custom threshold.
        dtypes = {"a": np.float32}
        art = sql_to_onnx("SELECT a FROM t", dtypes, large_model=True, external_threshold=512)
        self.assertIsNotNone(art.container)


# ---------------------------------------------------------------------------
# parsed_query_to_onnx — large_model / external_threshold
# ---------------------------------------------------------------------------


class TestParsedQueryToOnnxLargeModel(ExtTestCase):

    def _pq(self):
        return trace_dataframe(_simple_transform, {"a": np.float32, "b": np.float32})

    def test_large_model_false_container_is_none(self):
        art = parsed_query_to_onnx(self._pq(), large_model=False)
        self.assertIsNone(art.container)

    def test_large_model_true_sets_container(self):
        from yobx.container import ExtendedModelContainer

        art = parsed_query_to_onnx(self._pq(), large_model=True)
        self.assertIsNotNone(art.container)
        self.assertIsInstance(art.container, ExtendedModelContainer)

    def test_external_threshold_accepted(self):
        art = parsed_query_to_onnx(self._pq(), large_model=True, external_threshold=256)
        self.assertIsNotNone(art.container)


# ---------------------------------------------------------------------------
# dataframe_to_onnx — large_model / external_threshold
# ---------------------------------------------------------------------------


class TestDataframeToOnnxLargeModel(ExtTestCase):

    def test_large_model_false_container_is_none(self):
        art = dataframe_to_onnx(
            _simple_transform, {"a": np.float32, "b": np.float32}, large_model=False
        )
        self.assertIsNone(art.container)

    def test_large_model_true_sets_container(self):
        from yobx.container import ExtendedModelContainer

        art = dataframe_to_onnx(
            _simple_transform, {"a": np.float32, "b": np.float32}, large_model=True
        )
        self.assertIsNotNone(art.container)
        self.assertIsInstance(art.container, ExtendedModelContainer)

    def test_external_threshold_accepted(self):
        art = dataframe_to_onnx(
            _simple_transform,
            {"a": np.float32, "b": np.float32},
            large_model=True,
            external_threshold=256,
        )
        self.assertIsNotNone(art.container)


# ---------------------------------------------------------------------------
# trace_numpy_to_onnx — dynamic_shapes, large_model, external_threshold
# ---------------------------------------------------------------------------


class TestTraceNumpyToOnnxNewParams(ExtTestCase):

    def test_dynamic_shapes_overrides_batch_dim(self):
        """Both dims symbolic when dynamic_shapes covers both axes."""
        X = np.random.randn(4, 3).astype(np.float32)
        art = trace_numpy_to_onnx(_simple_numpy, X, dynamic_shapes=({0: "N", 1: "M"},))
        inp = art.proto.graph.input[0]
        dims = [
            d.dim_param if d.dim_param else d.dim_value for d in inp.type.tensor_type.shape.dim
        ]
        self.assertEqual(dims, ["N", "M"])

    def test_dynamic_shapes_partial(self):
        """Only axis 0 is symbolic; axis 1 remains static."""
        X = np.random.randn(4, 3).astype(np.float32)
        art = trace_numpy_to_onnx(_simple_numpy, X, dynamic_shapes=({0: "batch"},))
        inp = art.proto.graph.input[0]
        dims = [
            d.dim_param if d.dim_param else d.dim_value for d in inp.type.tensor_type.shape.dim
        ]
        self.assertEqual(dims[0], "batch")
        self.assertEqual(dims[1], 3)

    def test_dynamic_shapes_length_mismatch_raises(self):
        X = np.random.randn(4, 3).astype(np.float32)
        with self.assertRaises(ValueError):
            trace_numpy_to_onnx(_simple_numpy, X, dynamic_shapes=({0: "N"}, {1: "M"}))

    def test_dynamic_shapes_out_of_bounds_raises(self):
        X = np.random.randn(4, 3).astype(np.float32)
        with self.assertRaises(ValueError):
            trace_numpy_to_onnx(_simple_numpy, X, dynamic_shapes=({5: "N"},))

    def test_large_model_false_container_is_none(self):
        X = np.random.randn(4, 3).astype(np.float32)
        art = trace_numpy_to_onnx(_simple_numpy, X, large_model=False)
        self.assertIsNone(art.container)

    def test_large_model_true_sets_container(self):
        from yobx.container import ExtendedModelContainer

        X = np.random.randn(4, 3).astype(np.float32)
        art = trace_numpy_to_onnx(_simple_numpy, X, large_model=True)
        self.assertIsNotNone(art.container)
        self.assertIsInstance(art.container, ExtendedModelContainer)

    def test_external_threshold_accepted(self):
        X = np.random.randn(4, 3).astype(np.float32)
        art = trace_numpy_to_onnx(_simple_numpy, X, large_model=True, external_threshold=512)
        self.assertIsNotNone(art.container)

    def test_result_is_still_correct(self):
        X = np.random.randn(4, 3).astype(np.float32)
        art = trace_numpy_to_onnx(
            _simple_numpy, X, input_names=["myX"], dynamic_shapes=({0: "batch"},)
        )
        ref = ExtendedReferenceEvaluator(art)
        (out,) = ref.run(None, {"myX": X})
        np.testing.assert_allclose(out, _simple_numpy(X), rtol=1e-6)


# ---------------------------------------------------------------------------
# to_onnx (unified entry point) — input_names, dynamic_shapes,
#   large_model, external_threshold
# ---------------------------------------------------------------------------


class TestToOnnxNewParams(ExtTestCase):

    # ------------------------------------------------------------------
    # SQL string dispatch
    # ------------------------------------------------------------------

    def test_sql_large_model_true(self):
        from yobx.container import ExtendedModelContainer

        dtypes = {"a": np.float32}
        art = to_onnx("SELECT a FROM t", dtypes, large_model=True)
        self.assertIsNotNone(art.container)
        self.assertIsInstance(art.container, ExtendedModelContainer)

    def test_sql_large_model_false(self):
        dtypes = {"a": np.float32}
        art = to_onnx("SELECT a FROM t", dtypes, large_model=False)
        self.assertIsNone(art.container)

    def test_sql_external_threshold(self):
        dtypes = {"a": np.float32}
        art = to_onnx("SELECT a FROM t", dtypes, large_model=True, external_threshold=256)
        self.assertIsNotNone(art.container)

    # ------------------------------------------------------------------
    # DataFrame-tracing callable dispatch
    # ------------------------------------------------------------------

    def test_callable_large_model_true(self):
        from yobx.container import ExtendedModelContainer

        dtypes = {"a": np.float32, "b": np.float32}
        art = to_onnx(_simple_transform, dtypes, large_model=True)
        self.assertIsNotNone(art.container)
        self.assertIsInstance(art.container, ExtendedModelContainer)

    def test_callable_large_model_false(self):
        dtypes = {"a": np.float32, "b": np.float32}
        art = to_onnx(_simple_transform, dtypes, large_model=False)
        self.assertIsNone(art.container)

    # ------------------------------------------------------------------
    # numpy-function callable dispatch
    # ------------------------------------------------------------------

    def test_numpy_input_names_forwarded(self):
        X = np.random.randn(4, 3).astype(np.float32)
        art = to_onnx(_simple_numpy, (X,), input_names=["feat"])
        self.assertEqual(art.proto.graph.input[0].name, "feat")

    def test_numpy_dynamic_shapes_forwarded(self):
        X = np.random.randn(4, 3).astype(np.float32)
        art = to_onnx(_simple_numpy, (X,), dynamic_shapes=({0: "N", 1: "M"},))
        inp = art.proto.graph.input[0]
        dims = [
            d.dim_param if d.dim_param else d.dim_value for d in inp.type.tensor_type.shape.dim
        ]
        self.assertEqual(dims, ["N", "M"])

    def test_numpy_large_model_forwarded(self):
        from yobx.container import ExtendedModelContainer

        X = np.random.randn(4, 3).astype(np.float32)
        art = to_onnx(_simple_numpy, (X,), large_model=True)
        self.assertIsNotNone(art.container)
        self.assertIsInstance(art.container, ExtendedModelContainer)

    def test_numpy_external_threshold_forwarded(self):
        X = np.random.randn(4, 3).astype(np.float32)
        art = to_onnx(_simple_numpy, (X,), large_model=True, external_threshold=128)
        self.assertIsNotNone(art.container)

    def test_numpy_result_is_correct_with_custom_input_name(self):
        X = np.random.randn(4, 3).astype(np.float32)
        art = to_onnx(
            _simple_numpy, (X,), input_names=["myinput"], dynamic_shapes=({0: "batch"},)
        )
        ref = ExtendedReferenceEvaluator(art)
        (out,) = ref.run(None, {"myinput": X})
        np.testing.assert_allclose(out, _simple_numpy(X), rtol=1e-6)


# ---------------------------------------------------------------------------
# return_optimize_report — sql_to_onnx
# ---------------------------------------------------------------------------


class TestSqlToOnnxReturnOptimizeReport(ExtTestCase):

    def test_default_report_is_none(self):
        """Report is None by default (return_optimize_report=False)."""
        art = sql_to_onnx("SELECT a + b AS total FROM t", {"a": np.float32, "b": np.float32})
        self.assertIsNone(art.report)

    def test_report_populated_when_true(self):
        """Report is populated when return_optimize_report=True."""
        from yobx.container import ExportReport

        art = sql_to_onnx(
            "SELECT a + b AS total FROM t",
            {"a": np.float32, "b": np.float32},
            return_optimize_report=True,
        )
        self.assertIsNotNone(art.report)
        self.assertIsInstance(art.report, ExportReport)
        self.assertIsInstance(art.report.stats, list)
        self.assertGreater(len(art.report.stats), 0)


# ---------------------------------------------------------------------------
# return_optimize_report — parsed_query_to_onnx
# ---------------------------------------------------------------------------


class TestParsedQueryToOnnxReturnOptimizeReport(ExtTestCase):

    def _pq(self):
        return trace_dataframe(_simple_transform, {"a": np.float32, "b": np.float32})

    def test_default_report_is_none(self):
        art = parsed_query_to_onnx(self._pq())
        self.assertIsNone(art.report)

    def test_report_populated_when_true(self):
        from yobx.container import ExportReport

        art = parsed_query_to_onnx(self._pq(), return_optimize_report=True)
        self.assertIsNotNone(art.report)
        self.assertIsInstance(art.report, ExportReport)
        self.assertIsInstance(art.report.stats, list)
        self.assertGreater(len(art.report.stats), 0)


# ---------------------------------------------------------------------------
# return_optimize_report — dataframe_to_onnx
# ---------------------------------------------------------------------------


class TestDataframeToOnnxReturnOptimizeReport(ExtTestCase):

    def test_default_report_is_none(self):
        art = dataframe_to_onnx(_simple_transform, {"a": np.float32, "b": np.float32})
        self.assertIsNone(art.report)

    def test_report_populated_when_true(self):
        from yobx.container import ExportReport

        art = dataframe_to_onnx(
            _simple_transform, {"a": np.float32, "b": np.float32}, return_optimize_report=True
        )
        self.assertIsNotNone(art.report)
        self.assertIsInstance(art.report, ExportReport)
        self.assertIsInstance(art.report.stats, list)
        self.assertGreater(len(art.report.stats), 0)


# ---------------------------------------------------------------------------
# return_optimize_report — trace_numpy_to_onnx
# ---------------------------------------------------------------------------


class TestTraceNumpyToOnnxReturnOptimizeReport(ExtTestCase):

    def test_default_report_is_none(self):
        X = np.random.randn(4, 3).astype(np.float32)
        art = trace_numpy_to_onnx(_simple_numpy, X)
        self.assertIsNone(art.report)

    def test_report_populated_when_true(self):
        from yobx.container import ExportReport

        X = np.random.randn(4, 3).astype(np.float32)
        art = trace_numpy_to_onnx(_simple_numpy, X, return_optimize_report=True)
        self.assertIsNotNone(art.report)
        self.assertIsInstance(art.report, ExportReport)
        self.assertIsInstance(art.report.stats, list)
        self.assertGreater(len(art.report.stats), 0)


# ---------------------------------------------------------------------------
# return_optimize_report — to_onnx (unified entry point)
# ---------------------------------------------------------------------------


class TestToOnnxReturnOptimizeReport(ExtTestCase):

    def test_sql_default_report_is_none(self):
        dtypes = {"a": np.float32, "b": np.float32}
        art = to_onnx("SELECT a + b AS total FROM t", dtypes)
        self.assertIsNone(art.report)

    def test_sql_report_populated_when_true(self):
        from yobx.container import ExportReport

        dtypes = {"a": np.float32, "b": np.float32}
        art = to_onnx("SELECT a + b AS total FROM t", dtypes, return_optimize_report=True)
        self.assertIsNotNone(art.report)
        self.assertIsInstance(art.report, ExportReport)
        self.assertGreater(len(art.report.stats), 0)

    def test_callable_default_report_is_none(self):
        dtypes = {"a": np.float32, "b": np.float32}
        art = to_onnx(_simple_transform, dtypes)
        self.assertIsNone(art.report)

    def test_callable_report_populated_when_true(self):
        from yobx.container import ExportReport

        dtypes = {"a": np.float32, "b": np.float32}
        art = to_onnx(_simple_transform, dtypes, return_optimize_report=True)
        self.assertIsNotNone(art.report)
        self.assertIsInstance(art.report, ExportReport)
        self.assertGreater(len(art.report.stats), 0)

    def test_numpy_default_report_is_none(self):
        X = np.random.randn(4, 3).astype(np.float32)
        art = to_onnx(_simple_numpy, (X,))
        self.assertIsNone(art.report)

    def test_numpy_report_populated_when_true(self):
        from yobx.container import ExportReport

        X = np.random.randn(4, 3).astype(np.float32)
        art = to_onnx(_simple_numpy, (X,), return_optimize_report=True)
        self.assertIsNotNone(art.report)
        self.assertIsInstance(art.report, ExportReport)
        self.assertGreater(len(art.report.stats), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
