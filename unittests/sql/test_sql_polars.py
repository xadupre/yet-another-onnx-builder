"""Unit tests for :mod:`yobx.sql.polars_convert` — polars LazyFrame to ONNX."""

import unittest

import numpy as np

from yobx.ext_test_case import ExtTestCase, has_onnxruntime


def _has_polars() -> bool:
    try:
        import polars  # noqa: F401

        return True
    except ImportError:
        return False


def _ort_run(onx, feeds):
    from onnxruntime import InferenceSession
    from yobx.container import ExportArtifact

    proto = onx.proto if isinstance(onx, ExportArtifact) else onx
    sess = InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


@unittest.skipUnless(_has_polars(), "polars not installed")
class TestPolarsDtypeMapping(ExtTestCase):
    """Tests for :func:`~yobx.sql.polars_convert._polars_dtype_to_numpy`."""

    def _import(self):
        from yobx.sql.polars_convert import _polars_dtype_to_numpy

        return _polars_dtype_to_numpy

    def test_float32(self):
        import polars as pl

        fn = self._import()
        self.assertEqual(fn(pl.Float32), np.dtype("float32"))

    def test_float64(self):
        import polars as pl

        fn = self._import()
        self.assertEqual(fn(pl.Float64), np.dtype("float64"))

    def test_int32(self):
        import polars as pl

        fn = self._import()
        self.assertEqual(fn(pl.Int32), np.dtype("int32"))

    def test_int64(self):
        import polars as pl

        fn = self._import()
        self.assertEqual(fn(pl.Int64), np.dtype("int64"))

    def test_bool(self):
        import polars as pl

        fn = self._import()
        self.assertEqual(fn(pl.Boolean), np.dtype("bool"))

    def test_string(self):
        import polars as pl

        fn = self._import()
        self.assertEqual(fn(pl.String), np.dtype("object"))

    def test_unsupported_raises(self):
        import polars as pl

        fn = self._import()
        with self.assertRaises(ValueError):
            fn(pl.List(pl.Float32))


@unittest.skipUnless(_has_polars(), "polars not installed")
class TestPolarsExprToSql(ExtTestCase):
    """Tests for :func:`~yobx.sql.polars_convert._polars_expr_to_sql`."""

    def _fn(self):
        from yobx.sql.polars_convert import _polars_expr_to_sql

        return _polars_expr_to_sql

    def test_bare_column(self):
        self.assertEqual(self._fn()('col("a")'), "a")

    def test_column_alias(self):
        self.assertEqual(self._fn()('col("a").alias("x")'), "a AS x")

    def test_arithmetic_add(self):
        self.assertEqual(self._fn()('(col("a")) + (col("b"))'), "(a) + (b)")

    def test_arithmetic_sub(self):
        self.assertEqual(self._fn()('(col("a")) - (col("b"))'), "(a) - (b)")

    def test_arithmetic_mul(self):
        self.assertEqual(self._fn()('(col("a")) * (col("b"))'), "(a) * (b)")

    def test_arithmetic_div(self):
        self.assertEqual(self._fn()('(col("a")) / (col("b"))'), "(a) / (b)")

    def test_comparison_gt(self):
        self.assertEqual(self._fn()('(col("a")) > (0.0)'), "a > 0.0")

    def test_comparison_lt(self):
        self.assertEqual(self._fn()('(col("a")) < (1.0)'), "a < 1.0")

    def test_comparison_ge(self):
        self.assertEqual(self._fn()('(col("a")) >= (0.0)'), "a >= 0.0")

    def test_comparison_le(self):
        self.assertEqual(self._fn()('(col("a")) <= (5.0)'), "a <= 5.0")

    def test_expr_alias(self):
        self.assertEqual(
            self._fn()('[(col("a")) + (col("b"))].alias("total")'), "(a) + (b) AS total"
        )

    def test_agg_sum(self):
        self.assertEqual(self._fn()('col("v").sum()'), "SUM(v)")

    def test_agg_mean(self):
        self.assertEqual(self._fn()('col("v").mean()'), "AVG(v)")

    def test_agg_min(self):
        self.assertEqual(self._fn()('col("v").min()'), "MIN(v)")

    def test_agg_max(self):
        self.assertEqual(self._fn()('col("v").max()'), "MAX(v)")

    def test_agg_count(self):
        self.assertEqual(self._fn()('col("v").count()'), "COUNT(v)")

    def test_agg_sum_alias(self):
        self.assertEqual(self._fn()('col("v").sum().alias("total")'), "SUM(v) AS total")

    def test_boolean_and(self):
        expr = '([(col("a")) > (0.0)]) & ([(col("b")) < (6.0)])'
        result = self._fn()(expr)
        self.assertIn("AND", result)
        self.assertIn("a > 0.0", result)
        self.assertIn("b < 6.0", result)

    def test_numeric_literal(self):
        self.assertEqual(self._fn()("1.5"), "1.5")

    def test_integer_literal(self):
        self.assertEqual(self._fn()("42"), "42")


@unittest.skipUnless(_has_polars(), "polars not installed")
class TestLazyframeToOnnx(ExtTestCase):
    """Tests for :func:`~yobx.sql.lazyframe_to_onnx`."""

    def _run(self, lf, input_dtypes, feeds):
        """Convert *lf*, run with reference evaluator and (when available) ORT."""
        from yobx.sql import lazyframe_to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        artifact = lazyframe_to_onnx(lf, input_dtypes)
        ref = ExtendedReferenceEvaluator(artifact)
        ref_outputs = ref.run(None, feeds)
        if has_onnxruntime():
            ort_outputs = _ort_run(artifact, feeds)
            self.assertEqual(len(ref_outputs), len(ort_outputs))
            for ro, oo in zip(ref_outputs, ort_outputs):
                np.testing.assert_allclose(oo, ro, rtol=1e-5, atol=1e-6)
        return ref_outputs

    # ------------------------------------------------------------------
    # Column pass-through
    # ------------------------------------------------------------------

    def test_select_single_column_alias(self):
        import polars as pl

        lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0]})
        lf = lf.select([pl.col("a").alias("out_a")])
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        (out,) = self._run(lf, {"a": np.float64}, {"a": a})
        self.assertEqualArray(out, a)

    def test_filter_passthrough(self):
        """FILTER without SELECT passes through all source columns."""
        import polars as pl

        lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        lf = lf.filter(pl.col("a") > 1.0)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        out_a, out_b = self._run(lf, {"a": np.float64, "b": np.float64}, {"a": a, "b": b})
        self.assertEqualArray(out_a, np.array([2.0, 3.0]))
        self.assertEqualArray(out_b, np.array([5.0, 6.0]))

    # ------------------------------------------------------------------
    # Arithmetic expressions
    # ------------------------------------------------------------------

    def test_select_add(self):
        import polars as pl

        lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        lf = lf.select([(pl.col("a") + pl.col("b")).alias("total")])
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        (total,) = self._run(lf, {"a": np.float64, "b": np.float64}, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)

    def test_select_sub(self):
        import polars as pl

        lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        lf = lf.select([(pl.col("a") - pl.col("b")).alias("diff")])
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        (diff,) = self._run(lf, {"a": np.float64, "b": np.float64}, {"a": a, "b": b})
        np.testing.assert_allclose(diff, a - b)

    def test_select_mul(self):
        import polars as pl

        lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        lf = lf.select([(pl.col("a") * pl.col("b")).alias("product")])
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        (product,) = self._run(lf, {"a": np.float64, "b": np.float64}, {"a": a, "b": b})
        np.testing.assert_allclose(product, a * b)

    def test_select_div(self):
        import polars as pl

        lf = pl.LazyFrame({"a": [2.0, 4.0, 6.0], "b": [1.0, 2.0, 3.0]})
        lf = lf.select([(pl.col("a") / pl.col("b")).alias("ratio")])
        a = np.array([2.0, 4.0, 6.0], dtype=np.float64)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        (ratio,) = self._run(lf, {"a": np.float64, "b": np.float64}, {"a": a, "b": b})
        np.testing.assert_allclose(ratio, a / b)

    def test_select_multiple_exprs(self):
        import polars as pl

        lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        lf = lf.select(
            [(pl.col("a") + pl.col("b")).alias("s"), (pl.col("a") - pl.col("b")).alias("d")]
        )
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        s, d = self._run(lf, {"a": np.float64, "b": np.float64}, {"a": a, "b": b})
        np.testing.assert_allclose(s, a + b)
        np.testing.assert_allclose(d, a - b)

    # ------------------------------------------------------------------
    # Filtering (WHERE clause)
    # ------------------------------------------------------------------

    def test_filter_gt(self):
        import polars as pl

        lf = pl.LazyFrame({"a": [1.0, -2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        lf = lf.filter(pl.col("a") > 0).select([(pl.col("a") + pl.col("b")).alias("total")])
        a = np.array([1.0, -2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        (total,) = self._run(lf, {"a": np.float64, "b": np.float64}, {"a": a, "b": b})
        np.testing.assert_allclose(total, np.array([5.0, 9.0]))

    def test_filter_and(self):
        """Compound AND filter."""
        import polars as pl

        lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        lf = lf.filter((pl.col("a") > 1.0) & (pl.col("b") < 6.0))
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        out_a, out_b = self._run(lf, {"a": np.float64, "b": np.float64}, {"a": a, "b": b})
        np.testing.assert_allclose(out_a, np.array([2.0]))
        np.testing.assert_allclose(out_b, np.array([5.0]))

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------

    def test_agg_sum(self):
        import polars as pl

        lf = pl.LazyFrame({"v": [1.0, 2.0, 3.0, 4.0]})
        lf = lf.select([pl.col("v").sum().alias("total")])
        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        (total,) = self._run(lf, {"v": np.float64}, {"v": v})
        np.testing.assert_allclose(total, np.array([10.0]))

    def test_agg_mean(self):
        import polars as pl

        lf = pl.LazyFrame({"v": [1.0, 2.0, 3.0, 4.0]})
        lf = lf.select([pl.col("v").mean().alias("avg")])
        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        (avg,) = self._run(lf, {"v": np.float64}, {"v": v})
        np.testing.assert_allclose(avg, np.array([2.5]))

    def test_agg_min(self):
        import polars as pl

        lf = pl.LazyFrame({"v": [3.0, 1.0, 2.0]})
        lf = lf.select([pl.col("v").min().alias("mn")])
        v = np.array([3.0, 1.0, 2.0], dtype=np.float64)
        (mn,) = self._run(lf, {"v": np.float64}, {"v": v})
        np.testing.assert_allclose(mn, np.array([1.0]))

    def test_agg_max(self):
        import polars as pl

        lf = pl.LazyFrame({"v": [3.0, 1.0, 2.0]})
        lf = lf.select([pl.col("v").max().alias("mx")])
        v = np.array([3.0, 1.0, 2.0], dtype=np.float64)
        (mx,) = self._run(lf, {"v": np.float64}, {"v": v})
        np.testing.assert_allclose(mx, np.array([3.0]))

    # ------------------------------------------------------------------
    # Float32 inputs
    # ------------------------------------------------------------------

    def test_float32_select_add(self):
        import polars as pl

        lf = pl.LazyFrame(
            {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]},
            schema={"a": pl.Float32, "b": pl.Float32},
        )
        lf = lf.select([(pl.col("a") + pl.col("b")).alias("total")])
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = self._run(lf, {"a": np.float32, "b": np.float32}, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)

    # ------------------------------------------------------------------
    # Public API export test
    # ------------------------------------------------------------------

    def test_imported_from_sql_package(self):
        from yobx.sql import lazyframe_to_onnx  # noqa: F401

        self.assertTrue(callable(lazyframe_to_onnx))

    def test_to_onnx_with_lazyframe(self):
        """The unified to_onnx() entry point accepts a polars LazyFrame."""
        import polars as pl
        from yobx.sql.convert import to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        lf = lf.select([(pl.col("a") + pl.col("b")).alias("total")])
        dtypes = {"a": np.float64, "b": np.float64}
        artifact = to_onnx(lf, dtypes)
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)

    def test_to_onnx_imported_from_sql_package(self):
        from yobx.sql import to_onnx  # noqa: F401

        self.assertTrue(callable(to_onnx))


class TestToOnnxSqlString(ExtTestCase):
    """Tests for the unified :func:`~yobx.sql.convert.to_onnx` with a SQL string."""

    def test_to_onnx_with_sql_string(self):
        """to_onnx() also accepts a plain SQL query string."""
        from yobx.sql.convert import to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        dtypes = {"a": np.float64, "b": np.float64}
        artifact = to_onnx("SELECT a + b AS total FROM t", dtypes)
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, a + b)

    def test_to_onnx_with_custom_functions(self):
        """to_onnx() forwards custom_functions to sql_to_onnx."""
        from yobx.sql.convert import to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        dtypes = {"a": np.float64}
        artifact = to_onnx(
            "SELECT my_sqrt(a) AS r FROM t", dtypes, custom_functions={"my_sqrt": np.sqrt}
        )
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 4.0, 9.0], dtype=np.float64)
        (r,) = ref.run(None, {"a": a})
        np.testing.assert_allclose(r, np.sqrt(a))

    def test_to_onnx_with_filter(self):
        """to_onnx() forwards the WHERE clause correctly."""
        from yobx.sql.convert import to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        dtypes = {"a": np.float64, "b": np.float64}
        artifact = to_onnx("SELECT a + b AS total FROM t WHERE a > 0", dtypes)
        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        (total,) = ref.run(None, {"a": a, "b": b})
        np.testing.assert_allclose(total, np.array([5.0, 9.0]))


if __name__ == "__main__":
    unittest.main()
