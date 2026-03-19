"""
Unit tests for :func:`yobx.sql.to_onnx` — polars LazyFrame → ONNX conversion.
"""

import unittest
import numpy as np

from yobx.ext_test_case import ExtTestCase
from yobx.reference import ExtendedReferenceEvaluator
from yobx.sql import to_onnx

try:
    import polars as _pl  # noqa: F401

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def _run(onx, feeds):
    """Run *onx* with the reference evaluator and return outputs."""
    ref = ExtendedReferenceEvaluator(onx)
    return ref.run(None, feeds)


@unittest.skipUnless(HAS_POLARS, "polars not installed")
class TestToOnnxFromLazyFrame(ExtTestCase):
    """Tests for :func:`~yobx.sql.to_onnx` with polars LazyFrame inputs."""

    def test_basic_select(self):
        """Schema inference from LazyFrame, simple column pass-through."""
        import polars as pl

        lf = pl.LazyFrame({"a": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32)})
        onx = to_onnx(lf, "SELECT a FROM t")
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (out,) = _run(onx, {"a": a})
        self.assertEqualArray(out, a)

    def test_arithmetic_expression(self):
        """Element-wise addition with schema from LazyFrame."""
        import polars as pl

        lf = pl.LazyFrame(
            {
                "a": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32),
                "b": pl.Series([4.0, 5.0, 6.0], dtype=pl.Float32),
            }
        )
        onx = to_onnx(lf, "SELECT a + b AS total FROM t")
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run(onx, {"a": a, "b": b})
        self.assertEqualArray(total, a + b, atol=1e-6)

    def test_where_filter(self):
        """WHERE clause applied correctly when using LazyFrame schema."""
        import polars as pl

        lf = pl.LazyFrame({"a": pl.Series([1.0, -2.0, 3.0], dtype=pl.Float32)})
        onx = to_onnx(lf, "SELECT a FROM t WHERE a > 0")
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        (out,) = _run(onx, {"a": a})
        self.assertEqualArray(out, np.array([1.0, 3.0], dtype=np.float32))

    def test_combined_filter_and_expression(self):
        """WHERE + arithmetic in SELECT."""
        import polars as pl

        lf = pl.LazyFrame(
            {
                "a": pl.Series([1.0, -2.0, 3.0], dtype=pl.Float32),
                "b": pl.Series([4.0, 5.0, 6.0], dtype=pl.Float32),
            }
        )
        onx = to_onnx(lf, "SELECT a + b AS total FROM t WHERE a > 0")
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = _run(onx, {"a": a, "b": b})
        self.assertEqualArray(total, np.array([5.0, 9.0], dtype=np.float32), atol=1e-6)

    def test_int64_columns(self):
        """Int64 polars dtype maps to int64 numpy dtype."""
        import polars as pl

        lf = pl.LazyFrame({"x": pl.Series([10, 20, 30], dtype=pl.Int64)})
        onx = to_onnx(lf, "SELECT x FROM t")
        x = np.array([10, 20, 30], dtype=np.int64)
        (out,) = _run(onx, {"x": x})
        self.assertEqualArray(out, x)

    def test_dataframe_input(self):
        """polars.DataFrame should be accepted in place of LazyFrame."""
        import polars as pl

        df = pl.DataFrame(
            {
                "a": pl.Series([1.0, 2.0], dtype=pl.Float32),
                "b": pl.Series([3.0, 4.0], dtype=pl.Float32),
            }
        )
        onx = to_onnx(df, "SELECT a + b AS total FROM t")
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        (total,) = _run(onx, {"a": a, "b": b})
        self.assertEqualArray(total, a + b, atol=1e-6)

    def test_returns_model_proto(self):
        """to_onnx must return an onnx.ModelProto."""
        from onnx import ModelProto
        import polars as pl

        lf = pl.LazyFrame({"a": pl.Series([1.0], dtype=pl.Float32)})
        onx = to_onnx(lf, "SELECT a FROM t")
        self.assertIsInstance(onx, ModelProto)

    def test_inputs_named_after_columns(self):
        """ONNX inputs must use the column names from the polars schema."""
        import polars as pl

        lf = pl.LazyFrame(
            {
                "x": pl.Series([1.0], dtype=pl.Float32),
                "y": pl.Series([2.0], dtype=pl.Float32),
            }
        )
        onx = to_onnx(lf, "SELECT x, y FROM t")
        input_names = [inp.name for inp in onx.graph.input]
        self.assertIn("x", input_names)
        self.assertIn("y", input_names)

    def test_result_matches_sql_to_onnx(self):
        """to_onnx result must be numerically identical to sql_to_onnx."""
        import polars as pl
        from yobx.sql import sql_to_onnx

        lf = pl.LazyFrame(
            {
                "a": pl.Series([1.0, -2.0, 3.0], dtype=pl.Float32),
                "b": pl.Series([4.0, 5.0, 6.0], dtype=pl.Float32),
            }
        )
        query = "SELECT a + b AS total FROM t WHERE a > 0"
        onx_polars = to_onnx(lf, query)
        onx_direct = sql_to_onnx(query, {"a": np.float32, "b": np.float32})

        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        feeds = {"a": a, "b": b}
        (out_polars,) = _run(onx_polars, feeds)
        (out_direct,) = _run(onx_direct, feeds)
        self.assertEqualArray(out_polars, out_direct, atol=1e-6)

    def test_custom_functions(self):
        """Custom Python functions should work through to_onnx."""
        import polars as pl

        def double(x):
            return x * np.float32(2)

        lf = pl.LazyFrame({"a": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32)})
        onx = to_onnx(lf, "SELECT double(a) AS r FROM t", custom_functions={"double": double})
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        (r,) = _run(onx, {"a": a})
        self.assertEqualArray(r, double(a), atol=1e-6)

    def test_unsupported_dtype_raises(self):
        """An unsupported polars dtype should raise ValueError."""
        import polars as pl

        lf = pl.LazyFrame({"a": pl.Series([[1, 2], [3, 4]])})
        with self.assertRaises(ValueError):
            to_onnx(lf, "SELECT a FROM t")

    def test_invalid_frame_type_raises(self):
        """Passing a non-polars object should raise TypeError."""
        with self.assertRaises(TypeError):
            to_onnx({"a": np.float32}, "SELECT a FROM t")


@unittest.skipIf(HAS_POLARS, "polars is installed — skipping import-error test")
class TestToOnnxImportError(ExtTestCase):
    """Verify ImportError when polars is not installed."""

    def test_import_error(self):
        with self.assertRaises(ImportError):
            to_onnx(object(), "SELECT a FROM t")


if __name__ == "__main__":
    unittest.main()
