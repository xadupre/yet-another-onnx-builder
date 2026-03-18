"""
Unit tests for yobx.polars.to_onnx.
"""

import unittest
import numpy as np
import onnx
from onnx import TensorProto

try:
    import polars as pl
    import onnxruntime

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


@unittest.skipUnless(HAS_POLARS, "polars not installed")
class TestPolarsToOnnx(unittest.TestCase):
    """Tests for :func:`yobx.polars.to_onnx`."""

    def _get_to_onnx(self):
        from yobx.polars import to_onnx

        return to_onnx

    def test_basic_dataframe(self):
        """to_onnx on a simple DataFrame produces a valid ModelProto."""
        to_onnx = self._get_to_onnx()
        df = pl.DataFrame({"a": [1, 2], "b": [1.0, 2.0]})
        onx = to_onnx(df)
        self.assertIsInstance(onx, onnx.ModelProto)
        onnx.checker.check_model(onx)

    def test_schema_input(self):
        """to_onnx also accepts a polars.Schema directly."""
        to_onnx = self._get_to_onnx()
        schema = pl.Schema({"x": pl.Float32, "y": pl.Int64})
        onx = to_onnx(schema)
        self.assertIsInstance(onx, onnx.ModelProto)
        onnx.checker.check_model(onx)

    def test_dict_input(self):
        """to_onnx accepts a plain dict mapping name → dtype instance."""
        to_onnx = self._get_to_onnx()
        schema = {"p": pl.Float32(), "q": pl.Boolean()}
        onx = to_onnx(schema)
        onnx.checker.check_model(onx)

    def test_dict_input_dtype_class(self):
        """to_onnx accepts a plain dict mapping name → dtype class."""
        to_onnx = self._get_to_onnx()
        schema = {"p": pl.Float32, "q": pl.Boolean}
        onx = to_onnx(schema)
        onnx.checker.check_model(onx)

    def test_graph_inputs_match_columns(self):
        """Graph inputs correspond 1-to-1 with DataFrame columns."""
        to_onnx = self._get_to_onnx()
        df = pl.DataFrame({"col_a": [1], "col_b": [2.0], "col_c": ["x"]})
        onx = to_onnx(df)
        input_names = [inp.name for inp in onx.graph.input]
        self.assertEqual(input_names, ["col_a", "col_b", "col_c"])

    def test_elem_types(self):
        """Each column dtype maps to the correct ONNX element type."""
        to_onnx = self._get_to_onnx()
        df = pl.DataFrame(
            {
                "i8": pl.Series([1], dtype=pl.Int8),
                "i16": pl.Series([1], dtype=pl.Int16),
                "i32": pl.Series([1], dtype=pl.Int32),
                "i64": pl.Series([1], dtype=pl.Int64),
                "u8": pl.Series([1], dtype=pl.UInt8),
                "u16": pl.Series([1], dtype=pl.UInt16),
                "u32": pl.Series([1], dtype=pl.UInt32),
                "u64": pl.Series([1], dtype=pl.UInt64),
                "f32": pl.Series([1.0], dtype=pl.Float32),
                "f64": pl.Series([1.0], dtype=pl.Float64),
                "b": pl.Series([True], dtype=pl.Boolean),
                "s": pl.Series(["x"], dtype=pl.String),
            }
        )
        onx = to_onnx(df)
        onnx.checker.check_model(onx)
        expected = {
            "i8": TensorProto.INT8,
            "i16": TensorProto.INT16,
            "i32": TensorProto.INT32,
            "i64": TensorProto.INT64,
            "u8": TensorProto.UINT8,
            "u16": TensorProto.UINT16,
            "u32": TensorProto.UINT32,
            "u64": TensorProto.UINT64,
            "f32": TensorProto.FLOAT,
            "f64": TensorProto.DOUBLE,
            "b": TensorProto.BOOL,
            "s": TensorProto.STRING,
        }
        for inp in onx.graph.input:
            self.assertEqual(inp.type.tensor_type.elem_type, expected[inp.name], inp.name)

    def test_identity_nodes(self):
        """The graph contains one Identity node per column."""
        to_onnx = self._get_to_onnx()
        df = pl.DataFrame({"a": [1], "b": [2]})
        onx = to_onnx(df)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertEqual(op_types, ["Identity", "Identity"])

    def test_dynamic_batch_dim(self):
        """Default batch_dim='N' yields a symbolic first dimension."""
        to_onnx = self._get_to_onnx()
        df = pl.DataFrame({"x": [1.0]})
        onx = to_onnx(df, batch_dim="N")
        inp = onx.graph.input[0]
        shape = inp.type.tensor_type.shape
        self.assertEqual(shape.dim[0].dim_param, "N")

    def test_fixed_batch_dim(self):
        """Integer batch_dim creates a fixed-size first dimension."""
        to_onnx = self._get_to_onnx()
        df = pl.DataFrame({"x": [1.0]})
        onx = to_onnx(df, batch_dim=8)
        inp = onx.graph.input[0]
        shape = inp.type.tensor_type.shape
        self.assertEqual(shape.dim[0].dim_value, 8)

    def test_no_batch_dim(self):
        """batch_dim=None produces inputs with no shape information."""
        to_onnx = self._get_to_onnx()
        df = pl.DataFrame({"x": [1.0]})
        onx = to_onnx(df, batch_dim=None)
        inp = onx.graph.input[0]
        self.assertFalse(inp.type.tensor_type.HasField("shape"))

    def test_empty_schema_raises(self):
        """An empty DataFrame/schema raises ValueError."""
        to_onnx = self._get_to_onnx()
        with self.assertRaises(ValueError):
            to_onnx(pl.DataFrame())

    def test_invalid_input_type_raises(self):
        """Passing an unsupported type raises TypeError."""
        to_onnx = self._get_to_onnx()
        with self.assertRaises(TypeError):
            to_onnx([1, 2, 3])

    def test_target_opset(self):
        """to_onnx respects the target_opset argument as int."""
        to_onnx = self._get_to_onnx()
        df = pl.DataFrame({"x": [1.0]})
        onx = to_onnx(df, target_opset=17)
        opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertEqual(opsets[""], 17)

    def test_target_opset_dict(self):
        """to_onnx respects the target_opset argument as a dictionary."""
        to_onnx = self._get_to_onnx()
        df = pl.DataFrame({"x": [1.0]})
        onx = to_onnx(df, target_opset={"": 17})
        opsets = {op.domain: op.version for op in onx.opset_import}
        self.assertEqual(opsets[""], 17)

    def test_target_opset_invalid_raises(self):
        """Passing an invalid target_opset type raises TypeError."""
        to_onnx = self._get_to_onnx()
        df = pl.DataFrame({"x": [1.0]})
        with self.assertRaises(TypeError):
            to_onnx(df, target_opset="17")

    def test_date_dtype(self):
        """Date columns map to INT32."""
        to_onnx = self._get_to_onnx()
        schema = pl.Schema({"d": pl.Date})
        onx = to_onnx(schema)
        onnx.checker.check_model(onx)
        inp = onx.graph.input[0]
        self.assertEqual(inp.type.tensor_type.elem_type, TensorProto.INT32)

    def test_datetime_dtype(self):
        """Datetime columns map to INT64."""
        to_onnx = self._get_to_onnx()
        schema = pl.Schema({"ts": pl.Datetime("us")})
        onx = to_onnx(schema)
        onnx.checker.check_model(onx)
        inp = onx.graph.input[0]
        self.assertEqual(inp.type.tensor_type.elem_type, TensorProto.INT64)

    def test_ort_runs_float_model(self):
        """The produced float model can be executed with onnxruntime."""
        to_onnx = self._get_to_onnx()
        df = pl.DataFrame({"score": pl.Series([0.1, 0.9], dtype=pl.Float32)})
        onx = to_onnx(df, batch_dim=2)
        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        feeds = {"score": np.array([0.1, 0.9], dtype=np.float32)}
        (result,) = sess.run(None, feeds)
        np.testing.assert_array_equal(result, feeds["score"])

    def test_lazy_frame_input(self):
        """to_onnx accepts a polars.LazyFrame and extracts its schema."""
        to_onnx = self._get_to_onnx()
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
        lf = df.lazy()
        onx = to_onnx(lf)
        self.assertIsInstance(onx, onnx.ModelProto)
        onnx.checker.check_model(onx)
        input_names = [inp.name for inp in onx.graph.input]
        self.assertEqual(input_names, ["a", "b"])

    def test_row_filter_adds_mask_input(self):
        """row_filter=True adds a boolean 'mask' input to the graph."""
        to_onnx = self._get_to_onnx()
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        onx = to_onnx(df, row_filter=True)
        onnx.checker.check_model(onx)
        input_names = [inp.name for inp in onx.graph.input]
        self.assertIn("mask", input_names)
        # mask should be the first input
        self.assertEqual(input_names[0], "mask")
        mask_input = onx.graph.input[0]
        self.assertEqual(mask_input.type.tensor_type.elem_type, TensorProto.BOOL)

    def test_row_filter_uses_compress_nodes(self):
        """row_filter=True produces Compress nodes instead of Identity."""
        to_onnx = self._get_to_onnx()
        df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        onx = to_onnx(df, row_filter=True)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertEqual(op_types.count("Compress"), 2)
        self.assertNotIn("Identity", op_types)

    def test_row_filter_ort_execution(self):
        """row_filter=True model filters rows correctly at runtime."""
        to_onnx = self._get_to_onnx()
        df = pl.DataFrame({"age": pl.Series([25, 30, 35], dtype=pl.Int64)})
        onx = to_onnx(df, row_filter=True)
        sess = onnxruntime.InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        mask = np.array([True, False, True])
        age = np.array([25, 30, 35], dtype=np.int64)
        (result,) = sess.run(None, {"mask": mask, "age": age})
        np.testing.assert_array_equal(result, np.array([25, 35], dtype=np.int64))

    def test_row_filter_output_has_dynamic_dim(self):
        """row_filter=True outputs carry a dynamic 'K' first dimension."""
        to_onnx = self._get_to_onnx()
        df = pl.DataFrame({"x": [1.0]})
        onx = to_onnx(df, row_filter=True, batch_dim="N")
        out = onx.graph.output[0]
        shape = out.type.tensor_type.shape
        self.assertEqual(shape.dim[0].dim_param, "K")


@unittest.skipUnless(HAS_POLARS, "polars not installed")
class TestPolarsHelpers(unittest.TestCase):
    """Tests for helper functions in yobx.polars.convert."""

    def test_polars_dtype_to_onnx_element_type(self):
        from yobx.polars.convert import polars_dtype_to_onnx_element_type

        self.assertEqual(polars_dtype_to_onnx_element_type(pl.Float32()), TensorProto.FLOAT)
        self.assertEqual(polars_dtype_to_onnx_element_type(pl.Int64()), TensorProto.INT64)
        self.assertEqual(polars_dtype_to_onnx_element_type(pl.String()), TensorProto.STRING)
        self.assertEqual(polars_dtype_to_onnx_element_type(pl.Boolean()), TensorProto.BOOL)

    def test_polars_dtype_unsupported_raises(self):
        from yobx.polars.convert import polars_dtype_to_onnx_element_type

        class _FakeDtype:
            pass

        with self.assertRaises(TypeError):
            polars_dtype_to_onnx_element_type(_FakeDtype())

    def test_schema_to_numpy_dtypes(self):
        from yobx.polars.convert import schema_to_numpy_dtypes

        schema = pl.Schema({"x": pl.Float32, "y": pl.Int64})
        dtypes = schema_to_numpy_dtypes(schema)
        self.assertEqual(dtypes["x"], np.dtype("float32"))
        self.assertEqual(dtypes["y"], np.dtype("int64"))

    def test_schema_to_numpy_dtypes_from_dataframe(self):
        from yobx.polars.convert import schema_to_numpy_dtypes

        df = pl.DataFrame({"a": pl.Series([1.0], dtype=pl.Float64)})
        dtypes = schema_to_numpy_dtypes(df)
        self.assertEqual(dtypes["a"], np.dtype("float64"))

    def test_schema_to_numpy_dtypes_from_lazyframe(self):
        from yobx.polars.convert import schema_to_numpy_dtypes

        lf = pl.DataFrame({"x": pl.Series([1.0], dtype=pl.Float32)}).lazy()
        dtypes = schema_to_numpy_dtypes(lf)
        self.assertEqual(dtypes["x"], np.dtype("float32"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
