from typing import Dict
import numpy as np
import onnx


def polars_dtype_to_onnx_element_type(dtype) -> int:
    """
    Maps a :class:`polars.DataType` to the corresponding
    ``onnx.TensorProto`` element-type integer.

    :param dtype: a Polars data type (e.g. ``polars.Float32``,
        ``polars.Int64``, ``polars.String``).
    :return: ONNX element-type integer constant (e.g.
        ``onnx.TensorProto.FLOAT``).
    :raises TypeError: when the Polars dtype has no ONNX equivalent.
    """
    import polars as pl

    # Accept both dtype *instances* (e.g. pl.Float32()) and dtype *classes*
    # (e.g. pl.Float32).  Normalise to an instance for isinstance checks.
    if isinstance(dtype, type) and issubclass(dtype, pl.DataType):
        try:  # noqa: SIM105
            dtype = dtype()
        except TypeError:
            # Parametric types that require arguments (e.g. pl.Datetime)
            # fall through to the isinstance checks below.
            pass

    # --- simple scalar mappings ---
    if isinstance(dtype, pl.Int8):
        return onnx.TensorProto.INT8
    if isinstance(dtype, pl.Int16):
        return onnx.TensorProto.INT16
    if isinstance(dtype, pl.Int32):
        return onnx.TensorProto.INT32
    if isinstance(dtype, pl.Int64):
        return onnx.TensorProto.INT64
    if isinstance(dtype, pl.UInt8):
        return onnx.TensorProto.UINT8
    if isinstance(dtype, pl.UInt16):
        return onnx.TensorProto.UINT16
    if isinstance(dtype, pl.UInt32):
        return onnx.TensorProto.UINT32
    if isinstance(dtype, pl.UInt64):
        return onnx.TensorProto.UINT64
    if isinstance(dtype, pl.Float32):
        return onnx.TensorProto.FLOAT
    if isinstance(dtype, pl.Float64):
        return onnx.TensorProto.DOUBLE
    if isinstance(dtype, pl.Boolean):
        return onnx.TensorProto.BOOL
    # String / Utf8 are aliases in modern Polars
    if isinstance(dtype, (pl.String, pl.Utf8)):
        return onnx.TensorProto.STRING
    # Date → int32 (days since Unix epoch)
    if isinstance(dtype, pl.Date):
        return onnx.TensorProto.INT32
    # Datetime / Duration / Time → int64
    if isinstance(dtype, (pl.Datetime, pl.Duration, pl.Time)):
        return onnx.TensorProto.INT64
    # Categorical / Enum → STRING
    if isinstance(dtype, (pl.Categorical, pl.Enum)):
        return onnx.TensorProto.STRING

    # Also handle uninstantiated parametric classes (e.g. pl.Datetime itself)
    for pl_cls, onnx_type in [
        (pl.Datetime, onnx.TensorProto.INT64),
        (pl.Duration, onnx.TensorProto.INT64),
    ]:
        if dtype is pl_cls or (isinstance(dtype, type) and issubclass(dtype, pl_cls)):
            return onnx_type

    raise TypeError(
        f"Polars dtype {dtype!r} (type {type(dtype).__name__!r}) "
        "has no supported ONNX onnx.TensorProto mapping."
    )


def schema_to_numpy_dtypes(schema) -> Dict[str, np.dtype]:
    """
    Returns a mapping from column name to :class:`numpy.dtype` for each
    column in the given :class:`polars.Schema`.

    :param schema: a :class:`polars.Schema`, :class:`polars.DataFrame`, or
        :class:`polars.LazyFrame`.
    :return: dict mapping ``{column_name: numpy_dtype}``.

    Example::

        import polars as pl
        from yobx.polars.convert import schema_to_numpy_dtypes

        schema = pl.Schema({"x": pl.Float32, "y": pl.Int64})
        dtypes = schema_to_numpy_dtypes(schema)
        # {"x": dtype("float32"), "y": dtype("int64")}
    """
    import polars as pl
    from onnx.helper import tensor_dtype_to_np_dtype

    if isinstance(schema, pl.DataFrame):
        schema = schema.schema
    elif isinstance(schema, pl.LazyFrame):
        schema = schema.collect_schema()

    result = {}
    for col_name, col_dtype in schema.items():
        elem_type = polars_dtype_to_onnx_element_type(col_dtype)
        result[col_name] = tensor_dtype_to_np_dtype(elem_type)
    return result
