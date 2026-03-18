"""
Converts a :class:`polars.DataFrame` schema into an ONNX model
(:class:`onnx.ModelProto`).

The resulting model contains one input per DataFrame column, typed
according to the column's Polars dtype, and passes every input straight
through to the corresponding output (identity graph).  The primary use-case
is to capture the *schema contract* of a DataFrame as a portable,
runtime-agnostic ONNX artifact.
"""

from typing import Callable, Dict, Optional, Union

import numpy as np
from onnx import ModelProto, TensorProto

from .. import DEFAULT_TARGET_OPSET
from ..xbuilder import GraphBuilder


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
        return TensorProto.INT8
    if isinstance(dtype, pl.Int16):
        return TensorProto.INT16
    if isinstance(dtype, pl.Int32):
        return TensorProto.INT32
    if isinstance(dtype, pl.Int64):
        return TensorProto.INT64
    if isinstance(dtype, pl.UInt8):
        return TensorProto.UINT8
    if isinstance(dtype, pl.UInt16):
        return TensorProto.UINT16
    if isinstance(dtype, pl.UInt32):
        return TensorProto.UINT32
    if isinstance(dtype, pl.UInt64):
        return TensorProto.UINT64
    if isinstance(dtype, pl.Float32):
        return TensorProto.FLOAT
    if isinstance(dtype, pl.Float64):
        return TensorProto.DOUBLE
    if isinstance(dtype, pl.Boolean):
        return TensorProto.BOOL
    # String / Utf8 are aliases in modern Polars
    if isinstance(dtype, (pl.String, pl.Utf8)):
        return TensorProto.STRING
    # Date → int32 (days since Unix epoch)
    if isinstance(dtype, pl.Date):
        return TensorProto.INT32
    # Datetime / Duration / Time → int64
    if isinstance(dtype, (pl.Datetime, pl.Duration, pl.Time)):
        return TensorProto.INT64
    # Categorical / Enum → STRING
    if isinstance(dtype, (pl.Categorical, pl.Enum)):
        return TensorProto.STRING

    # Also handle uninstantiated parametric classes (e.g. pl.Datetime itself)
    for pl_cls, onnx_type in [(pl.Datetime, TensorProto.INT64), (pl.Duration, TensorProto.INT64)]:
        if dtype is pl_cls or (isinstance(dtype, type) and issubclass(dtype, pl_cls)):
            return onnx_type

    raise TypeError(
        f"Polars dtype {dtype!r} (type {type(dtype).__name__!r}) "
        "has no supported ONNX TensorProto mapping."
    )


def to_onnx(
    df_or_schema,
    batch_dim: Optional[Union[int, str]] = "N",
    target_opset: Union[int, Dict[str, int]] = DEFAULT_TARGET_OPSET,
    builder_cls: Union[type, Callable] = GraphBuilder,
) -> ModelProto:
    """
    Converts a :class:`polars.DataFrame` (or its schema) into an ONNX
    identity model that encodes the schema as typed inputs/outputs.

    Each column in the DataFrame maps to one ONNX graph input and one
    output connected by an ``Identity`` node.  The primary purpose is to
    capture the *schema contract* of a DataFrame as a portable ONNX model.

    :param df_or_schema: a :class:`polars.DataFrame` or
        :class:`polars.Schema` whose column names and dtypes define the
        ONNX inputs.
    :param batch_dim: controls the first (batch) axis of every input tensor.
        Pass a :class:`str` (e.g. ``"N"`` or ``"batch"``) for a symbolic
        dynamic dimension (default), an :class:`int` for a fixed size, or
        ``None`` to omit shape information entirely.
    :param target_opset: ONNX opset version for the produced model.
        Either an integer for the default domain (``""``), or a dictionary
        mapping domain names to opset versions,
        e.g. ``{"": 20}``.
        Default: :data:`yobx.DEFAULT_TARGET_OPSET`.
    :param builder_cls: by default the graph builder is a
        :class:`yobx.xbuilder.GraphBuilder` but any builder can
        be used as long it implements the apis :ref:`builder-api`
        and :ref:`builder-api-make`.
    :return: an :class:`onnx.ModelProto` with one ``Identity`` node per
        column.

    Example::

        import polars as pl
        from yobx.polars import to_onnx

        df = pl.DataFrame({"age": [25, 30], "score": [0.8, 0.9]})
        onx = to_onnx(df)
    """
    import polars as pl

    if isinstance(df_or_schema, pl.DataFrame):
        schema = df_or_schema.schema
    elif isinstance(df_or_schema, pl.Schema):
        schema = df_or_schema
    elif isinstance(df_or_schema, dict):
        schema = df_or_schema
    else:
        raise TypeError(
            f"df_or_schema must be a polars.DataFrame or polars.Schema, "
            f"got {type(df_or_schema)!r}."
        )

    if not schema:
        raise ValueError("The DataFrame/Schema has no columns.")

    if isinstance(target_opset, int):
        dict_target_opset: Dict[str, int] = {"": target_opset}
    else:
        if not isinstance(target_opset, dict):
            raise TypeError(
                f"target_opset must be a dictionary or an integer not {target_opset!r}"
            )
        dict_target_opset = target_opset.copy()
        if "" not in dict_target_opset:
            dict_target_opset[""] = DEFAULT_TARGET_OPSET

    g = builder_cls(dict_target_opset)

    shape = (batch_dim,) if batch_dim is not None else None
    for col_name, col_dtype in schema.items():
        elem_type = polars_dtype_to_onnx_element_type(col_dtype)
        out_name = f"{col_name}_out"
        g.make_tensor_input(col_name, elem_type, shape)
        g.op.Identity(col_name, outputs=[out_name], name=f"id_{col_name}")
        g.make_tensor_output(out_name, elem_type=elem_type, shape=shape, indexed=False)

    if isinstance(g, GraphBuilder):
        onx, _ = g.to_onnx(return_optimize_report=True)
    else:
        onx = g.to_onnx()
    return onx  # type: ignore


def schema_to_numpy_dtypes(schema) -> Dict[str, np.dtype]:
    """
    Returns a mapping from column name to :class:`numpy.dtype` for each
    column in the given :class:`polars.Schema`.

    :param schema: a :class:`polars.Schema` (or a polars DataFrame).
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

    result = {}
    for col_name, col_dtype in schema.items():
        elem_type = polars_dtype_to_onnx_element_type(col_dtype)
        result[col_name] = tensor_dtype_to_np_dtype(elem_type)
    return result
