"""
Converts a :class:`polars.DataFrame` schema into an ONNX model
(:class:`onnx.ModelProto`).

The resulting model contains one input per DataFrame column, typed
according to the column's Polars dtype, and passes every input straight
through to the corresponding output (identity graph).  The primary use-case
is to capture the *schema contract* of a DataFrame as a portable,
runtime-agnostic ONNX artifact.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from onnx import ModelProto, TensorProto
import onnx.helper as oh

from .. import DEFAULT_TARGET_OPSET


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
        try:
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
    for pl_cls, onnx_type in [
        (pl.Datetime, TensorProto.INT64),
        (pl.Duration, TensorProto.INT64),
    ]:
        if dtype is pl_cls or (isinstance(dtype, type) and issubclass(dtype, pl_cls)):
            return onnx_type

    raise TypeError(
        f"Polars dtype {dtype!r} (type {type(dtype).__name__!r}) "
        "has no supported ONNX TensorProto mapping."
    )


def _schema_to_onnx_inputs_outputs(
    schema,
    batch_dim: Optional[Union[int, str]],
) -> Tuple[list, list, list]:
    """Build ONNX input/output value-info and identity nodes from a schema.

    :param schema: a :class:`polars.Schema` (or any mapping ``{name: dtype}``).
    :param batch_dim: first-axis specification — ``None`` for a fully-static
        1-D input, an ``int`` for a fixed batch size, or a ``str`` for a
        dynamic (symbolic) batch dimension.
    :return: ``(inputs, outputs, nodes)`` ready to be passed to
        :func:`onnx.helper.make_graph`.
    """
    inputs = []
    outputs = []
    nodes = []
    for col_name, col_dtype in schema.items():
        elem_type = polars_dtype_to_onnx_element_type(col_dtype)

        if batch_dim is None:
            shape: Optional[list] = None
        else:
            shape = [batch_dim]

        inp = oh.make_tensor_value_info(col_name, elem_type, shape)
        out_name = f"{col_name}_out"
        out = oh.make_tensor_value_info(out_name, elem_type, shape)
        node = oh.make_node("Identity", inputs=[col_name], outputs=[out_name], name=f"id_{col_name}")
        inputs.append(inp)
        outputs.append(out)
        nodes.append(node)
    return inputs, outputs, nodes


def to_onnx(
    df_or_schema,
    batch_dim: Optional[Union[int, str]] = "N",
    target_opset: int = DEFAULT_TARGET_OPSET,
    name: str = "polars_schema",
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
    :param target_opset: ONNX opset version for the produced model
        (default: :data:`yobx.DEFAULT_TARGET_OPSET`).
    :param name: ONNX graph name embedded in the model proto.
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

    inputs, outputs, nodes = _schema_to_onnx_inputs_outputs(schema, batch_dim)

    graph = oh.make_graph(nodes, name, inputs, outputs)
    model = oh.make_model(
        graph,
        opset_imports=[oh.make_opsetid("", target_opset)],
    )
    return model


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
