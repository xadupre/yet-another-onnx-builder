"""
Converts a :class:`polars.DataFrame` schema into an ONNX model
(:class:`onnx.ModelProto`).

The resulting model contains one input per DataFrame column, typed
according to the column's Polars dtype, and passes every input straight
through to the corresponding output (identity graph).  The primary use-case
is to capture the *schema contract* of a DataFrame as a portable,
runtime-agnostic ONNX artifact.
"""

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import onnx
from .. import DEFAULT_TARGET_OPSET
from ..xbuilder import GraphBuilder
from .polars_helper import parse_polars_explained, polars_dtype_to_onnx_element_type


def to_onnx(
    lazy_df: "polars.LazyFrame",
    initial_schema: Union["polars.Schema", List[onnx.TensorValueInfoProto]],
    batch_dim: Optional[Union[int, str]] = "N",
    target_opset: Union[int, Dict[str, int]] = DEFAULT_TARGET_OPSET,
    builder_cls: Union[type, Callable] = GraphBuilder,
    row_filter: bool = False,
) -> onnx.ModelProto:
    """
    Converts a :class:`polars.LazyFrame` (or its schema) into an ONNX
    model that encodes the schema as typed inputs/outputs.

    Each column in the DataFrame maps to one ONNX graph input.  By
    default each input is passed straight through to the corresponding
    output via an ``Identity`` node.  When *row_filter* is ``True`` a
    boolean ``mask`` input is added and each column is filtered through
    an ONNX ``Compress`` node instead, enabling row-level selection at
    inference time.

    :param lazy_df: a :class:`polars.LazyFrame`,
        :class:`polars.LazyFrame`, or :class:`polars.Schema` whose
        column names and dtypes define the ONNX inputs.  A plain
        ``{name: dtype}`` dict is also accepted.
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
    :param row_filter: when ``True``, a boolean input named ``mask`` of
        shape ``(batch_dim,)`` is added to the graph, and each column
        output is produced via ``Compress(col, mask, axis=0)`` instead of
        ``Identity``.  The filtered outputs have a dynamic first dimension
        ``"K"`` (where ``K ≤ N``).  Requires *batch_dim* to be set in
        order to have a meaningful input shape for ``mask``.
    :return: an :class:`onnx.ModelProto` with one node per column.
    """
    import polars as pl

    if not isinstance(lazy_df, pl.LazyFrame):
        raise TypeError(
            f"Only LazyFrame capturing the schema lead to an ONNX model "
            f"but type is {type(lazy_df)}."
        )

    schema = lazy_df.collect_schema()

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

    for col_name, col_dtype in initial_schema.items():
        elem_type = polars_dtype_to_onnx_element_type(col_dtype)
        out_name = f"{col_name}_out"
        g.make_tensor_input(col_name, elem_type, (shape, 1))

    # execution plan

    schema = lazy_df.explain(format="plain")
    operations = parse_polars_explained(schema)

    for op in operations:
        pass

    for col_name, col_dtype in schema.items():
        elem_type = polars_dtype_to_onnx_element_type(col_dtype)
        out_name = f"{col_name}_out"
        g.make_tensor_output(out_name, elem_type=elem_type, shape=shape, indexed=False)

    if isinstance(g, GraphBuilder):
        onx, _ = g.to_onnx(return_optimize_report=True)
    else:
        onx = g.to_onnx()
    return onx  # type: ignore
