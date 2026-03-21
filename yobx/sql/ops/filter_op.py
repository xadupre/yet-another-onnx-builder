"""
ONNX converter for :class:`~yobx.sql.parse.FilterOp` (SQL ``WHERE`` clause).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ...typing import GraphBuilderExtendedProtocol
from ..parse import FilterOp
from .._expr import _ExprEmitter
from .register import register_sql_op_converter


@register_sql_op_converter(FilterOp)
def convert_filter_op(
    g: GraphBuilderExtendedProtocol,
    sts: Optional[Dict],
    outputs: List[str],
    op: FilterOp,
    col_map: Dict[str, str],
    right_col_map: Dict[str, str],
) -> Dict[str, str]:
    """
    Apply a ``WHERE`` filter to every column in *col_map* using ``Compress``.

    :param g: the graph builder to add ONNX nodes to.
    :param sts: shape/type context dict (forwarded verbatim; may be empty).
    :param outputs: expected output column names (keys of the returned col_map).
    :param op: the :class:`~yobx.sql.parse.FilterOp` describing the condition.
    :param col_map: mapping from column name to current ONNX tensor name.
    :param right_col_map: unused for filter; present for uniform signature.
    :return: a new col_map with each column tensor replaced by its filtered
        (row-compressed) counterpart.
    """
    custom_functions = sts.get("custom_functions", {}) if sts else {}
    emitter = _ExprEmitter(g, col_map, custom_functions=custom_functions)
    mask = emitter.emit(op.condition, name="filter_mask")
    new_map: Dict[str, str] = {}
    for col, tensor in col_map.items():
        compressed = g.op.Compress(tensor, mask, axis=0, name=f"filter_{col}")
        new_map[col] = compressed  # type: ignore
    return new_map
