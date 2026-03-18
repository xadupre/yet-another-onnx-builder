"""
ONNX converter for :class:`~yobx.sql.parse.JoinOp` (SQL ``JOIN`` clause).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from onnx import TensorProto

from ...xbuilder import GraphBuilder
from ..parse import JoinOp
from .register import register_sql_op_converter


@register_sql_op_converter(JoinOp)
def convert_join_op(
    g: GraphBuilder,
    sts: Optional[Dict],
    outputs: List[str],
    op: JoinOp,
    col_map: Dict[str, str],
    right_col_map: Dict[str, str],
) -> Dict[str, str]:
    """
    Apply an equi-join between the left *col_map* and the right *right_col_map*.

    Only inner-join semantics on a single key column are implemented.  For
    each left row the converter finds the first matching right row by
    broadcasting an equality check over both key tensors, then uses
    ``Compress`` (to drop non-matching left rows) and ``Gather`` (to align
    right-table rows with their matching left-table rows).

    :param g: the graph builder to add ONNX nodes to.
    :param sts: shape/type context dict (forwarded verbatim; may be empty).
    :param outputs: expected output column names (keys of the returned col_map).
    :param op: the :class:`~yobx.sql.parse.JoinOp` describing the join keys.
    :param col_map: mapping from column name to current ONNX tensor name
        (left table).
    :param right_col_map: same for the right table.
    :return: a merged col_map containing all columns from both tables, aligned
        on the join key.
    """
    left_key_tensor = col_map[op.left_key]
    right_key_tensor = right_col_map[op.right_key]

    # Broadcast equality: match[i,j] = (left_key[i] == right_key[j])
    lk_2d = g.op.Unsqueeze(left_key_tensor, np.array([1], dtype=np.int64), name="join_lk2d")
    rk_2d = g.op.Unsqueeze(right_key_tensor, np.array([0], dtype=np.int64), name="join_rk2d")
    match_matrix = g.op.Equal(lk_2d, rk_2d, name="join_match")

    # Cast bool → int32 to use ArgMax; pick the first matching right index
    match_int = g.op.Cast(match_matrix, to=TensorProto.INT32, name="join_match_int")
    right_indices = g.op.ArgMax(match_int, axis=1, name="join_ridx")

    # Filter left to rows that have at least one match
    any_match = g.op.ReduceMax(
        match_int, np.array([1], dtype=np.int64), keepdims=0, name="join_any"
    )
    has_match = g.op.Cast(any_match, to=TensorProto.BOOL, name="join_hasmatch")

    new_map: Dict[str, str] = {}
    # Compress left columns to matched rows
    for col, tensor in col_map.items():
        compressed = g.op.Compress(tensor, has_match, axis=0, name=f"join_l_{col}")
        new_map[col] = compressed  # type: ignore

    # Compress right_indices then Gather right columns
    right_indices_filtered = g.op.Compress(
        right_indices, has_match, axis=0, name="join_ridx_filt"
    )
    for col, tensor in right_col_map.items():
        gathered = g.op.Gather(tensor, right_indices_filtered, axis=0, name=f"join_r_{col}")
        new_map[col] = gathered  # type: ignore

    return new_map
