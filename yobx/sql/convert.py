"""
SQL to ONNX converter.

:func:`sql_to_onnx` converts a SQL query string into a self-contained
:class:`onnx.ModelProto`.

Design
------
Every referenced *column* becomes a **separate 1-D ONNX input** so that the
caller can supply column vectors independently (matching the convention used
for tabular data).  The converter processes the :class:`~yobx.sql.parse.ParsedQuery`
operation list in execution order:

1. :class:`~yobx.sql.parse.JoinOp` — equi-join on a key column.
2. :class:`~yobx.sql.parse.FilterOp` — boolean mask applied to all columns.
3. :class:`~yobx.sql.parse.GroupByOp` — group key used for aggregations.
4. :class:`~yobx.sql.parse.SelectOp` — column expressions emitted as outputs.

ONNX operations used
~~~~~~~~~~~~~~~~~~~~
* ``Compress`` — row-wise filtering (WHERE clause).
* ``Gather`` — index-based row selection (JOIN / GROUP BY key alignment).
* ``Add``, ``Sub``, ``Mul``, ``Div`` — arithmetic expressions.
* ``Equal``, ``Less``, ``Greater``, ``LessOrEqual``, ``GreaterOrEqual`` —
  comparison predicates.
* ``And``, ``Or`` — compound predicates.
* ``ReduceSum``, ``ReduceMean``, ``ReduceMin``, ``ReduceMax`` —
  ``SUM / AVG / MIN / MAX`` aggregations.

Limitations
~~~~~~~~~~~
* Only *equi-joins* on a single key column are supported.
* ``GROUP BY`` aggregation is limited to the row-wise functions listed above;
  true SQL group-by semantics (unique key extraction) are not yet implemented
  as ONNX lacks a native GroupBy node.
* ``COUNT(*)`` emits the total row count as a scalar ``int64`` tensor.
* ``SELECT DISTINCT`` is not yet supported and raises :class:`NotImplementedError`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from onnx import ModelProto, TensorProto

from .. import DEFAULT_TARGET_OPSET
from ..xbuilder import GraphBuilder
from ._expr import _ExprEmitter
from .ops import get_sql_op_converter
from .parse import GroupByOp, JoinOp, ParsedQuery, SelectOp, parse_sql

# ---------------------------------------------------------------------------
# Dtype helper
# ---------------------------------------------------------------------------

_NP_TO_ONNX: Dict[np.dtype, int] = {
    np.dtype("float32"): TensorProto.FLOAT,
    np.dtype("float64"): TensorProto.DOUBLE,
    np.dtype("int32"): TensorProto.INT32,
    np.dtype("int64"): TensorProto.INT64,
    np.dtype("bool"): TensorProto.BOOL,
    np.dtype("object"): TensorProto.STRING,
}


def _np_dtype_to_onnx(dt: Union[np.dtype, type, str]) -> int:
    dt = np.dtype(dt)
    if dt in _NP_TO_ONNX:
        return _NP_TO_ONNX[dt]
    raise ValueError(f"Unsupported numpy dtype for SQL conversion: {dt}")


def sql_to_onnx(
    query: str,
    input_dtypes: Dict[str, Union[np.dtype, type, str]],
    right_input_dtypes: Optional[Dict[str, Union[np.dtype, type, str]]] = None,
    target_opset: int = DEFAULT_TARGET_OPSET,
    n_rows: Optional[int] = None,
) -> ModelProto:
    """
    Convert a SQL *query* to a self-contained :class:`onnx.ModelProto`.

    Each column in the query is represented as a **separate 1-D ONNX input**
    tensor, allowing the caller to feed column vectors independently.  The
    resulting model's outputs correspond to the columns (or expressions) in
    the ``SELECT`` clause, in order.

    :param query: a SQL string.  Supported clauses:
        ``SELECT``, ``FROM``, ``[INNER|LEFT|RIGHT|FULL] JOIN … ON``,
        ``WHERE``, ``GROUP BY``.
    :param input_dtypes: a mapping from *left-table* column name to numpy
        dtype (``np.float32``, ``np.int64``, etc.).  Only columns actually
        referenced in the query need to be listed.
    :param right_input_dtypes: if the query contains a ``JOIN``, a mapping
        from *right-table* column name to numpy dtype.  Defaults to
        ``input_dtypes`` when ``None``.
    :param target_opset: ONNX opset version to target (default:
        :data:`yobx.DEFAULT_TARGET_OPSET`).
    :param n_rows: optional static number of rows; used to fix the first
        dimension of every input tensor.  When ``None`` the first dimension is
        symbolic (``"N"``).
    :return: a :class:`onnx.ModelProto` ready for inference.

    Example::

        import numpy as np
        from yobx.sql import sql_to_onnx

        dtypes = {"a": np.float32, "b": np.float32}
        onx = sql_to_onnx("SELECT a + b AS total FROM t WHERE a > 0", dtypes)

    .. note::

        ``GROUP BY`` aggregations are computed over the **whole filtered
        dataset**.  True SQL group-by semantics (one output row per unique
        key) would require an ONNX ``Loop`` or custom kernel and are not
        yet supported.
    """
    pq = parse_sql(query)
    return _build_onnx(
        pq,
        input_dtypes=input_dtypes,
        right_input_dtypes=right_input_dtypes,
        target_opset=target_opset,
        n_rows=n_rows,
    )


# ---------------------------------------------------------------------------
# Internal ONNX graph builder
# ---------------------------------------------------------------------------


def _build_onnx(
    pq: ParsedQuery,
    input_dtypes: Dict[str, Union[np.dtype, type, str]],
    right_input_dtypes: Optional[Dict[str, Union[np.dtype, type, str]]],
    target_opset: int,
    n_rows: Optional[int],
) -> ModelProto:
    dim = n_rows if n_rows is not None else "N"

    # Resolve dtype dicts
    left_dtypes: Dict[str, np.dtype] = {k: np.dtype(v) for k, v in input_dtypes.items()}
    if right_input_dtypes is not None:
        right_dtypes: Dict[str, np.dtype] = {
            k: np.dtype(v) for k, v in right_input_dtypes.items()
        }
    else:
        right_dtypes = left_dtypes

    # Build list of inputs needed for left table
    # (only columns that appear in the query)
    all_cols = pq.columns
    left_inputs: List[Tuple[str, int, Tuple]] = []
    right_inputs: List[Tuple[str, int, Tuple]] = []

    seen_right: List[str] = []
    join_op: Optional[JoinOp] = None
    for op in pq.operations:
        if isinstance(op, JoinOp):
            join_op = op
            break

    for col in all_cols:
        if col in left_dtypes:
            onnx_type = _np_dtype_to_onnx(left_dtypes[col])
            left_inputs.append((col, onnx_type, (dim,)))
        elif col in right_dtypes and join_op is not None:
            onnx_type = _np_dtype_to_onnx(right_dtypes[col])
            right_inputs.append((col, onnx_type, (dim,)))
            seen_right.append(col)

    # If we have a join, make sure key columns are registered on the right side
    if join_op is not None and join_op.right_key not in seen_right:
        if join_op.right_key in right_dtypes:
            onnx_type = _np_dtype_to_onnx(right_dtypes[join_op.right_key])
            right_inputs.append((join_op.right_key, onnx_type, (dim,)))

    all_inputs = left_inputs + right_inputs

    g = GraphBuilder(target_opset, ir_version=10)
    for inp_name, inp_type, inp_shape in all_inputs:
        g.make_tensor_input(inp_name, inp_type, inp_shape)

    # col_map: current ONNX tensor name for each column (left side)
    col_map: Dict[str, str] = {name: name for name, _, _ in left_inputs}
    right_col_map: Dict[str, str] = {name: name for name, _, _ in right_inputs}

    select_op: Optional[SelectOp] = None
    group_op: Optional[GroupByOp] = None

    for op in pq.operations:
        converter = get_sql_op_converter(type(op))
        if converter is not None:
            col_map = converter(g, {}, list(col_map.keys()), op, col_map, right_col_map)
        elif isinstance(op, GroupByOp):
            group_op = op  # retained for SelectOp aggregation
        elif isinstance(op, SelectOp):
            select_op = op

    if select_op is None:
        raise ValueError("No SELECT clause found in the query.")

    if group_op is None:
        raise ValueError("GROUP bY does not seem implemented.")

    if select_op.distinct:
        raise NotImplementedError("SELECT DISTINCT is not yet supported by the ONNX converter.")

    # Emit SELECT expressions
    emitter = _ExprEmitter(g, col_map)
    output_names: List[str] = []
    for i, item in enumerate(select_op.items):
        out_name = item.output_name()
        # Use indexed tensor name to avoid collision with input names
        tensor_name = f"output_{i}"
        result = emitter.emit(item.expr, name=f"select_{out_name}")
        # Always emit an Identity node to give the output tensor a unique indexed name
        g.op.Identity(result, outputs=[tensor_name], name=f"out_{out_name}")
        output_names.append(tensor_name)

    for out_name in output_names:
        g.make_tensor_output(out_name, indexed=False, allow_untyped_output=True)

    onx, _ = g.to_onnx(return_optimize_report=True)
    return onx  # type: ignore[return-value]
