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
* ``SELECT DISTINCT`` is not yet converted (it is parsed but ignored).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from onnx import ModelProto, TensorProto

from .. import DEFAULT_TARGET_OPSET
from ..xbuilder import GraphBuilder
from .parse import (
    AggExpr,
    BinaryExpr,
    ColumnRef,
    Condition,
    FilterOp,
    GroupByOp,
    JoinOp,
    Literal,
    ParsedQuery,
    SelectItem,
    SelectOp,
    parse_sql,
)

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


# ---------------------------------------------------------------------------
# Expression emitter
# ---------------------------------------------------------------------------


class _ExprEmitter:
    """Emits ONNX nodes for SQL expressions."""

    def __init__(self, g: GraphBuilder, col_map: Dict[str, str]):
        self._g = g
        # col_map: column_name → current ONNX tensor name for that column
        self._col_map = col_map

    def emit(self, node: object, name: str = "sql") -> str:
        """Emit *node* into the graph and return the output tensor name."""
        if isinstance(node, ColumnRef):
            col = node.column
            if col not in self._col_map:
                raise KeyError(
                    f"Column {col!r} not found in inputs. "
                    f"Available: {list(self._col_map)}"
                )
            return self._col_map[col]

        if isinstance(node, Literal):
            v = node.value
            if isinstance(v, bool):
                arr = np.array([v], dtype=np.bool_)
            elif isinstance(v, int):
                arr = np.array([v], dtype=np.int64)
            elif isinstance(v, float):
                arr = np.array([v], dtype=np.float32)
            else:
                raise TypeError(f"Unsupported literal type {type(v)}")
            cst = self._g.make_initializer(f"{name}_lit", arr)
            return cst

        if isinstance(node, BinaryExpr):
            left = self.emit(node.left, name=f"{name}_l")
            right = self.emit(node.right, name=f"{name}_r")
            op_map = {
                "+": "Add",
                "-": "Sub",
                "*": "Mul",
                "/": "Div",
                "=": "Equal",
                "<": "Less",
                ">": "Greater",
                "<=": "LessOrEqual",
                ">=": "GreaterOrEqual",
                "<>": "Not",  # handled below
            }
            # Cast the right operand to match the left operand type when one
            # side is a column (has a known type) and the other is a literal.
            if isinstance(node.right, Literal) and self._g.has_type(left):
                right = self._g.op.CastLike(right, left, name=f"{name}_cast")
            elif isinstance(node.left, Literal) and self._g.has_type(right):
                left = self._g.op.CastLike(left, right, name=f"{name}_cast")
            if node.op == "<>":
                eq = self._g.op.Equal(left, right, name=f"{name}_eq")
                return self._g.op.Not(eq, name=name)
            onnx_op = op_map.get(node.op)
            if onnx_op is None:
                raise ValueError(f"Unsupported binary operator: {node.op!r}")
            return getattr(self._g.op, onnx_op)(left, right, name=name)

        if isinstance(node, Condition):
            op = node.op
            if op == "and":
                left = self.emit(node.left, name=f"{name}_l")
                right = self.emit(node.right, name=f"{name}_r")
                return self._g.op.And(left, right, name=name)
            if op == "or":
                left = self.emit(node.left, name=f"{name}_l")
                right = self.emit(node.right, name=f"{name}_r")
                return self._g.op.Or(left, right, name=name)
            # leaf comparison — treat it as a BinaryExpr
            return self.emit(
                BinaryExpr(left=node.left, op=node.op, right=node.right),
                name=name,
            )

        if isinstance(node, AggExpr):
            func = node.func
            if func == "count":
                # COUNT(*) → total row count of the first known column
                first_col = next(iter(self._col_map.values()))
                size = self._g.op.Shape(first_col, name=f"{name}_shape")
                return self._g.op.Gather(
                    size,
                    np.array(0, dtype=np.int64),
                    axis=0,
                    name=f"{name}_count",
                )
            arg_tensor = self.emit(node.arg, name=f"{name}_arg")
            reduce_axes = self._g.make_initializer(
                f"{name}_axes", np.array([0], dtype=np.int64)
            )
            agg_map = {
                "sum": "ReduceSum",
                "avg": "ReduceMean",
                "min": "ReduceMin",
                "max": "ReduceMax",
            }
            onnx_op = agg_map.get(func)
            if onnx_op is None:
                raise ValueError(f"Unsupported aggregation function: {func!r}")
            return getattr(self._g.op, onnx_op)(
                arg_tensor, reduce_axes, keepdims=0, name=name
            )

        raise TypeError(f"Cannot emit expression of type {type(node)}")


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


def _apply_filter(
    g: GraphBuilder,
    col_map: Dict[str, str],
    condition: Condition,
    name: str = "filter",
) -> Dict[str, str]:
    """
    Apply a WHERE *condition* to every column in *col_map* using ``Compress``.

    Returns a new ``col_map`` mapping each column to a filtered tensor.
    """
    emitter = _ExprEmitter(g, col_map)
    mask = emitter.emit(condition, name=f"{name}_mask")
    new_map: Dict[str, str] = {}
    for col, tensor in col_map.items():
        compressed = g.op.Compress(tensor, mask, axis=0, name=f"{name}_{col}")
        new_map[col] = compressed
    return new_map


def _apply_join(
    g: GraphBuilder,
    col_map: Dict[str, str],
    join_op: JoinOp,
    right_col_map: Dict[str, str],
    name: str = "join",
) -> Dict[str, str]:
    """
    Apply an equi-join between *col_map* (left) and *right_col_map* (right).

    Only inner-join semantics on a single key column are implemented.
    Returns a merged ``col_map`` with all columns from both sides aligned.
    """
    left_key_tensor = col_map[join_op.left_key]
    right_key_tensor = right_col_map[join_op.right_key]

    # For each left row, find the matching index in the right key
    # We implement this as a loop-free broadcast comparison:
    #   match[i,j] = (left_key[i] == right_key[j])
    # then select the first match per left row with ArgMax.
    lk_2d = g.op.Unsqueeze(left_key_tensor, np.array([1], dtype=np.int64), name=f"{name}_lk2d")
    rk_2d = g.op.Unsqueeze(right_key_tensor, np.array([0], dtype=np.int64), name=f"{name}_rk2d")
    match_matrix = g.op.Equal(lk_2d, rk_2d, name=f"{name}_match")

    # Cast bool → int to use ArgMax; pick the first matching right index
    match_int = g.op.Cast(match_matrix, to=TensorProto.INT32, name=f"{name}_match_int")
    right_indices = g.op.ArgMax(match_int, axis=1, name=f"{name}_ridx")

    # Filter left to rows that have at least one match
    any_match = g.op.ReduceMax(
        match_int,
        np.array([1], dtype=np.int64),
        keepdims=0,
        name=f"{name}_any",
    )
    has_match = g.op.Cast(any_match, to=TensorProto.BOOL, name=f"{name}_hasmatch")

    new_map: Dict[str, str] = {}
    # Filter left columns
    for col, tensor in col_map.items():
        compressed = g.op.Compress(tensor, has_match, axis=0, name=f"{name}_l_{col}")
        new_map[col] = compressed

    # Filter right_indices to matched rows, then gather right columns
    right_indices_filtered = g.op.Compress(
        right_indices, has_match, axis=0, name=f"{name}_ridx_filt"
    )
    for col, tensor in right_col_map.items():
        gathered = g.op.Gather(tensor, right_indices_filtered, axis=0, name=f"{name}_r_{col}")
        new_map[col] = gathered

    return new_map


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
        right_dtypes: Dict[str, np.dtype] = {k: np.dtype(v) for k, v in right_input_dtypes.items()}
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
        if isinstance(op, JoinOp):
            col_map = _apply_join(g, col_map, op, right_col_map, name="join")
        elif isinstance(op, FilterOp):
            col_map = _apply_filter(g, col_map, op.condition, name="filter")
        elif isinstance(op, GroupByOp):
            group_op = op  # retained for SelectOp aggregation
        elif isinstance(op, SelectOp):
            select_op = op

    if select_op is None:
        raise ValueError("No SELECT clause found in the query.")

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
