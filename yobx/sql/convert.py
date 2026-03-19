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

from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from onnx import TensorProto

from .. import DEFAULT_TARGET_OPSET
from ..container import ExportArtifact, ExportReport
from ..typing import GraphBuilderProtocol
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


def sql_to_onnx_graph(
    g: GraphBuilderProtocol,
    sts: Optional[Dict],
    outputs: List[str],
    query: str,
    input_dtypes: Dict[str, Union[np.dtype, type, str]],
    right_input_dtypes: Optional[Dict[str, Union[np.dtype, type, str]]] = None,
    n_rows: Optional[int] = None,
) -> List[str]:
    """
    Build ONNX nodes for a SQL *query* into an existing graph builder *g*.

    This is the low-level entry point for callers that are already managing a
    :class:`~yobx.typing.GraphBuilderProtocol` instance (e.g. as part of a
    larger model).  The signature follows the standard SQL op converter
    convention ``(g, sts, outputs, ...)``.  Any column referenced in the
    query that is not yet registered as an input in *g* is added
    automatically.  The SELECT expressions are emitted as model outputs and
    their tensor names are returned.

    :param g: an existing graph builder that satisfies
        :class:`~yobx.typing.GraphBuilderProtocol`.  Nodes and inputs are
        added to this builder in-place.
    :param sts: shape/type context dict forwarded to SQL op converters, or
        ``None``.  May contain:

        * ``"custom_functions"`` — a mapping from function name (as it
          appears in the SQL string) to a Python callable.  Each callable
          must accept one or more numpy arrays and return a numpy array.  The
          function body is traced with
          :func:`~yobx.xtracing.trace_numpy_function` so that numpy
          arithmetic is translated into ONNX nodes.

        When ``None`` (or an empty dict) no custom functions are available.

    :param outputs: expected output column names for the query result.
        Passed through to individual op converters following the standard
        converter convention; may be an empty list when the caller does not
        know the output names in advance.
    :param query: a SQL string.  Supported clauses:
        ``SELECT``, ``FROM``, ``[INNER|LEFT|RIGHT|FULL] JOIN … ON``,
        ``WHERE``, ``GROUP BY``.
    :param input_dtypes: a mapping from *left-table* column name to numpy
        dtype (``np.float32``, ``np.int64``, etc.).  Only columns actually
        referenced in the query need to be listed.
    :param right_input_dtypes: if the query contains a ``JOIN``, a mapping
        from *right-table* column name to numpy dtype.  Defaults to
        ``input_dtypes`` when ``None``.
    :param n_rows: optional static number of rows; used to fix the first
        dimension of every input tensor that is newly added to *g*.  When
        ``None`` the first dimension is symbolic (``"N"``).
    :return: a list of output tensor names that were added to *g* as model
        outputs (one per expression in the ``SELECT`` clause, in order).

    Example::

        import numpy as np
        from yobx.xbuilder import GraphBuilder
        from yobx.sql import sql_to_onnx_graph

        g = GraphBuilder(18, ir_version=10)
        dtypes = {"a": np.float32, "b": np.float32}
        out_names = sql_to_onnx_graph(
            g,
            None,
            [],
            "SELECT a + b AS total FROM t WHERE a > 0",
            dtypes,
        )
        onx, _ = g.to_onnx(return_optimize_report=True)
    """
    custom_functions = (sts or {}).get("custom_functions", {})
    pq = parse_sql(query)
    return _populate_graph(
        g,
        pq,
        input_dtypes=input_dtypes,
        right_input_dtypes=right_input_dtypes,
        n_rows=n_rows,
        custom_functions=custom_functions,
        desired_outputs=outputs or None,
    )


def sql_to_onnx(
    query: str,
    input_dtypes: Dict[str, Union[np.dtype, type, str]],
    right_input_dtypes: Optional[Dict[str, Union[np.dtype, type, str]]] = None,
    target_opset: int = DEFAULT_TARGET_OPSET,
    n_rows: Optional[int] = None,
    custom_functions: Optional[Dict[str, Callable]] = None,
    builder_cls: Union[type, Callable] = GraphBuilder,
) -> ExportArtifact:
    """
    Convert a SQL *query* to a self-contained ONNX model.

    Each column in the query is represented as a **separate 1-D ONNX input**
    tensor, allowing the caller to feed column vectors independently.  The
    resulting model's outputs correspond to the columns (or expressions) in
    the ``SELECT`` clause, in order.

    Internally this function creates a fresh
    :class:`~yobx.xbuilder.GraphBuilder` (or the class supplied via
    *builder_cls*), delegates to :func:`sql_to_onnx_graph` to populate it,
    and then calls :meth:`~yobx.xbuilder.GraphBuilder.to_onnx` to finalise
    the model.
    Use :func:`sql_to_onnx_graph` directly when you need to embed the SQL
    subgraph inside a larger ONNX model you are already building.

    :param query: a SQL string.  Supported clauses:
        ``SELECT``, ``FROM``, ``[INNER|LEFT|RIGHT|FULL] JOIN … ON``,
        ``WHERE``, ``GROUP BY``.
        Custom Python functions can be called by name in the ``SELECT`` and
        ``WHERE`` clauses when registered via *custom_functions*.
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
    :param custom_functions: an optional mapping from function name (as it
        appears in the SQL string) to a Python callable.  Each callable must
        accept one or more numpy arrays and return a numpy array.  The
        function body is traced with :func:`~yobx.xtracing.trace_numpy_function`
        so that numpy arithmetic is translated into ONNX nodes.

        Example::

            import numpy as np
            from yobx.sql import sql_to_onnx

            dtypes = {"a": np.float32}
            artifact = sql_to_onnx(
                "SELECT my_sqrt(a) AS r FROM t",
                dtypes,
                custom_functions={"my_sqrt": np.sqrt},
            )

    :param builder_cls: the graph-builder class (or factory callable) to
        instantiate when creating the internal
        :class:`~yobx.xbuilder.GraphBuilder`.  Defaults to
        :class:`~yobx.xbuilder.GraphBuilder`.  Any class that implements
        the :ref:`builder-api` can be supplied here, e.g. a custom subclass
        that adds extra optimisation passes.

    :return: :class:`~yobx.container.ExportArtifact` wrapping the exported
        ONNX proto together with an :class:`~yobx.container.ExportReport`.

    Example::

        import numpy as np
        from yobx.sql import sql_to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        dtypes = {"a": np.float32, "b": np.float32}
        artifact = sql_to_onnx("SELECT a + b AS total FROM t WHERE a > 0", dtypes)

        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0,  5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})

    .. note::

        ``GROUP BY`` aggregations are computed over the **whole filtered
        dataset**.  True SQL group-by semantics (one output row per unique
        key) would require an ONNX ``Loop`` or custom kernel and are not
        yet supported.
    """
    g = builder_cls(target_opset, ir_version=10)
    sts = {"custom_functions": custom_functions or {}}
    sql_to_onnx_graph(
        g, sts, [], query, input_dtypes, right_input_dtypes=right_input_dtypes, n_rows=n_rows
    )
    onx, stats = g.to_onnx(return_optimize_report=True)  # type: ignore
    return ExportArtifact(proto=onx, report=ExportReport(stats=stats or []))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Internal ONNX graph builder
# ---------------------------------------------------------------------------


def _populate_graph(
    g: GraphBuilderProtocol,
    pq: ParsedQuery,
    input_dtypes: Dict[str, Union[np.dtype, type, str]],
    right_input_dtypes: Optional[Dict[str, Union[np.dtype, type, str]]],
    n_rows: Optional[int],
    custom_functions: Optional[Dict[str, Callable]] = None,
    desired_outputs: Optional[List[str]] = None,
    _finalize: bool = True,
) -> List[str]:
    """Populate *g* with ONNX nodes for *pq* and return the output tensor names.

    :param _finalize: when ``True`` (default) the SELECT output tensors are
        registered as ONNX model outputs.  Pass ``False`` when processing an
        inner subquery so that its outputs remain intermediate tensors.
    """
    dim = n_rows if n_rows is not None else "N"

    # Resolve dtype dicts
    left_dtypes: Dict[str, np.dtype] = {k: np.dtype(v) for k, v in input_dtypes.items()}
    if right_input_dtypes is not None:
        right_dtypes: Dict[str, np.dtype] = {
            k: np.dtype(v) for k, v in right_input_dtypes.items()
        }
    else:
        right_dtypes = left_dtypes

    resolved_custom_functions = custom_functions or {}

    if pq.subquery is not None:
        # Process the inner subquery first.  Its SELECT outputs become the
        # column tensors for the outer query.
        inner_output_names = _populate_graph(
            g,
            pq.subquery,
            input_dtypes,
            right_input_dtypes,
            n_rows,
            custom_functions=resolved_custom_functions,
            desired_outputs=None,
            _finalize=False,
        )
        # Build col_map: inner SELECT alias → ONNX tensor name
        inner_select_op: Optional[SelectOp] = next(
            (op for op in pq.subquery.operations if isinstance(op, SelectOp)), None
        )
        if inner_select_op is None:
            raise ValueError("Subquery has no SELECT clause.")
        col_map: Dict[str, str] = {
            item.output_name(): tensor_name
            for item, tensor_name in zip(inner_select_op.items, inner_output_names)
        }
        right_col_map: Dict[str, str] = {}
    else:
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

        # Add inputs to the graph only when not already registered
        for inp_name, inp_type, inp_shape in all_inputs:
            if not g.has_name(inp_name):
                g.make_tensor_input(inp_name, inp_type, inp_shape)

        # col_map: current ONNX tensor name for each column (left side)
        col_map = {name: name for name, _, _ in left_inputs}
        right_col_map = {name: name for name, _, _ in right_inputs}

    select_op: Optional[SelectOp] = None
    _group_op: Optional[GroupByOp] = None

    for op in pq.operations:
        converter = get_sql_op_converter(type(op))
        if converter is not None:
            sts_ctx = {"custom_functions": resolved_custom_functions}
            col_map = converter(g, sts_ctx, list(col_map.keys()), op, col_map, right_col_map)
        elif isinstance(op, GroupByOp):
            _group_op = op  # retained for SelectOp aggregation
        elif isinstance(op, SelectOp):
            select_op = op

    if select_op is None:
        raise ValueError("No SELECT clause found in the query.")

    if select_op.distinct:
        raise NotImplementedError(
            f"SELECT DISTINCT is not yet supported by the ONNX converter, {select_op=}"
        )

    # Emit SELECT expressions
    emitter = _ExprEmitter(g, col_map, custom_functions=resolved_custom_functions)  # type: ignore
    output_names: List[str] = []
    used_tensor_names: Set[str] = set()
    for i, item in enumerate(select_op.items):
        out_name = item.output_name()
        # Determine the desired tensor name: caller-supplied, then query alias, then indexed
        if desired_outputs and i < len(desired_outputs):
            candidate = desired_outputs[i]
        else:
            candidate = out_name
        # Fall back to an indexed name if the candidate conflicts with existing names
        if g.has_name(candidate) or candidate in used_tensor_names:
            # Find a free indexed name (output_{i}, output_{i+1}, ...) that is not taken
            j = i
            candidate = f"output_{j}"
            while g.has_name(candidate) or candidate in used_tensor_names:
                j += 1
                candidate = f"output_{j}"
        tensor_name = candidate
        used_tensor_names.add(tensor_name)
        result = emitter.emit(item.expr, name=f"select_{out_name}")
        g.op.Identity(result, outputs=[tensor_name], name=f"out_{out_name}")  # type: ignore
        output_names.append(tensor_name)

    if _finalize:
        for out_name in output_names:
            g.make_tensor_output(out_name, indexed=False, allow_untyped_output=True)

    return output_names
