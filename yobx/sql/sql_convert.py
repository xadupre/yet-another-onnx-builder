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
  ``SUM / AVG / MIN / MAX`` aggregations (global, when no ``GROUP BY``).
* ``Unique`` — extracting distinct group keys for ``GROUP BY``.
* ``ScatterElements`` — per-group accumulation for ``GROUP BY`` aggregations.

Limitations
~~~~~~~~~~~
* Only *equi-joins* on a single key column are supported.
* ``GROUP BY`` with ``SUM``, ``AVG``, ``MIN``, ``MAX``, ``COUNT(*)``
  produces one output row per unique key combination.  For multi-column
  ``GROUP BY`` the key columns are internally cast to ``float64`` before
  grouping, which may lose precision for integers larger than 2^53.
* ``SELECT DISTINCT`` is not yet supported and raises :class:`NotImplementedError`.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from onnx import TensorProto

from .. import DEFAULT_TARGET_OPSET
from ..container import ExportArtifact
from ..helpers.to_onnx_helper import _dataframe_to_dtypes, _is_dataframe
from ..typing import GraphBuilderExtendedProtocol
from ..xbuilder import GraphBuilder
from ._expr import _ExprEmitter
from .coverage import not_implemented_error
from .ops import get_sql_op_converter
from yobx.xtracing.parse import GroupByOp, JoinOp, ParsedQuery, SelectOp, parse_sql

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
    g: GraphBuilderExtendedProtocol,
    sts: Optional[Dict],
    outputs: Optional[List[str]],
    query: str,
    input_dtypes: Dict[str, Union[np.dtype, type, str]],
    right_input_dtypes: Optional[Dict[str, Union[np.dtype, type, str]]] = None,
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
    :param sts: usually unused
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
    :return: a list of output tensor names that were added to *g* as model
        outputs (one per expression in the ``SELECT`` clause, in order).

    Example:

    .. runpython::
        :showcode:

        import numpy as np
        from yobx.helpers.onnx_helper import pretty_onnx
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
        art = g.to_onnx()
        print(pretty_onnx(art))
    """
    custom_functions = (sts or {}).get("custom_functions", {})
    pq = parse_sql(query)
    return _populate_graph(
        g,
        pq,
        input_dtypes=input_dtypes,
        right_input_dtypes=right_input_dtypes,
        custom_functions=custom_functions,
        desired_outputs=outputs or None,
    )


def sql_to_onnx(
    query: str,
    input_dtypes: Dict[str, Union[np.dtype, type, str]],
    right_input_dtypes: Optional[Dict[str, Union[np.dtype, type, str]]] = None,
    target_opset: int = DEFAULT_TARGET_OPSET,
    custom_functions: Optional[Dict[str, Callable]] = None,
    builder_cls: Union[type, Callable] = GraphBuilder,
    filename: Optional[str] = None,
    verbose: int = 0,
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
        referenced in the query need to be listed.  A pandas
        :class:`~pandas.DataFrame` is also accepted; column names and dtypes
        are extracted automatically.
    :param right_input_dtypes: if the query contains a ``JOIN``, a mapping
        from *right-table* column name to numpy dtype.  Defaults to
        ``input_dtypes`` when ``None``.  A pandas :class:`~pandas.DataFrame`
        is also accepted.
    :param target_opset: ONNX opset version to target (default:
        :data:`yobx.DEFAULT_TARGET_OPSET`).
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
    :param filename: if set, the exported ONNX model is saved to this path and
        the :class:`~yobx.container.ExportReport` is written as a companion
        Excel file (same base name with ``.xlsx`` extension).
    :param verbose: verbosity level (0 = silent).
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

        ``GROUP BY`` produces one output row per unique key combination.
        Supported aggregations: ``SUM``, ``AVG``, ``MIN``, ``MAX``,
        ``COUNT(*)``.  For multi-column ``GROUP BY`` the grouping keys are
        cast to ``float64`` internally, which may lose precision for integers
        larger than 2^53.
    """
    if _is_dataframe(input_dtypes):  # type: ignore
        input_dtypes = _dataframe_to_dtypes(input_dtypes)
    if _is_dataframe(right_input_dtypes):  # type: ignore
        right_input_dtypes = _dataframe_to_dtypes(right_input_dtypes)
    g = builder_cls(target_opset, ir_version=10)
    sts = {"custom_functions": custom_functions or {}}
    sql_to_onnx_graph(g, sts, [], query, input_dtypes, right_input_dtypes=right_input_dtypes)
    artifact = g.to_onnx(return_optimize_report=True)
    if filename:
        if verbose:
            print(f"[yobx.sql.sql_to_onnx] saving model to {filename!r}")
        artifact.save(filename)
    return artifact


def parsed_query_to_onnx_graph(
    g: GraphBuilderExtendedProtocol,
    sts: Optional[Dict],
    outputs: Optional[List[str]],
    pq: ParsedQuery,
    input_dtypes: Dict[str, Union[np.dtype, type, str]],
    right_input_dtypes: Optional[Dict[str, Union[np.dtype, type, str]]] = None,
    _finalize: bool = True,
) -> List[str]:
    """
    Build ONNX nodes for an already-parsed :class:`~yobx.sql.parse.ParsedQuery`
    into an existing graph builder *g*.

    This is the low-level companion to :func:`sql_to_onnx_graph` for callers
    that have already obtained a :class:`~yobx.sql.parse.ParsedQuery` (e.g.
    via :func:`~yobx.sql.dataframe_trace.trace_dataframe`) and do not want to
    re-serialise it to a SQL string.

    :param g: an existing graph builder.
    :param sts: context dictionary; may contain a ``"custom_functions"`` key.
    :param outputs: expected output column names (may be empty).
    :param pq: the parsed query to convert.
    :param input_dtypes: mapping from left-table column name to numpy dtype.
    :param right_input_dtypes: mapping for right-table columns (JOIN queries).
    :param _finalize: when ``True`` (default) the SELECT output tensors are
        registered as ONNX model outputs via
        :meth:`~yobx.xbuilder.GraphBuilder.make_tensor_output`.  Pass
        ``False`` when embedding the conversion in an outer graph (e.g.
        inside an sklearn converter) so that the caller can register the
        outputs itself.
    :return: list of output tensor names added to *g*.
    """
    custom_functions = (sts or {}).get("custom_functions", {})
    return _populate_graph(
        g,
        pq,
        input_dtypes=input_dtypes,
        right_input_dtypes=right_input_dtypes,
        custom_functions=custom_functions,
        desired_outputs=outputs or None,
        _finalize=_finalize,
    )


def parsed_query_to_onnx(
    pq: ParsedQuery,
    input_dtypes: Dict[str, Union[np.dtype, type, str]],
    right_input_dtypes: Optional[Dict[str, Union[np.dtype, type, str]]] = None,
    target_opset: int = DEFAULT_TARGET_OPSET,
    custom_functions: Optional[Dict[str, Callable]] = None,
    builder_cls: Union[type, Callable] = GraphBuilder,
    filename: Optional[str] = None,
    verbose: int = 0,
) -> ExportArtifact:
    """
    Convert an already-parsed :class:`~yobx.sql.parse.ParsedQuery` to ONNX.

    This is the companion to :func:`sql_to_onnx` for callers that already have
    a :class:`~yobx.sql.parse.ParsedQuery` object (produced by
    :func:`~yobx.sql.parse.parse_sql` or by
    :func:`~yobx.sql.dataframe_trace.trace_dataframe`) and do not need to
    re-serialise it to a SQL string.

    :param pq: a :class:`~yobx.sql.parse.ParsedQuery` produced by
        :func:`~yobx.sql.parse.parse_sql` or
        :func:`~yobx.sql.dataframe_trace.trace_dataframe`.
    :param input_dtypes: mapping from left-table column name to numpy dtype.
    :param right_input_dtypes: mapping for right-table columns (JOIN queries).
    :param target_opset: ONNX opset version to target.
    :param custom_functions: optional mapping from function name to Python
        callable.  Each callable is traced via
        :func:`~yobx.xtracing.trace_numpy_function`.
    :param builder_cls: graph-builder class or factory callable.
    :param filename: if set, the exported ONNX model is saved to this path and
        the :class:`~yobx.container.ExportReport` is written as a companion
        Excel file (same base name with ``.xlsx`` extension).
    :param verbose: verbosity level (0 = silent).
    :return: :class:`~yobx.container.ExportArtifact` wrapping the exported
        ONNX model together with an :class:`~yobx.container.ExportReport`.

    Example::

        import numpy as np
        from yobx.sql import parse_sql
        from yobx.sql.sql_convert import parsed_query_to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        pq = parse_sql("SELECT a + b AS total FROM t WHERE a > 0")
        dtypes = {"a": np.float32, "b": np.float32}
        artifact = parsed_query_to_onnx(pq, dtypes)

        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0,  5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
    """
    g = builder_cls(target_opset, ir_version=10)
    sts = {"custom_functions": custom_functions or {}}
    parsed_query_to_onnx_graph(g, sts, [], pq, input_dtypes, right_input_dtypes)
    artifact = g.to_onnx(return_optimize_report=True)
    if filename:
        if verbose:
            print(f"[yobx.sql.parsed_query_to_onnx] saving model to {filename!r}")
        artifact.save(filename)
    return artifact


# ---------------------------------------------------------------------------
# Internal ONNX graph builder
# ---------------------------------------------------------------------------


def _build_group_by_tensors(
    g: GraphBuilderExtendedProtocol, group_op: GroupByOp, col_map: Dict[str, str]
) -> Tuple[str, str, str, Dict[str, str]]:
    """Compute ONNX tensors for GROUP BY and return an updated column map.

    Uses ONNX ``Unique`` to identify the distinct groups and produces two
    index tensors that are later consumed by :class:`~yobx.sql._expr._ExprEmitter`
    to implement per-group aggregations via ``ScatterElements``.

    :param g: the graph builder.
    :param group_op: the :class:`~yobx.sql.parse.GroupByOp` describing the
        grouping columns.
    :param col_map: current column-name → ONNX-tensor-name mapping (after
        any prior ``WHERE`` / ``JOIN`` processing).
    :return: a 4-tuple ``(inverse_indices, first_indices, n_groups, updated_col_map)``
        where:

        * *inverse_indices* — tensor name, shape ``(N,)`` int64: for each
          input row, the index of its group in the sorted unique-groups list.
        * *first_indices* — tensor name, shape ``(n_groups,)`` int64: for each
          group, the row index of its first occurrence in the input.
        * *n_groups* — tensor name, scalar int64: number of unique groups.
        * *updated_col_map* — copy of *col_map* where every GROUP BY key column
          is remapped to its per-group unique-value tensor so that plain column
          references in SELECT produce one value per group.
    """
    group_cols = group_op.columns

    if len(group_cols) == 1:
        # Single GROUP BY column: apply Unique directly.
        col_tensor = col_map[group_cols[0]]
        unique_vals, first_indices, inverse_indices, _ = g.op.Unique(  # type: ignore[misc]
            col_tensor, sorted=1, outputs=4, name="gb_unique"
        )
        n_groups = g.op.Gather(
            g.op.Shape(unique_vals, name="gb_uvals_shape"),
            np.array(0, dtype=np.int64),
            axis=0,
            name="gb_n_groups",
        )
        updated_col_map = dict(col_map)
        updated_col_map[group_cols[0]] = unique_vals  # type: ignore[assignment]
    else:
        # Multiple GROUP BY columns: cast each to float64, stack into a 2-D
        # matrix and apply Unique with axis=0 so that each unique row
        # corresponds to one distinct key combination.
        col_tensors = [col_map[c] for c in group_cols]
        cast_tensors = [
            g.op.Cast(t, to=TensorProto.DOUBLE, name=f"gb_cast_{c}")  # type: ignore[misc]
            for t, c in zip(col_tensors, group_cols)
        ]
        unsqueezed = [
            g.op.Unsqueeze(t, np.array([1], dtype=np.int64), name=f"gb_unsq_{c}")  # type: ignore[misc]
            for t, c in zip(cast_tensors, group_cols)
        ]
        stacked = g.op.Concat(*unsqueezed, axis=1, name="gb_stack")  # type: ignore[misc]
        unique_rows, first_indices, inverse_indices, _ = g.op.Unique(  # type: ignore[misc]
            stacked, axis=0, sorted=1, outputs=4, name="gb_unique"
        )
        n_groups = g.op.Gather(
            g.op.Shape(unique_rows, name="gb_urows_shape"),
            np.array(0, dtype=np.int64),
            axis=0,
            name="gb_n_groups",
        )
        # Remap each GROUP BY key column to its per-group unique values
        # (gathered from the original, pre-cast column to preserve dtype).
        updated_col_map = dict(col_map)
        for j, c in enumerate(group_cols):
            col_unique_f64 = g.op.Gather(  # type: ignore[misc]
                unique_rows, np.array(j, dtype=np.int64), axis=1, name=f"gb_col_{c}_f64"
            )
            # Cast back to the original column's dtype.
            col_unique = g.op.CastLike(  # type: ignore[misc]
                col_unique_f64, col_map[c], name=f"gb_col_{c}"
            )
            updated_col_map[c] = col_unique  # type: ignore[assignment]

    return (  # type: ignore[return-value]
        inverse_indices,
        first_indices,
        n_groups,
        updated_col_map,
    )


def _populate_graph(
    g: GraphBuilderExtendedProtocol,
    pq: ParsedQuery,
    input_dtypes: Dict[str, Union[np.dtype, type, str]],
    right_input_dtypes: Optional[Dict[str, Union[np.dtype, type, str]]],
    custom_functions: Optional[Dict[str, Callable]] = None,
    desired_outputs: Optional[List[str]] = None,
    _finalize: bool = True,
) -> List[str]:
    """
    Populates *g* with ONNX nodes for *pq* and return the output tensor names.

    :param _finalize: when ``True`` (default) the SELECT output tensors are
        registered as ONNX model outputs.  Pass ``False`` when processing an
        inner subquery so that its outputs remain intermediate tensors.
    """
    dim = "N"

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
        raise not_implemented_error("sql", "SELECT DISTINCT")

    # Compute GROUP BY tensors when a GROUP BY clause is present.
    gb_inverse_indices: Optional[str] = None
    gb_first_indices: Optional[str] = None
    gb_n_groups: Optional[str] = None
    effective_col_map = col_map
    if _group_op is not None:
        gb_inverse_indices, gb_first_indices, gb_n_groups, effective_col_map = (
            _build_group_by_tensors(g, _group_op, col_map)
        )

    # Emit SELECT expressions
    emitter = _ExprEmitter(  # type: ignore
        g,
        effective_col_map,
        custom_functions=resolved_custom_functions,
        group_by_inverse_indices=gb_inverse_indices,
        group_by_first_indices=gb_first_indices,
        group_by_n_groups=gb_n_groups,
    )
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
