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
import onnx.numpy_helper as onh
from onnx import TensorProto

from .. import DEFAULT_TARGET_OPSET
from ..container import ExportArtifact
from ..helpers.to_onnx_helper import _dataframe_to_dtypes, _is_dataframe
from ..typing import GraphBuilderExtendedProtocol
from ..xbuilder import GraphBuilder
from ._expr import _ExprEmitter
from .coverage import not_implemented_error
from .ops import get_sql_op_converter
from yobx.xtracing.parse import GroupByOp, JoinOp, ParsedQuery, PivotTableOp, SelectOp, parse_sql

# ---------------------------------------------------------------------------
# Dtype helper
# ---------------------------------------------------------------------------

_NP_TO_ONNX: Dict[np.dtype, int] = {
    np.dtype("float16"): TensorProto.FLOAT16,
    np.dtype("float32"): TensorProto.FLOAT,
    np.dtype("float64"): TensorProto.DOUBLE,
    np.dtype("int8"): TensorProto.INT8,
    np.dtype("int16"): TensorProto.INT16,
    np.dtype("int32"): TensorProto.INT32,
    np.dtype("int64"): TensorProto.INT64,
    np.dtype("uint8"): TensorProto.UINT8,
    np.dtype("uint16"): TensorProto.UINT16,
    np.dtype("uint32"): TensorProto.UINT32,
    np.dtype("uint64"): TensorProto.UINT64,
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
    right_input_dtypes: Optional[
        Union[Dict[str, Union[np.dtype, type, str]], List[Dict[str, Union[np.dtype, type, str]]]]
    ] = None,
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
    right_input_dtypes: Optional[
        Union[Dict[str, Union[np.dtype, type, str]], List[Dict[str, Union[np.dtype, type, str]]]]
    ] = None,
    target_opset: int = DEFAULT_TARGET_OPSET,
    custom_functions: Optional[Dict[str, Callable]] = None,
    builder_cls: Union[type, Callable] = GraphBuilder,
    filename: Optional[str] = None,
    verbose: int = 0,
    large_model: bool = False,
    external_threshold: int = 1024,
    return_optimize_report: bool = False,
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
    :param right_input_dtypes: for queries with a single ``JOIN``, a mapping
        from *right-table* column name to numpy dtype.  For queries with
        **multiple JOINs**, pass a list of such mappings where the *i*-th dict
        covers the *i*-th right table (in JOIN order).  A single dict may also
        be used with multiple JOINs when all right tables share the same column
        schema (backward compatible).  Defaults to ``input_dtypes`` when
        ``None``.  A pandas :class:`~pandas.DataFrame` (or a list thereof) is
        also accepted; column names and dtypes are extracted automatically.
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
    :param large_model: if True the returned :class:`~yobx.container.ExportArtifact`
        has its :attr:`~yobx.container.ExportArtifact.container` attribute set to
        an :class:`~yobx.container.ExtendedModelContainer`
    :param external_threshold: if ``large_model`` is True, every tensor whose
        element count exceeds this threshold is stored as external data
    :param return_optimize_report: if True, the returned
        :class:`~yobx.container.ExportArtifact` has its
        :attr:`~yobx.container.ExportArtifact.report` attribute populated with
        per-pattern optimization statistics
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
        input_dtypes = _dataframe_to_dtypes(input_dtypes)  # type: ignore
    if isinstance(right_input_dtypes, (list, tuple)):
        right_input_dtypes = [  # type: ignore
            _dataframe_to_dtypes(rd) if _is_dataframe(rd) else rd  # type: ignore
            for rd in right_input_dtypes
        ]
    elif _is_dataframe(right_input_dtypes):  # type: ignore
        right_input_dtypes = _dataframe_to_dtypes(right_input_dtypes)  # type: ignore
    g = builder_cls(target_opset, ir_version=10)
    sts = {"custom_functions": custom_functions or {}}
    sql_to_onnx_graph(g, sts, [], query, input_dtypes, right_input_dtypes=right_input_dtypes)
    if isinstance(g, GraphBuilder):
        artifact = g.to_onnx(
            large_model=large_model,
            external_threshold=external_threshold,
            return_optimize_report=return_optimize_report,
        )
    else:
        artifact = g.to_onnx(large_model=large_model, external_threshold=external_threshold)
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
    _finalize: bool = True,
) -> List[str]:
    """
    Build ONNX nodes for an already-parsed :class:`~yobx.sql.parse.ParsedQuery`
    into an existing graph builder *g*.

    This is the low-level companion to :func:`sql_to_onnx_graph` for callers
    that have already obtained a :class:`~yobx.sql.parse.ParsedQuery` produced
    by :func:`~yobx.xtracing.dataframe_trace.trace_dataframe`.  All type
    information is read from the :attr:`~yobx.xtracing.parse.ColumnRef.dtype`
    fields that the tracer populates; no ``input_dtypes`` argument is needed.

    :param g: an existing graph builder.
    :param sts: context dictionary; may contain a ``"custom_functions"`` key.
    :param outputs: expected output column names (may be empty).
    :param pq: the parsed query to convert.  Must have been produced by
        :func:`~yobx.xtracing.dataframe_trace.trace_dataframe` so that all
        :class:`~yobx.xtracing.parse.ColumnRef` objects carry a non-zero
        ``dtype``.
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
        custom_functions=custom_functions,
        desired_outputs=outputs or None,
        _finalize=_finalize,
    )


def parsed_query_to_onnx(
    pq: Union[ParsedQuery, List[ParsedQuery]],
    target_opset: int = DEFAULT_TARGET_OPSET,
    custom_functions: Optional[Dict[str, Callable]] = None,
    builder_cls: Union[type, Callable] = GraphBuilder,
    filename: Optional[str] = None,
    verbose: int = 0,
    large_model: bool = False,
    external_threshold: int = 1024,
    return_optimize_report: bool = False,
) -> ExportArtifact:
    """
    Convert an already-parsed :class:`~yobx.sql.parse.ParsedQuery` to ONNX.

    The query must have been produced by
    :func:`~yobx.xtracing.dataframe_trace.trace_dataframe` so that all
    :class:`~yobx.xtracing.parse.ColumnRef` objects carry a non-zero
    :attr:`~yobx.xtracing.parse.ColumnRef.dtype`.  Type information is read
    exclusively from those ``dtype`` fields; no ``input_dtypes`` argument is
    needed.  For queries produced by :func:`~yobx.xtracing.parse.parse_sql`
    (which do not carry dtype information) use :func:`sql_to_onnx` instead.

    :param pq: a :class:`~yobx.sql.parse.ParsedQuery` produced by
        :func:`~yobx.xtracing.dataframe_trace.trace_dataframe`, or a **list**
        of such queries when the traced function returns multiple dataframes.
        All queries in the list are compiled into a single ONNX graph whose
        shared inputs are de-duplicated and whose outputs are the concatenation
        of the outputs from each individual query.
    :param target_opset: ONNX opset version to target.
    :param custom_functions: optional mapping from function name to Python
        callable.  Each callable is traced via
        :func:`~yobx.xtracing.trace_numpy_function`.
    :param builder_cls: graph-builder class or factory callable.
    :param filename: if set, the exported ONNX model is saved to this path and
        the :class:`~yobx.container.ExportReport` is written as a companion
        Excel file (same base name with ``.xlsx`` extension).
    :param verbose: verbosity level (0 = silent).
    :param large_model: if True the returned :class:`~yobx.container.ExportArtifact`
        has its :attr:`~yobx.container.ExportArtifact.container` attribute set to
        an :class:`~yobx.container.ExtendedModelContainer`
    :param external_threshold: if ``large_model`` is True, every tensor whose
        element count exceeds this threshold is stored as external data
    :param return_optimize_report: if True, the returned
        :class:`~yobx.container.ExportArtifact` has its
        :attr:`~yobx.container.ExportArtifact.report` attribute populated with
        per-pattern optimization statistics
    :return: :class:`~yobx.container.ExportArtifact` wrapping the exported
        ONNX model together with an :class:`~yobx.container.ExportReport`.

    Example — single output::

        import numpy as np
        from yobx.xtracing.dataframe_trace import trace_dataframe
        from yobx.sql.sql_convert import parsed_query_to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        def transform(df):
            return df.select([(df["a"] + df["b"]).alias("total")])

        pq = trace_dataframe(transform, {"a": np.float32, "b": np.float32})
        artifact = parsed_query_to_onnx(pq)

        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0,  5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})

    Example — multiple outputs::

        import numpy as np
        from yobx.xtracing.dataframe_trace import trace_dataframe
        from yobx.sql.sql_convert import parsed_query_to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        def transform(df):
            out1 = df.select([(df["a"] + df["b"]).alias("sum_ab")])
            out2 = df.select([(df["a"] - df["b"]).alias("diff_ab")])
            return out1, out2

        pqs = trace_dataframe(transform, {"a": np.float32, "b": np.float32})
        artifact = parsed_query_to_onnx(pqs)

        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        sum_ab, diff_ab = ref.run(None, {"a": a, "b": b})
    """
    g = builder_cls(target_opset, ir_version=10)
    sts = {"custom_functions": custom_functions or {}}
    if isinstance(pq, list):
        if not pq:
            raise ValueError(
                "parsed_query_to_onnx: the list of ParsedQuery objects must not be empty"
            )
        for single_pq in pq:
            parsed_query_to_onnx_graph(g, sts, [], single_pq)
    else:
        parsed_query_to_onnx_graph(g, sts, [], pq)
    if isinstance(g, GraphBuilder):
        artifact = g.to_onnx(
            large_model=large_model,
            external_threshold=external_threshold,
            return_optimize_report=return_optimize_report,
        )
    else:
        artifact = g.to_onnx(large_model=large_model, external_threshold=external_threshold)
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
        unique_vals, first_indices, inverse_indices, _ = g.op.Unique(  # type: ignore
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
        unique_rows, first_indices, inverse_indices, _ = g.op.Unique(  # type: ignore
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


def _build_pivot_table_tensors(
    g: GraphBuilderExtendedProtocol,
    pivot_op: "PivotTableOp",
    col_map: Dict[str, str],
    desired_outputs: Optional[List[str]] = None,
) -> List[str]:
    """Build ONNX tensors for a :class:`~yobx.xtracing.parse.PivotTableOp`.

    Produces one output tensor for each index column (the sorted unique row-key
    values) and one output tensor per *(values_col, column_values entry)* pair.

    Algorithm
    ---------
    For each known column value *cv* (and for each values column):

    1. Create a boolean mask ``cat == cv`` (for single category column) or
       ``(cat1 == cv[0]) AND (cat2 == cv[1]) AND …`` (for multiple category
       columns).
    2. ``Compress`` the *values* column and the group-index vector to keep only
       matching rows.
    3. Aggregate the matching values into a ``(n_groups,)`` output tensor via
       ``ScatterElements`` with the appropriate reduction.

    For ``min`` / ``max`` the output is initialised with the first matching value
    per group (so that the aggregation seed is never a non-matching value), then
    ``ScatterElements(reduction='min'/'max')`` is applied over all matching rows.
    Groups that have no matching rows keep the *fill_value*.

    :param g: graph builder.
    :param pivot_op: the operation descriptor.
    :param col_map: current column-name → ONNX tensor-name mapping.
    :param desired_outputs: optional list of preferred output tensor names.
    :return: list of output tensor names (index column(s) first, then one per
        *(values_col, column_values entry)* pair).
    """
    index_cols = pivot_op.index
    col_col = pivot_op.columns
    val_col = pivot_op.values
    aggfunc = pivot_op.aggfunc
    column_values = pivot_op.column_values
    fill_value = pivot_op.fill_value

    # Normalise to lists so the rest of the function is uniform.
    col_cols: List[str] = [col_col] if isinstance(col_col, str) else list(col_col)
    val_cols: List[str] = [val_col] if isinstance(val_col, str) else list(val_col)

    if aggfunc not in ("sum", "mean", "min", "max", "count"):
        raise ValueError(
            f"pivot_table: unsupported aggfunc {aggfunc!r}. "
            "Choose from 'sum', 'mean', 'min', 'max', 'count'."
        )

    # ------------------------------------------------------------------
    # Step 1 — Unique groups for the index column(s) (row keys)
    # ------------------------------------------------------------------
    if len(index_cols) == 1:
        idx_tensor = col_map[index_cols[0]]
        unique_idx, _first_idx, inv_idx, _ = g.op.Unique(  # type: ignore
            idx_tensor, sorted=1, outputs=4, name="pt_unique"
        )
        n_groups = g.op.Gather(
            g.op.Shape(unique_idx, name="pt_shape"),
            np.array(0, dtype=np.int64),
            axis=0,
            name="pt_n_groups",
        )
        unique_rows = None
    else:
        # Multi-column index: same technique as multi-column GROUP BY.
        idx_tensors = [col_map[c] for c in index_cols]
        cast_idx = [
            g.op.Cast(t, to=TensorProto.DOUBLE, name=f"pt_cast_{c}")  # type: ignore[misc]
            for t, c in zip(idx_tensors, index_cols)
        ]
        unsqueezed_idx = [
            g.op.Unsqueeze(t, np.array([1], dtype=np.int64), name=f"pt_unsq_{c}")  # type: ignore[misc]
            for t, c in zip(cast_idx, index_cols)
        ]
        stacked_idx = g.op.Concat(*unsqueezed_idx, axis=1, name="pt_stack")  # type: ignore[misc]
        unique_rows, _first_idx, inv_idx, _ = g.op.Unique(  # type: ignore
            stacked_idx, axis=0, sorted=1, outputs=4, name="pt_unique"
        )
        n_groups = g.op.Gather(
            g.op.Shape(unique_rows, name="pt_shape"),
            np.array(0, dtype=np.int64),
            axis=0,
            name="pt_n_groups",
        )

    n_groups_1d = g.op.Unsqueeze(n_groups, np.array([0], dtype=np.int64), name="pt_n_groups_1d")

    cat_tensors = [col_map[c] for c in col_cols]

    output_names: List[str] = []
    used_tensor_names: Set[str] = set()
    out_idx = 0

    # ------------------------------------------------------------------
    # Emit index-column output(s) — unique sorted row-key values
    # ------------------------------------------------------------------
    if len(index_cols) == 1:
        if desired_outputs and out_idx < len(desired_outputs):
            candidate = desired_outputs[out_idx]
        else:
            candidate = index_cols[0]
        if g.has_name(candidate) or candidate in used_tensor_names:
            candidate = f"output_{out_idx}"
        g.op.Identity(unique_idx, outputs=[candidate], name=f"pt_out_{index_cols[0]}")  # type: ignore
        output_names.append(candidate)
        used_tensor_names.add(candidate)
        out_idx += 1
    else:
        assert unique_rows is not None, "unique_rows must not be None"
        for j, idx_col in enumerate(index_cols):
            idx_col_unique_f64 = g.op.Gather(  # type: ignore[misc]
                unique_rows, np.array(j, dtype=np.int64), axis=1, name=f"pt_col_{idx_col}_f64"
            )
            idx_col_unique = g.op.CastLike(  # type: ignore[misc]
                idx_col_unique_f64, col_map[idx_col], name=f"pt_col_{idx_col}"
            )
            if desired_outputs and out_idx < len(desired_outputs):
                candidate = desired_outputs[out_idx]
            else:
                candidate = idx_col
            if g.has_name(candidate) or candidate in used_tensor_names:
                candidate = f"output_{out_idx}"
            g.op.Identity(idx_col_unique, outputs=[candidate], name=f"pt_out_{idx_col}")  # type: ignore
            output_names.append(candidate)
            used_tensor_names.add(candidate)
            out_idx += 1

    # ------------------------------------------------------------------
    # Emit one aggregated-value column per (values_col, column_value) pair
    # ------------------------------------------------------------------
    for vi, vc in enumerate(val_cols):
        val_tensor = col_map[vc]
        for j, cv in enumerate(column_values):
            # Build output column name — handle tuple cv for multi-col categories.
            if isinstance(cv, (tuple, list)):
                cv_str = "_".join(str(v) for v in cv)
            else:
                cv_str = str(cv)
            out_col_name = f"{vc}_{cv_str}"
            slot = f"v{vi}_j{j}"

            if desired_outputs and out_idx < len(desired_outputs):
                candidate = desired_outputs[out_idx]
            else:
                candidate = out_col_name
            if g.has_name(candidate) or candidate in used_tensor_names:
                candidate = f"output_{out_idx}"

            # -- Mask for rows where cat == cv (or AND of equalities for multi-col) --
            def _cv_cst(col_idx: int, scalar_cv: object, _slot: str = slot) -> str:
                if isinstance(scalar_cv, str):
                    arr = np.array([scalar_cv], dtype=object)
                elif isinstance(scalar_cv, (float, np.floating)):
                    arr = np.array([scalar_cv], dtype=np.float32)
                elif isinstance(scalar_cv, (int, np.integer)):
                    arr = np.array([scalar_cv], dtype=np.int64)
                else:
                    arr = np.array([scalar_cv])
                return g.make_initializer(f"pt_cv_{_slot}_{col_idx}", arr, give_unique_name=True)

            if len(col_cols) == 1:
                cv_cst_name = _cv_cst(0, cv)
                mask = g.op.Equal(cat_tensors[0], cv_cst_name, name=f"pt_mask_{slot}")  # type: ignore[misc]
            else:
                # Multi-column category: cv must be a tuple/list, one value per col.
                cv_seq = list(cv) if isinstance(cv, (tuple, list)) else [cv]
                partial_masks = [
                    g.op.Equal(  # type: ignore[misc]
                        cat_tensors[k], _cv_cst(k, cv_seq[k]), name=f"pt_mask_{slot}_c{k}"
                    )
                    for k in range(len(col_cols))
                ]
                mask = partial_masks[0]
                for m in partial_masks[1:]:
                    mask = g.op.And(mask, m, name=f"pt_and_{slot}")  # type: ignore[misc]

            # Compress: keep only matching rows
            matching_vals = g.op.Compress(val_tensor, mask, axis=0, name=f"pt_match_v_{slot}")  # type: ignore[misc]
            matching_inv_idx = g.op.Compress(inv_idx, mask, axis=0, name=f"pt_match_idx_{slot}")  # type: ignore[misc]

            # Count of matching rows per group (int64)
            n_match = g.op.Gather(
                g.op.Shape(matching_inv_idx, name=f"pt_match_sz_{slot}"),
                np.array(0, dtype=np.int64),
                axis=0,
                name=f"pt_n_match_{slot}",
            )
            n_match_1d = g.op.Unsqueeze(
                n_match, np.array([0], dtype=np.int64), name=f"pt_n_match_1d_{slot}"
            )
            zeros_i64 = g.op.ConstantOfShape(
                n_groups_1d,
                value=onh.from_array(np.array([0], dtype=np.int64)),
                name=f"pt_zi64_{slot}",
            )
            ones_match = g.op.ConstantOfShape(
                n_match_1d,
                value=onh.from_array(np.array([1], dtype=np.int64)),
                name=f"pt_ones_{slot}",
            )
            cnt = g.op.ScatterElements(  # type: ignore[misc]
                zeros_i64,
                matching_inv_idx,
                ones_match,
                reduction="add",
                axis=0,
                name=f"pt_cnt_{slot}",
            )
            has_data = g.op.Greater(  # type: ignore[misc]
                cnt,
                g.make_initializer(
                    f"pt_zero_i64_{slot}", np.array([0], dtype=np.int64), give_unique_name=True
                ),
                name=f"pt_has_{slot}",
            )

            if aggfunc == "count":
                result = cnt
            else:
                # Zero-filled buffer of shape (n_groups,) with val_tensor's dtype
                zeros_val = g.op.CastLike(  # type: ignore[misc]
                    g.op.ConstantOfShape(n_groups_1d, name=f"pt_zval_{slot}"),
                    val_tensor,
                    name=f"pt_zval_c_{slot}",
                )

                if aggfunc == "sum":
                    sum_result = g.op.ScatterElements(  # type: ignore[misc]
                        zeros_val,
                        matching_inv_idx,
                        matching_vals,
                        reduction="add",
                        axis=0,
                        name=f"pt_sum_{slot}",
                    )
                    if fill_value != 0.0:
                        fv_cast = g.op.CastLike(  # type: ignore[misc]
                            g.make_initializer(
                                f"pt_fv_{slot}",
                                np.array([fill_value], dtype=np.float32),
                                give_unique_name=True,
                            ),
                            sum_result,
                            name=f"pt_fv_c_{slot}",
                        )
                        result = g.op.Where(has_data, sum_result, fv_cast, name=f"pt_r_{slot}")  # type: ignore[misc]
                    else:
                        result = sum_result

                elif aggfunc == "mean":
                    sum_vals = g.op.ScatterElements(  # type: ignore[misc]
                        zeros_val,
                        matching_inv_idx,
                        matching_vals,
                        reduction="add",
                        axis=0,
                        name=f"pt_sum_{slot}",
                    )
                    # Use ones as safe denominator for empty groups
                    # (replaced by fill_value anyway)
                    ones_f = g.op.CastLike(  # type: ignore[misc]
                        g.op.ConstantOfShape(
                            n_groups_1d,
                            value=onh.from_array(np.array([1.0], dtype=np.float32)),
                            name=f"pt_ones_f_{slot}",
                        ),
                        sum_vals,
                        name=f"pt_ones_f_c_{slot}",
                    )
                    cnt_f = g.op.CastLike(cnt, sum_vals, name=f"pt_cnt_f_{slot}")  # type: ignore[misc]
                    safe_cnt = g.op.Where(has_data, cnt_f, ones_f, name=f"pt_safe_cnt_{slot}")  # type: ignore[misc]
                    mean_result = g.op.Div(sum_vals, safe_cnt, name=f"pt_mean_{slot}")  # type: ignore[misc]
                    fv_cast = g.op.CastLike(  # type: ignore[misc]
                        g.make_initializer(
                            f"pt_fv_{slot}",
                            np.array([fill_value], dtype=np.float32),
                            give_unique_name=True,
                        ),
                        mean_result,
                        name=f"pt_fv_c_{slot}",
                    )
                    result = g.op.Where(has_data, mean_result, fv_cast, name=f"pt_r_{slot}")  # type: ignore[misc]

                elif aggfunc in ("min", "max"):
                    # Initialise each group's cell with the first matching value so
                    # that the ScatterElements seed is always a valid observation.
                    # Groups with no matching rows are seeded with fill_value and
                    # will never be overwritten by the scatter.
                    fv_init = g.op.CastLike(  # type: ignore[misc]
                        g.op.ConstantOfShape(
                            n_groups_1d,
                            value=onh.from_array(np.array([fill_value], dtype=np.float32)),
                            name=f"pt_fv_init_{slot}",
                        ),
                        val_tensor,
                        name=f"pt_fv_init_c_{slot}",
                    )
                    # Among matching rows, find the first occurrence of each group.
                    unique_grp, first_in_match, _, _ = g.op.Unique(  # type: ignore
                        matching_inv_idx, sorted=1, outputs=4, name=f"pt_umatch_{slot}"
                    )
                    first_match_vals = g.op.Gather(  # type: ignore[misc]
                        matching_vals, first_in_match, axis=0, name=f"pt_first_mv_{slot}"
                    )
                    # Override fill_value with first matching value for groups that
                    # have at least one matching row.
                    init_seeded = g.op.ScatterElements(  # type: ignore[misc]
                        fv_init, unique_grp, first_match_vals, axis=0, name=f"pt_init_s_{slot}"
                    )
                    reduction_name = "min" if aggfunc == "min" else "max"
                    result = g.op.ScatterElements(  # type: ignore[misc]
                        init_seeded,
                        matching_inv_idx,
                        matching_vals,
                        reduction=reduction_name,
                        axis=0,
                        name=f"pt_{aggfunc}_{slot}",
                    )
                    # Groups with no matching rows still have fill_value — correct.

                else:
                    # Unreachable: the validation at the top of this function covers all cases.
                    raise AssertionError(f"Unexpected aggfunc {aggfunc!r}")

            g.op.Identity(result, outputs=[candidate], name=f"pt_out_{slot}")  # type: ignore
            output_names.append(candidate)
            used_tensor_names.add(candidate)
            out_idx += 1

    return output_names


def _populate_graph(
    g: GraphBuilderExtendedProtocol,
    pq: ParsedQuery,
    input_dtypes: Optional[Dict[str, Union[np.dtype, type, str]]] = None,
    right_input_dtypes: Optional[
        Union[Dict[str, Union[np.dtype, type, str]], List[Dict[str, Union[np.dtype, type, str]]]]
    ] = None,
    custom_functions: Optional[Dict[str, Callable]] = None,
    desired_outputs: Optional[List[str]] = None,
    _finalize: bool = True,
) -> List[str]:
    """
    Populates *g* with ONNX nodes for *pq* and return the output tensor names.

    Type information is read from :attr:`~yobx.xtracing.parse.ColumnRef.dtype`
    when the query was produced by the tracer.  For SQL-string-parsed queries
    (where ``col_ref.dtype == 0``), *input_dtypes* and *right_input_dtypes*
    serve as fallback.

    Left/right classification for JOIN queries follows this priority:

    1. When :attr:`~yobx.xtracing.parse.JoinOp.right_columns` is non-empty
       (tracer-produced join): columns listed there go to the right side.
    2. Otherwise (SQL-string join): columns found in *right_input_dtypes* go
       to the right side, those in *input_dtypes* go to the left side.
       For queries with multiple JOINs, *right_input_dtypes* may be a list
       of dicts (one per JOIN, in order).

    :param _finalize: when ``True`` (default) the SELECT output tensors are
        registered as ONNX model outputs.  Pass ``False`` when processing an
        inner subquery so that its outputs remain intermediate tensors.
    """
    dim = "N"

    # Resolve dtype dicts — used only for SQL-string-parsed queries where
    # col_ref.dtype is 0 for every column.
    left_dtypes: Dict[str, np.dtype] = (
        {k: np.dtype(v) for k, v in input_dtypes.items()} if input_dtypes else {}
    )
    # right_input_dtypes may now be a List[Dict] (one per JOIN) or a single Dict.
    # Build a flat merged dict for backward-compat code paths that still need it,
    # and also store the structured per-join list for the multi-join path.
    if isinstance(right_input_dtypes, (list, tuple)):
        _right_per_join_raw: List[Optional[Dict]] = list(right_input_dtypes)
        right_dtypes: Dict[str, np.dtype] = {}
        for _rd in right_input_dtypes:
            if _rd is not None:
                for _k, _v in _rd.items():
                    right_dtypes[_k] = np.dtype(_v)
    elif right_input_dtypes is not None:
        _right_per_join_raw = [right_input_dtypes]
        right_dtypes = {k: np.dtype(v) for k, v in right_input_dtypes.items()}
    else:
        _right_per_join_raw = []
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
        per_join_right_col_maps: List[Dict[str, str]] = []
    else:
        # Build list of inputs needed for left and right tables
        # (only columns that appear in the query).
        all_cols = pq.columns
        left_inputs: List[Tuple[str, int, Tuple]] = []
        right_inputs: List[Tuple[str, int, Tuple]] = []

        # Collect ALL JoinOps (not just the first).
        join_ops: List[JoinOp] = [op for op in pq.operations if isinstance(op, JoinOp)]
        join_op: Optional[JoinOp] = join_ops[0] if join_ops else None

        # ---------------------------------------------------------------
        # Tracer path — every JoinOp carries explicit right_columns lists.
        # Build per-join rename maps (col → ONNX tensor name) for all joins.
        # ---------------------------------------------------------------
        # combined set of all right-side columns across all JoinOps
        combined_right_col_set: Set[str] = set()
        # per-join: column name → ONNX tensor name (handles clashes)
        per_join_right_col_onnx_name: List[Dict[str, str]] = []

        for jop in join_ops:
            if jop.right_columns:
                # Use the JoinOp's left_columns to determine the accumulated
                # left side at this join step (includes all previously merged cols).
                left_set_at_join: Set[str] = {cr.column for cr in jop.left_columns}
                rcm_i: Dict[str, str] = {}
                for cr in jop.right_columns:
                    col = cr.column
                    combined_right_col_set.add(col)
                    # Suffix right column if its name clashes with an accumulated
                    # left-side column at this join step.
                    rcm_i[col] = f"{col}_right" if col in left_set_at_join else col
                per_join_right_col_onnx_name.append(rcm_i)
            else:
                per_join_right_col_onnx_name.append({})

        # ---------------------------------------------------------------
        # Per-join right dtype lookup for the SQL-string path.
        # _right_per_join_raw is [] (no right_input_dtypes), [single_dict]
        # (backward-compat), or a list with one dict per join.
        # ---------------------------------------------------------------
        def _per_join_right_dtypes(join_idx: int) -> Dict[str, np.dtype]:
            if join_idx < len(_right_per_join_raw):
                rd = _right_per_join_raw[join_idx]
                if rd is None:
                    return left_dtypes
                return {k: np.dtype(v) for k, v in rd.items()}
            # Fallback: if fewer dicts were supplied than JoinOps, use the
            # last-supplied dict (or left_dtypes if none was supplied).
            if _right_per_join_raw:
                last = _right_per_join_raw[-1]
                if last is None:
                    return left_dtypes
                return {k: np.dtype(v) for k, v in last.items()}
            return left_dtypes

        # ---------------------------------------------------------------
        # Input classification: assign each column to left or right side.
        # ---------------------------------------------------------------
        seen_right: List[str] = []

        # For SQL-string path with per-join dtypes: we need to know which
        # columns belong to which join so we can register them separately.
        # Build a mapping: column → join index (first join that owns the col).
        sql_col_to_join_idx: Dict[str, int] = {}
        if not combined_right_col_set:  # SQL-string path
            for ji, jop in enumerate(join_ops):
                jrd = _per_join_right_dtypes(ji)
                for col in jrd:
                    if col not in sql_col_to_join_idx:
                        sql_col_to_join_idx[col] = ji
                # Also register the right keys even if not in the dtypes
                for rk in jop.right_keys:
                    if rk not in sql_col_to_join_idx and rk in jrd:
                        sql_col_to_join_idx[rk] = ji

        for col_ref in all_cols:
            col = col_ref.column
            if combined_right_col_set and col in combined_right_col_set:
                # Tracer-produced JOIN: column is explicitly listed as right-side
                # of at least one JoinOp.  Find its join and use that join's
                # rename map to get the ONNX tensor name.
                onnx_name = col
                for ji, _jop in enumerate(join_ops):
                    if col in per_join_right_col_onnx_name[ji]:
                        onnx_name = per_join_right_col_onnx_name[ji][col]
                        break
                right_inputs.append((onnx_name, col_ref.dtype, (dim,)))
                seen_right.append(col)
                # If this column also exists in the accumulated left frame at
                # the corresponding join step, register it as a left input too.
                for ji, jop in enumerate(join_ops):
                    if col in per_join_right_col_onnx_name[ji]:
                        left_set_at_join = {cr.column for cr in jop.left_columns}
                        if col in left_set_at_join and col_ref.dtype != 0:
                            left_inputs.append((col, col_ref.dtype, (dim,)))
                        break
            elif col_ref.dtype != 0:
                # Tracer-produced query (no join, or left-side join column).
                left_inputs.append((col, col_ref.dtype, (dim,)))
            elif col in right_dtypes and join_op is not None:
                # SQL-string JOIN: column found in right_input_dtypes.
                right_inputs.append((col, _np_dtype_to_onnx(right_dtypes[col]), (dim,)))
                seen_right.append(col)
            elif col in left_dtypes:
                # SQL-string: column found in input_dtypes.
                left_inputs.append((col, _np_dtype_to_onnx(left_dtypes[col]), (dim,)))

        # ---------------------------------------------------------------
        # SQL-string path: ensure key columns are registered on the correct
        # side for EVERY JoinOp, not just the first.
        # ---------------------------------------------------------------
        if join_ops and not combined_right_col_set:
            for ji, jop in enumerate(join_ops):
                jrd = _per_join_right_dtypes(ji)
                # Make sure all right key columns are registered on the right side.
                for rk in jop.right_keys:
                    if rk not in seen_right and rk in jrd:
                        right_inputs.append((rk, _np_dtype_to_onnx(jrd[rk]), (dim,)))
                        seen_right.append(rk)

            # Make sure all left key columns for ALL JoinOps are in left_inputs.
            left_input_names = {name for name, _, _ in left_inputs}
            for jop in join_ops:
                for lk in jop.left_keys:
                    if lk not in left_input_names and lk in left_dtypes:
                        left_inputs.append((lk, _np_dtype_to_onnx(left_dtypes[lk]), (dim,)))
                        left_input_names.add(lk)

            # Rename right inputs that conflict with any left input.
            # Each join might introduce its own right columns; we rename
            # per-join to keep all inputs uniquely named.
            left_input_names = {name for name, _, _ in left_inputs}

            # Build per-join rename maps for the SQL-string path.
            # We track which columns belong to which join (from sql_col_to_join_idx)
            # and ensure that clashing names get per-join suffixes.
            # Strategy: within each join, collect its right column names, check
            # for clashes with the accumulated left_input_names (which grows after
            # each join as newly added right columns move to the left side), and
            # assign rename entries.
            sql_per_join_rename: List[Dict[str, str]] = [{} for _ in join_ops]
            new_right_inputs: List[Tuple[str, int, Tuple]] = []

            # For each right input, determine which join it belongs to.
            for onnx_name, tp, shape in right_inputs:
                join_idx_for_col = sql_col_to_join_idx.get(onnx_name, 0)
                if onnx_name in left_input_names:
                    new_name = f"{onnx_name}_right"
                    sql_per_join_rename[join_idx_for_col][onnx_name] = new_name
                    new_right_inputs.append((new_name, tp, shape))
                else:
                    new_right_inputs.append((onnx_name, tp, shape))
            right_inputs = new_right_inputs
        else:
            sql_per_join_rename = []

        all_inputs = left_inputs + right_inputs

        # Add inputs to the graph only when not already registered.
        for inp_name, inp_type, inp_shape in all_inputs:
            if not g.has_name(inp_name):
                g.make_tensor_input(inp_name, inp_type, inp_shape)

        # col_map: current ONNX tensor name for each column (left side)
        col_map = {name: name for name, _, _ in left_inputs}

        # ---------------------------------------------------------------
        # Build per-join right_col_maps.
        # per_join_right_col_maps[i] maps right-column name → ONNX tensor
        # name for the i-th JoinOp.
        # ---------------------------------------------------------------
        per_join_right_col_maps = []

        if combined_right_col_set:
            # Tracer path: use per_join_right_col_onnx_name (already computed).
            for ji, jop in enumerate(join_ops):
                rcm = {}
                rcm_onnx = per_join_right_col_onnx_name[ji]
                for cr in jop.right_columns:
                    col = cr.column
                    onnx_name = rcm_onnx.get(col, col)
                    if onnx_name in {n for n, _, _ in right_inputs}:
                        rcm[col] = onnx_name
                # Also include right-key columns if not already present.
                right_input_names = {n for n, _, _ in right_inputs}
                for rk in jop.right_keys:
                    if rk not in rcm and rk in right_input_names:
                        rcm[rk] = rcm_onnx.get(rk, rk)
                per_join_right_col_maps.append(rcm)
        elif sql_per_join_rename:
            # SQL-string path with per-join right dtypes or same-name clash.
            for ji, _jop in enumerate(join_ops):
                jrd = _per_join_right_dtypes(ji)
                rename_i = sql_per_join_rename[ji] if ji < len(sql_per_join_rename) else {}
                rcm = {}
                for col in jrd:
                    onnx_name = rename_i.get(col, col)
                    # Only include if it was actually registered as an input.
                    if onnx_name in {n for n, _, _ in right_inputs}:
                        rcm[col] = onnx_name
                per_join_right_col_maps.append(rcm)
        elif join_ops:
            # SQL-string path, single flat right_dtypes, no rename needed.
            # All joins share the same right_col_map (backward-compat).
            single_rcm = {name: name for name, _, _ in right_inputs}
            for _ in join_ops:
                per_join_right_col_maps.append(single_rcm)

        # For backward-compat non-join converters (FilterOp etc.), keep a
        # global right_col_map pointing to the first join's map (if any).
        right_col_map = per_join_right_col_maps[0] if per_join_right_col_maps else {}

    select_op: Optional[SelectOp] = None
    _group_op: Optional[GroupByOp] = None
    _pivot_op: Optional[PivotTableOp] = None

    _join_idx = 0  # tracks which JoinOp we are processing in the loop below
    for op in pq.operations:
        converter = get_sql_op_converter(type(op))
        if converter is not None:
            sts_ctx = {"custom_functions": resolved_custom_functions}
            if isinstance(op, JoinOp):
                # Use the per-join right_col_map for this specific join.
                rcm = (
                    per_join_right_col_maps[_join_idx]
                    if per_join_right_col_maps and _join_idx < len(per_join_right_col_maps)
                    else right_col_map
                )
                col_map = converter(g, sts_ctx, list(col_map.keys()), op, col_map, rcm)
                _join_idx += 1
            else:
                col_map = converter(g, sts_ctx, list(col_map.keys()), op, col_map, right_col_map)
        elif isinstance(op, GroupByOp):
            _group_op = op  # retained for SelectOp aggregation
        elif isinstance(op, SelectOp):
            select_op = op
        elif isinstance(op, PivotTableOp):
            _pivot_op = op

    # PivotTableOp is a terminal operation: it produces its own outputs,
    # replacing the GroupBy + Select pattern.
    if _pivot_op is not None:
        output_names = _build_pivot_table_tensors(g, _pivot_op, col_map, desired_outputs)
        if _finalize:
            for out_name in output_names:
                g.make_tensor_output(out_name, indexed=False, allow_untyped_output=True)
        return output_names

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
    output_names = []
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
