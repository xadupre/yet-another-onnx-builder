"""
DataFrame function tracer — convert a Python function operating on a
:class:`TracedDataFrame` into an ONNX model via
:class:`~yobx.sql.parse.ParsedQuery`.

The tracer provides a lightweight pandas-inspired API.  When you call
:func:`dataframe_to_onnx` (or the lower-level :func:`trace_dataframe`), a
:class:`TracedDataFrame` is created from the *input_dtypes* mapping and passed
to your function.  Every operation performed on the frame — column access,
arithmetic, filtering, aggregation — is *recorded* as an AST node rather than
being executed.  After the function returns, the accumulated nodes are assembled
into a :class:`~yobx.sql.parse.ParsedQuery` that is compiled to ONNX by the
existing SQL converter.

Supported operations
--------------------
* **Column access**: ``df["col"]`` or ``df.col``
* **Arithmetic**: ``+``, ``-``, ``*``, ``/`` between columns and/or scalars
* **Comparison**: ``>``, ``<``, ``>=``, ``<=``, ``==``, ``!=``
* **Boolean combination**: ``&`` (AND), ``|`` (OR) on :class:`TracedCondition`
* **Row filtering**: :meth:`TracedDataFrame.filter` or ``df[condition]``
* **Column projection**: :meth:`TracedDataFrame.select`
* **New columns**: :meth:`TracedDataFrame.assign`
* **Aggregation**: ``.sum()``, ``.mean()``, ``.min()``, ``.max()``,
  ``.count()`` on a :class:`TracedSeries`
* **Column alias**: ``.alias(name)`` on a :class:`TracedSeries`
* **Group-by**: :meth:`TracedDataFrame.groupby` + :meth:`TracedGroupBy.agg`
* **Function chaining**: :meth:`TracedDataFrame.pipe`

Example
-------
::

    import numpy as np
    from yobx.sql.dataframe_trace import dataframe_to_onnx
    from yobx.reference import ExtendedReferenceEvaluator

    def transform(df):
        df = df.filter(df["a"] > 0)
        return df.select([(df["a"] + df["b"]).alias("total")])

    dtypes = {"a": np.float32, "b": np.float32}
    artifact = dataframe_to_onnx(transform, dtypes)

    ref = ExtendedReferenceEvaluator(artifact)
    a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    b = np.array([4.0,  5.0, 6.0], dtype=np.float32)
    (total,) = ref.run(None, {"a": a, "b": b})
    # total == array([5., 9.], dtype=float32)  (rows where a > 0)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import numpy as np

from .. import DEFAULT_TARGET_OPSET
from ..container import ExportArtifact
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
    _collect_columns,
)
from .sql_convert import parsed_queries_to_onnx, parsed_query_to_onnx


def _to_ast(value: object) -> object:
    """Convert a Python scalar or :class:`TracedSeries` to an AST expression.

    :param value: a :class:`TracedSeries`, :class:`bool`, :class:`int`, or
        :class:`float`.
    :return: the corresponding AST expression node.
    :raises TypeError: if *value* cannot be converted.
    """
    if isinstance(value, TracedSeries):
        return value._expr
    if isinstance(value, bool):
        return Literal(bool(value))
    if isinstance(value, int):
        return Literal(int(value))
    if isinstance(value, float):
        return Literal(float(value))
    raise TypeError(
        f"Cannot convert {type(value).__name__!r} to an AST expression. "
        "Expected a TracedSeries, int, or float."
    )


class TracedSeries:
    """Proxy for a single column or computed expression in a traced DataFrame.

    Arithmetic and comparison operators are overloaded to return new
    :class:`TracedSeries` or :class:`TracedCondition` objects that accumulate
    the operation graph without executing any actual computation.

    :param expr: the underlying AST expression node (a
        :class:`~yobx.sql.parse.ColumnRef`,
        :class:`~yobx.sql.parse.BinaryExpr`,
        :class:`~yobx.sql.parse.AggExpr`, etc.).
    :param alias: optional output alias for this series.
    """

    def __init__(self, expr: object, alias: Optional[str] = None) -> None:
        self._expr = expr
        self._alias = alias

    def alias(self, name: str) -> TracedSeries:
        """Return a copy tagged with *name* as the output alias.

        :param name: the alias to assign.
        :return: a new :class:`TracedSeries` carrying the alias.
        """
        return TracedSeries(self._expr, alias=name)

    def __add__(self, other: object) -> TracedSeries:
        return TracedSeries(BinaryExpr(self._expr, "+", _to_ast(other)))

    def __radd__(self, other: object) -> TracedSeries:
        return TracedSeries(BinaryExpr(_to_ast(other), "+", self._expr))

    def __sub__(self, other: object) -> TracedSeries:
        return TracedSeries(BinaryExpr(self._expr, "-", _to_ast(other)))

    def __rsub__(self, other: object) -> TracedSeries:
        return TracedSeries(BinaryExpr(_to_ast(other), "-", self._expr))

    def __mul__(self, other: object) -> TracedSeries:
        return TracedSeries(BinaryExpr(self._expr, "*", _to_ast(other)))

    def __rmul__(self, other: object) -> TracedSeries:
        return TracedSeries(BinaryExpr(_to_ast(other), "*", self._expr))

    def __truediv__(self, other: object) -> TracedSeries:
        return TracedSeries(BinaryExpr(self._expr, "/", _to_ast(other)))

    def __rtruediv__(self, other: object) -> TracedSeries:
        return TracedSeries(BinaryExpr(_to_ast(other), "/", self._expr))

    def __gt__(self, other: object) -> TracedCondition:  # type: ignore[override]
        return TracedCondition(Condition(self._expr, ">", _to_ast(other)))

    def __lt__(self, other: object) -> TracedCondition:  # type: ignore[override]
        return TracedCondition(Condition(self._expr, "<", _to_ast(other)))

    def __ge__(self, other: object) -> TracedCondition:  # type: ignore[override]
        return TracedCondition(Condition(self._expr, ">=", _to_ast(other)))

    def __le__(self, other: object) -> TracedCondition:  # type: ignore[override]
        return TracedCondition(Condition(self._expr, "<=", _to_ast(other)))

    def __eq__(self, other: object) -> TracedCondition:  # type: ignore[override]
        return TracedCondition(Condition(self._expr, "=", _to_ast(other)))

    def __ne__(self, other: object) -> TracedCondition:  # type: ignore[override]
        return TracedCondition(Condition(self._expr, "<>", _to_ast(other)))

    def sum(self) -> TracedSeries:
        """Record a ``SUM(...)`` aggregation."""
        return TracedSeries(AggExpr("sum", self._expr))

    def mean(self) -> TracedSeries:
        """Record an ``AVG(...)`` aggregation."""
        return TracedSeries(AggExpr("avg", self._expr))

    def min(self) -> TracedSeries:
        """Record a ``MIN(...)`` aggregation."""
        return TracedSeries(AggExpr("min", self._expr))

    def max(self) -> TracedSeries:
        """Record a ``MAX(...)`` aggregation."""
        return TracedSeries(AggExpr("max", self._expr))

    def count(self) -> TracedSeries:
        """Record a ``COUNT(...)`` aggregation."""
        return TracedSeries(AggExpr("count", self._expr))

    def to_select_item(self) -> SelectItem:
        """Convert to a :class:`~yobx.sql.parse.SelectItem` for use in a query."""
        return SelectItem(expr=self._expr, alias=self._alias)

    def __repr__(self) -> str:
        if self._alias:
            return f"TracedSeries({self._expr!r} AS {self._alias!r})"
        return f"TracedSeries({self._expr!r})"

    # __eq__ is overridden to return TracedCondition, so Python would set
    # __hash__ = None making TracedSeries unhashable.  Restore identity-based
    # hashing so that instances can be used in sets / as dict keys.
    __hash__ = object.__hash__  # type: ignore[assignment]


class TracedCondition:
    """Proxy for a boolean predicate produced by comparing :class:`TracedSeries`.

    Use ``&`` and ``|`` to combine conditions; Python's ``and``/``or`` keywords
    cannot be used because they force boolean evaluation of the operands.

    :param condition: the underlying :class:`~yobx.sql.parse.Condition` AST node.
    """

    def __init__(self, condition: Condition) -> None:
        self._condition = condition

    def __and__(self, other: TracedCondition) -> TracedCondition:
        """Combine with AND (``&``)."""
        return TracedCondition(Condition(self._condition, "and", other._condition))

    def __or__(self, other: TracedCondition) -> TracedCondition:
        """Combine with OR (``|``)."""
        return TracedCondition(Condition(self._condition, "or", other._condition))

    def __bool__(self) -> bool:
        raise TypeError(
            "TracedCondition cannot be evaluated as a Python boolean. "
            "Use & (AND) and | (OR) to combine conditions."
        )

    def __repr__(self) -> str:
        return f"TracedCondition({self._condition!r})"


class TracedGroupBy:
    """Result of :meth:`TracedDataFrame.groupby`.

    Call :meth:`agg` to specify the aggregation expressions and obtain a new
    :class:`TracedDataFrame` representing the grouped result.

    :param df: the source :class:`TracedDataFrame`.
    :param by: list of column names to group by.
    """

    def __init__(self, df: TracedDataFrame, by: List[str]) -> None:
        self._df = df
        self._by = by

    def agg(self, exprs: Union[List[TracedSeries], Dict[str, TracedSeries]]) -> TracedDataFrame:
        """Record aggregation expressions and return a new :class:`TracedDataFrame`.

        :param exprs: either a list of :class:`TracedSeries` (each should carry
            an alias via ``.alias(name)``), or a ``{alias: TracedSeries}`` dict.
        :return: a new :class:`TracedDataFrame` with ``GroupByOp`` + ``SelectOp``
            recorded.
        """
        if isinstance(exprs, dict):
            series_list: List[TracedSeries] = [v.alias(k) for k, v in exprs.items()]
        else:
            series_list = list(exprs)

        new_ops = list(self._df._ops)
        new_ops.append(GroupByOp(columns=list(self._by)))
        select_items = [s.to_select_item() for s in series_list]
        new_ops.append(SelectOp(items=select_items))

        new_cols = {
            item.output_name(): TracedSeries(item.expr, item.alias) for item in select_items
        }
        return TracedDataFrame(new_cols, new_ops, list(self._df._source_columns))


class TracedDataFrame:
    """Proxy for a DataFrame that records SQL-like operations for ONNX export.

    Create a :class:`TracedDataFrame` via :func:`trace_dataframe` (recommended)
    or construct it directly with a column-name → dtype mapping.  Apply
    operations (``filter``, ``select``, arithmetic on columns) to build up the
    computation graph.  Convert to an ONNX model via :func:`dataframe_to_onnx`.

    :param columns: mapping from column name to :class:`TracedSeries`.
    :param ops: accumulated list of :class:`~yobx.sql.parse.SqlOperation`
        objects (usually starts empty).
    :param source_columns: ordered list of original source column names (those
        that will become ONNX inputs).  Defaults to ``list(columns.keys())``.
    """

    def __init__(
        self,
        columns: Dict[str, TracedSeries],
        ops: Optional[List] = None,
        source_columns: Optional[List[str]] = None,
    ) -> None:
        self._columns: Dict[str, TracedSeries] = columns
        self._ops: List = list(ops) if ops is not None else []
        self._source_columns: List[str] = (
            list(source_columns) if source_columns is not None else list(columns.keys())
        )

    # ------------------------------------------------------------------
    # Column access
    # ------------------------------------------------------------------

    def __getitem__(
        self, key: Union[str, List[str], TracedCondition]
    ) -> Union[TracedSeries, TracedDataFrame]:
        if isinstance(key, str):
            if key not in self._columns:
                raise KeyError(f"Column {key!r} not found in traced DataFrame")
            return self._columns[key]
        if isinstance(key, list):
            return self.select(key)  # type: ignore
        if isinstance(key, TracedCondition):
            return self.filter(key)
        raise TypeError(
            f"Unsupported key type: {type(key).__name__!r}. "
            "Expected a column name (str), a list of names, or a TracedCondition."
        )

    def __getattr__(self, name: str) -> TracedSeries:
        if name.startswith("_"):
            raise AttributeError(name)
        cols = object.__getattribute__(self, "_columns")
        if name in cols:
            return cols[name]
        raise AttributeError(
            f"TracedDataFrame has no column {name!r}. Available columns: {list(cols.keys())}"
        )

    @property
    def columns(self) -> List[str]:
        """Return the ordered list of column names in the current frame."""
        return list(self._columns.keys())

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def filter(self, condition: TracedCondition) -> TracedDataFrame:
        """Record a row-filter (WHERE) operation.

        :param condition: a :class:`TracedCondition` produced by comparing
            columns, e.g. ``df["a"] > 0``.
        :return: a new :class:`TracedDataFrame` with the filter recorded.
        """
        new_ops = [*self._ops, FilterOp(condition=condition._condition)]
        return TracedDataFrame(dict(self._columns), new_ops, list(self._source_columns))

    def select(
        self, exprs: Union[List[Union[str, TracedSeries]], Dict[str, TracedSeries]]
    ) -> TracedDataFrame:
        """Record a column-projection (SELECT) operation.

        :param exprs: one of:

            * a list of column-name strings — passes columns through unchanged;
            * a list of :class:`TracedSeries` — use ``.alias(name)`` to name
              each output;
            * a ``{alias: TracedSeries}`` dict — explicit alias → expression
              mapping.

        :return: a new :class:`TracedDataFrame` with the projection recorded.
        """
        if isinstance(exprs, dict):
            series_list: List[TracedSeries] = [v.alias(k) for k, v in exprs.items()]
        elif isinstance(exprs, list):
            series_list = []
            for e in exprs:
                if isinstance(e, str):
                    if e not in self._columns:
                        raise KeyError(f"Column {e!r} not found in traced DataFrame")
                    series_list.append(self._columns[e])
                elif isinstance(e, TracedSeries):
                    series_list.append(e)
                else:
                    raise TypeError(
                        f"select() expects strings or TracedSeries, got {type(e).__name__!r}"
                    )
        else:
            raise TypeError(f"select() expects a list or dict, not {type(exprs).__name__!r}")

        select_items = [s.to_select_item() for s in series_list]
        new_ops = [*self._ops, SelectOp(items=select_items)]
        new_cols = {
            item.output_name(): TracedSeries(item.expr, item.alias) for item in select_items
        }
        return TracedDataFrame(new_cols, new_ops, list(self._source_columns))

    def assign(self, **kwargs: TracedSeries) -> TracedDataFrame:
        """Record new computed columns without discarding existing ones.

        Each keyword argument names a new (or overwritten) column; the value
        must be a :class:`TracedSeries`.

        :return: a new :class:`TracedDataFrame` with the extra columns present.
        """
        new_cols = dict(self._columns)
        for name, expr in kwargs.items():
            if not isinstance(expr, TracedSeries):
                raise TypeError(
                    f"assign() expects TracedSeries values, "
                    f"got {type(expr).__name__!r} for {name!r}"
                )
            new_cols[name] = TracedSeries(expr._expr, alias=name)
        return TracedDataFrame(new_cols, list(self._ops), list(self._source_columns))

    def pipe(
        self, func: Callable[..., "TracedDataFrame"], *args: object, **kwargs: object
    ) -> "TracedDataFrame":
        """Apply *func* to this frame (pandas ``pipe`` idiom).

        Equivalent to ``func(self, *args, **kwargs)``.  Useful for chaining
        transformations written as standalone functions::

            def preprocess(df):
                return df.filter(df["a"] > 0)

            def add_feature(df):
                return df.assign(c=(df["a"] + df["b"]).alias("c"))

            def pipeline(df):
                return df.pipe(preprocess).pipe(add_feature)

        :param func: callable that accepts a :class:`TracedDataFrame` as its
            first argument and returns a :class:`TracedDataFrame`.
        :param args: additional positional arguments forwarded to *func*.
        :param kwargs: additional keyword arguments forwarded to *func*.
        :return: the result of ``func(self, *args, **kwargs)``.
        """
        return func(self, *args, **kwargs)

    def join(
        self, right: "TracedDataFrame", left_key: str, right_key: str, join_type: str = "inner"
    ) -> "TracedDataFrame":
        """Record an equi-join with another :class:`TracedDataFrame`.

        :param right: the right-hand :class:`TracedDataFrame` to join with.
        :param left_key: column name from *this* (left) frame used in the join
            predicate.
        :param right_key: column name from *right* frame used in the join
            predicate.
        :param join_type: join type — ``'inner'`` (default), ``'left'``,
            ``'right'``, or ``'full'``.
        :return: a new :class:`TracedDataFrame` containing columns from both
            sides with a :class:`~yobx.sql.parse.JoinOp` recorded.

        Example::

            import numpy as np
            from yobx.sql.dataframe_trace import dataframe_to_onnx

            def transform(df1, df2):
                return df1.join(df2, left_key="cid", right_key="id")

            dtypes1 = {"cid": np.int64, "a": np.float32}
            dtypes2 = {"id": np.int64, "b": np.float32}
            artifact = dataframe_to_onnx(transform, [dtypes1, dtypes2])
        """
        join_op = JoinOp(
            right_table="r", left_key=left_key, right_key=right_key, join_type=join_type
        )
        # Merge columns: left columns take priority on name collision.
        merged_cols = dict(self._columns)
        for name, series in right._columns.items():
            if name not in merged_cols:
                merged_cols[name] = series
        new_ops = [*self._ops, join_op]
        left_source_set = set(self._source_columns)
        all_source_cols = list(self._source_columns) + [
            col for col in right._source_columns if col not in left_source_set
        ]
        return TracedDataFrame(merged_cols, new_ops, all_source_cols)

    def groupby(self, by: Union[str, List[str]]) -> TracedGroupBy:
        """Begin a group-by aggregation.

        :param by: column name or list of column names to group by.
        :return: a :class:`TracedGroupBy` on which ``.agg(...)`` can be called.
        """
        if isinstance(by, str):
            by = [by]
        return TracedGroupBy(self, list(by))

    # ------------------------------------------------------------------
    # Query assembly
    # ------------------------------------------------------------------

    def to_parsed_query(self) -> ParsedQuery:
        """Assemble a :class:`~yobx.sql.parse.ParsedQuery` from the recorded ops.

        If no ``SelectOp`` has been recorded yet (e.g. only a filter was
        applied), a pass-through ``SELECT`` of all current columns is appended
        automatically.

        :param additional_traced_df:
        :return: a :class:`~yobx.sql.parse.ParsedQuery` ready for ONNX
            conversion via :func:`~yobx.sql.sql_convert.parsed_query_to_onnx`.
        """
        ops = list(self._ops)
        if not any(isinstance(op, SelectOp) for op in ops):
            items = [SelectItem(ColumnRef(col), alias=None) for col in self._columns]
            ops.append(SelectOp(items=items))
        columns = _collect_columns(ops)
        return ParsedQuery(operations=ops, from_table="t", columns=columns)

    def to_parsed_queries(self, *additional_traced_df: TracedDataFrame) -> List[ParsedQuery]:
        """Assemble a list of :class:`~yobx.sql.parse.ParsedQuery` objects, one per output frame.

        Each frame gets its own independent full query (all ops preserved) so that
        multiple output paths from the same input can be compiled into a single ONNX
        graph with several output tensors.

        If no ``SelectOp`` has been recorded yet (e.g. only a filter was
        applied), a pass-through ``SELECT`` of all current columns is appended
        automatically.

        :param additional_traced_df: additional output dataframes to export
        :return: list of :class:`~yobx.sql.parse.ParsedQuery` objects, one per
            output frame, ready for ONNX conversion via
            :func:`~yobx.sql.sql_convert.parsed_queries_to_onnx`.
        """

        def _frame_to_query(tdf: TracedDataFrame) -> ParsedQuery:
            ops = list(tdf._ops)
            if not any(isinstance(op, SelectOp) for op in ops):
                items = [SelectItem(ColumnRef(col), alias=None) for col in tdf._columns]
                ops.append(SelectOp(items=items))
            columns = _collect_columns(ops)
            return ParsedQuery(operations=ops, from_table="t", columns=columns)

        return [_frame_to_query(self)] + [_frame_to_query(tdf) for tdf in additional_traced_df]

    def __repr__(self) -> str:
        return f"TracedDataFrame(columns={list(self._columns.keys())!r})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def trace_dataframe(
    func: Callable,
    input_dtypes: Union[
        Dict[str, Union[np.dtype, type, str]], List[Dict[str, Union[np.dtype, type, str]]]
    ],
) -> Union[ParsedQuery, List[ParsedQuery]]:
    """Trace *func* and return the equivalent :class:`~yobx.sql.parse.ParsedQuery`.

    Constructs one or more :class:`TracedDataFrame` objects whose columns
    correspond to *input_dtypes*, calls *func* with them, and converts the
    resulting frame's recorded operations into a
    :class:`~yobx.sql.parse.ParsedQuery`.

    :param func: a callable that accepts one or more :class:`TracedDataFrame`
        objects and returns a :class:`TracedDataFrame` **or** a tuple of
        :class:`TracedDataFrame` objects (for multiple-output models).  The
        function may apply any combination of ``filter``, ``select``,
        arithmetic on columns, ``join``, and aggregations.
    :param input_dtypes: either

        * a single ``{column: dtype}`` mapping — *func* is called with one
          :class:`TracedDataFrame`; or
        * a **list** of ``{column: dtype}`` mappings — *func* is called with
          one :class:`TracedDataFrame` per mapping, in order.

        The dtypes are not used during tracing itself; they are used only when
        the returned query is subsequently compiled to ONNX.
    :return: a single :class:`~yobx.sql.parse.ParsedQuery` when *func* returns
        one frame, or a ``list`` of :class:`~yobx.sql.parse.ParsedQuery` objects
        (one per output frame) when *func* returns a tuple.

    Example — single dataframe::

        import numpy as np
        from yobx.sql.dataframe_trace import trace_dataframe

        def transform(df):
            df = df.filter(df["a"] > 0)
            return df.select([(df["a"] + df["b"]).alias("total")])

        pq = trace_dataframe(transform, {"a": np.float32, "b": np.float32})
        for op in pq.operations:
            print(type(op).__name__, "—", op)
        # FilterOp — FilterOp(condition=Condition(left=ColumnRef(column='a', ...
        # SelectOp — SelectOp(items=[SelectItem(expr=BinaryExpr(...), alias='total')], ...

    Example — two dataframes::

        import numpy as np
        from yobx.sql.dataframe_trace import trace_dataframe

        def transform(df1, df2):
            return df1.select([(df1["a"] + df2["b"]).alias("total")])

        pq = trace_dataframe(transform, [{"a": np.float32}, {"b": np.float32}])
    """
    if isinstance(input_dtypes, list):
        dfs = [
            TracedDataFrame(
                {name: TracedSeries(ColumnRef(name)) for name in d}, source_columns=list(d.keys())
            )
            for d in input_dtypes
        ]
        result = func(*dfs)
    else:
        columns = {name: TracedSeries(ColumnRef(name)) for name in input_dtypes}
        df = TracedDataFrame(columns, source_columns=list(input_dtypes.keys()))
        result = func(df)

    if isinstance(result, tuple):
        if not all(isinstance(r, TracedDataFrame) for r in result):
            raise TypeError(
                f"trace_dataframe: function must return a tuple of TracedDataFrame, "
                f"got {tuple(type(r).__name__ for r in result)}"
            )
        return result[0].to_parsed_queries(*result[1:])

    if not isinstance(result, TracedDataFrame):
        raise TypeError(
            f"trace_dataframe: function must return a TracedDataFrame, "
            f"got {type(result).__name__!r}"
        )
    return result.to_parsed_query()


def dataframe_to_onnx(
    func: Callable,
    input_dtypes: Union[
        Dict[str, Union[np.dtype, type, str]], List[Dict[str, Union[np.dtype, type, str]]]
    ],
    target_opset: int = DEFAULT_TARGET_OPSET,
    custom_functions: Optional[Dict[str, Callable]] = None,
    builder_cls: Union[type, Callable] = GraphBuilder,
    filename: Optional[str] = None,
    verbose: int = 0,
) -> ExportArtifact:
    """Trace *func* and convert the resulting computation to ONNX.

    Combines :func:`trace_dataframe` and
    :func:`~yobx.sql.sql_convert.parsed_query_to_onnx` into a single call.

    :param func: a callable that accepts one or more :class:`TracedDataFrame`
        objects and returns a :class:`TracedDataFrame`.
    :param input_dtypes: either

        * a single ``{column: dtype}`` mapping — *func* is called with one
          :class:`TracedDataFrame`; or
        * a **list** of ``{column: dtype}`` mappings — *func* is called with
          one :class:`TracedDataFrame` per mapping, in order.  When the traced
          function contains a :class:`~yobx.sql.parse.JoinOp`, the first
          mapping is used as the *left-table* dtypes and the second as the
          *right-table* dtypes.  For functions that simply share columns
          across multiple frames without a join, all mappings are merged into
          a single input-dtype dict.

    :param target_opset: ONNX opset version to target (default:
        :data:`yobx.DEFAULT_TARGET_OPSET`).
    :param custom_functions: optional mapping from function name to Python
        callable.  Functions registered here can be called inside the traced
        body via :class:`~yobx.sql.parse.FuncCallExpr` nodes if the traced
        function constructs them directly (advanced usage).
    :param builder_cls: graph-builder class or factory callable.  Defaults to
        :class:`~yobx.xbuilder.GraphBuilder`.
    :param filename: if set, the exported ONNX model is saved to this path and
        the :class:`~yobx.container.ExportReport` is written as a companion
        Excel file (same base name with ``.xlsx`` extension).
    :param verbose: verbosity level (0 = silent).
    :return: :class:`~yobx.container.ExportArtifact` wrapping the exported
        ONNX model together with an :class:`~yobx.container.ExportReport`.

    Example — single dataframe::

        import numpy as np
        from yobx.sql.dataframe_trace import dataframe_to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        def transform(df):
            df = df.filter(df["a"] > 0)
            return df.select([(df["a"] + df["b"]).alias("total")])

        dtypes = {"a": np.float32, "b": np.float32}
        artifact = dataframe_to_onnx(transform, dtypes)

        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0,  5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        # total == array([5., 9.], dtype=float32)  (rows where a > 0)

    Example — two independent dataframes (no join)::

        import numpy as np
        from yobx.sql.dataframe_trace import dataframe_to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        def transform(df1, df2):
            return df1.select([(df1["a"] + df2["b"]).alias("total")])

        artifact = dataframe_to_onnx(transform, [{"a": np.float32}, {"b": np.float32}])

        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        # total == array([5., 7., 9.], dtype=float32)

    Example — two dataframes joined on a key column::

        import numpy as np
        from yobx.sql.dataframe_trace import dataframe_to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        def transform(df1, df2):
            return df1.join(df2, left_key="cid", right_key="id")

        dtypes1 = {"cid": np.int64, "a": np.float32}
        dtypes2 = {"id": np.int64, "b": np.float32}
        artifact = dataframe_to_onnx(transform, [dtypes1, dtypes2])
    """
    pq = trace_dataframe(func, input_dtypes)
    # Resolve the dtype dicts for the left and right tables.
    if isinstance(input_dtypes, list):
        # Check whether any query involves a join.
        all_ops: List[object] = []
        if isinstance(pq, list):
            for q in pq:
                all_ops.extend(q.operations)
        else:
            all_ops = list(pq.operations)
        has_join = any(isinstance(op, JoinOp) for op in all_ops)
        if has_join:
            left_dtypes: Dict[str, Union[np.dtype, type, str]] = input_dtypes[0]
            right_dtypes: Optional[Dict[str, Union[np.dtype, type, str]]] = (
                input_dtypes[1] if len(input_dtypes) > 1 else None
            )
        else:
            # Merge all dtype dicts — columns come from a flat table.
            left_dtypes = {}
            for d in input_dtypes:
                left_dtypes.update(d)
            right_dtypes = None
    else:
        assert isinstance(input_dtypes, dict)
        left_dtypes = input_dtypes
        right_dtypes = None

    if isinstance(pq, list):
        return parsed_queries_to_onnx(
            pq,
            left_dtypes,
            right_input_dtypes=right_dtypes,
            target_opset=target_opset,
            custom_functions=custom_functions,
            builder_cls=builder_cls,
            filename=filename,
            verbose=verbose,
        )
    return parsed_query_to_onnx(
        pq,
        left_dtypes,
        right_input_dtypes=right_dtypes,
        target_opset=target_opset,
        custom_functions=custom_functions,
        builder_cls=builder_cls,
        filename=filename,
        verbose=verbose,
    )
