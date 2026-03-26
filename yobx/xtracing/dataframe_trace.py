"""
DataFrame function tracer — convert a Python function operating on a
:class:`TracedDataFrame` into an ONNX model via
:func:`yobx.sql.to_onnx`.

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
* **Arithmetic**: ``+``, ``-``, ``*``, ``/`` between columns and/or scalars,
  or applied element-wise across all columns of a :class:`TracedDataFrame`
  (e.g. ``df + 1``, ``df * 2.0``)
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
* **Copy**: :meth:`TracedDataFrame.copy`

Example
-------
::

    import numpy as np
    from yobx.xtracing.dataframe_trace import dataframe_to_onnx
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
from onnx import TensorProto

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

# Mapping from numpy dtype to onnx.TensorProto integer element type.
_NP_DTYPE_TO_TENSOR_PROTO: Dict[np.dtype, int] = {
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


def _to_tensor_proto_dtype(dt: Union[np.dtype, type, str]) -> int:
    """Convert a numpy dtype / type / string to an :data:`onnx.TensorProto` element type.

    :param dt: a :class:`numpy.dtype`, a numpy scalar type (``np.float32``,
        ``np.int64``, …), a Python built-in numeric type (``float`` maps to
        ``DOUBLE``, ``int`` maps to ``INT64``), or a dtype string
        (``"float32"``, ``"int64"``, …).  The value is first resolved via
        :func:`numpy.dtype` before looking up in :data:`_NP_DTYPE_TO_TENSOR_PROTO`.
    :return: the corresponding :data:`onnx.TensorProto` integer constant.
    :raises ValueError: if the dtype is not recognised.

    .. note::

        ``np.dtype("object")`` is mapped to :data:`onnx.TensorProto.STRING`
        following the same convention used elsewhere in this package
        (see :data:`yobx.sql.sql_convert._NP_TO_ONNX`).  This assumes that
        ``object`` columns contain string data.
    """
    resolved = np.dtype(dt)
    result = _NP_DTYPE_TO_TENSOR_PROTO.get(resolved)
    if result is None:
        raise ValueError(
            f"Cannot map numpy dtype {resolved!r} to an onnx.TensorProto element type."
        )
    return result


# ---------------------------------------------------------------------------
# Expression conversion helper
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TracedSeries
# ---------------------------------------------------------------------------


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

    # ------------------------------------------------------------------
    # Column alias
    # ------------------------------------------------------------------

    def alias(self, name: str) -> TracedSeries:
        """Return a copy tagged with *name* as the output alias.

        :param name: the alias to assign.
        :return: a new :class:`TracedSeries` carrying the alias.
        """
        return TracedSeries(self._expr, alias=name)

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Comparison operators — return TracedCondition
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Aggregation methods
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Conversion helper
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TracedCondition
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TracedGroupBy
# ---------------------------------------------------------------------------


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

        new_cols: Dict[ColumnRef, TracedSeries] = {}
        for item in select_items:
            # Use dtype from ColumnRef when available; otherwise TensorProto.UNDEFINED (0).
            dtype = item.expr.dtype if isinstance(item.expr, ColumnRef) else 0
            new_cols[ColumnRef(item.output_name(), dtype=dtype)] = TracedSeries(
                item.expr, item.alias
            )
        return TracedDataFrame(new_cols, new_ops, list(self._df._source_columns))


# ---------------------------------------------------------------------------
# TracedDataFrame
# ---------------------------------------------------------------------------


class TracedDataFrame:
    """Proxy for a DataFrame that records SQL-like operations for ONNX export.

    Create a :class:`TracedDataFrame` via :func:`trace_dataframe` (recommended)
    or construct it directly with a :class:`~yobx.xtracing.parse.ColumnRef` →
    :class:`TracedSeries` mapping.  Apply operations (``filter``, ``select``,
    arithmetic on columns) to build up the computation graph.  Convert to an
    ONNX model via :func:`dataframe_to_onnx`.

    :param columns: mapping from :class:`~yobx.xtracing.parse.ColumnRef` to
        :class:`TracedSeries`.
    :param ops: accumulated list of :class:`~yobx.sql.parse.SqlOperation`
        objects (usually starts empty).
    :param source_columns: ordered list of original source column names (those
        that will become ONNX inputs).  Defaults to the column names extracted
        from the keys of *columns*.
    """

    def __init__(
        self,
        columns: Dict[ColumnRef, TracedSeries],
        ops: Optional[List] = None,
        source_columns: Optional[List[str]] = None,
    ) -> None:
        self._columns: Dict[ColumnRef, TracedSeries] = columns
        self._ops: List = list(ops) if ops is not None else []
        self._source_columns: List[str] = (
            list(source_columns)
            if source_columns is not None
            else [ref.column for ref in columns]
        )

    # ------------------------------------------------------------------
    # Internal helpers for string-based column lookup
    # ------------------------------------------------------------------

    def _find_ref(self, name: str) -> Optional[ColumnRef]:
        """Return the :class:`ColumnRef` key whose ``.column`` matches *name*."""
        for ref in self._columns:
            if ref.column == name:
                return ref
        return None

    def _get_series_by_name(self, name: str) -> Optional[TracedSeries]:
        """Return the :class:`TracedSeries` for the column named *name*."""
        ref = self._find_ref(name)
        return self._columns[ref] if ref is not None else None

    # ------------------------------------------------------------------
    # Column access
    # ------------------------------------------------------------------

    def __getitem__(
        self, key: Union[str, List[str], TracedCondition]
    ) -> Union[TracedSeries, TracedDataFrame]:
        if isinstance(key, str):
            series = self._get_series_by_name(key)
            if series is None:
                raise KeyError(f"Column {key!r} not found in traced DataFrame")
            return series
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
        for ref, series in cols.items():
            if ref.column == name:
                return series
        raise AttributeError(
            f"TracedDataFrame has no column {name!r}. "
            f"Available columns: {[ref.column for ref in cols]}"
        )

    @property
    def columns(self) -> List[str]:
        """Return the ordered list of column names in the current frame."""
        return [ref.column for ref in self._columns]

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
                    series = self._get_series_by_name(e)
                    if series is None:
                        raise KeyError(f"Column {e!r} not found in traced DataFrame")
                    series_list.append(series)
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
        new_cols: Dict[ColumnRef, TracedSeries] = {}
        for item in select_items:
            # Use dtype from ColumnRef when available; otherwise TensorProto.UNDEFINED (0).
            dtype = item.expr.dtype if isinstance(item.expr, ColumnRef) else 0
            new_cols[ColumnRef(item.output_name(), dtype=dtype)] = TracedSeries(
                item.expr, item.alias
            )
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
            # Remove an existing key with the same column name, if any.
            existing_ref = self._find_ref(name)
            if existing_ref is not None:
                del new_cols[existing_ref]
            new_cols[ColumnRef(name)] = TracedSeries(expr._expr, alias=name)
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

    def copy(self, deep: bool = True) -> "TracedDataFrame":
        """Return a copy of this :class:`TracedDataFrame`.

        Because :class:`TracedDataFrame` holds immutable AST nodes rather than
        actual data, both shallow and deep copies are equivalent: a new
        :class:`TracedDataFrame` is returned with the same column expressions,
        recorded operations, and source-column list.  This method exists so
        that functions written for real ``pandas.DataFrame`` objects (which
        routinely call ``.copy()`` to avoid unintentional mutation) can be
        traced without modification.

        :param deep: accepted for API compatibility with ``pandas.DataFrame.copy``
            but has no effect.
        :return: a new :class:`TracedDataFrame` representing the same query.
        """
        return TracedDataFrame(dict(self._columns), list(self._ops), list(self._source_columns))

    def join(
        self,
        right: "TracedDataFrame",
        left_key: Union[str, List[str]],
        right_key: Union[str, List[str]],
        join_type: str = "inner",
    ) -> "TracedDataFrame":
        """Record an equi-join with another :class:`TracedDataFrame`.

        :param right: the right-hand :class:`TracedDataFrame` to join with.
        :param left_key: column name (or list of names) from *this* (left) frame
            used in the join predicate.
        :param right_key: column name (or list of names) from *right* frame
            used in the join predicate.  Must have the same length as
            *left_key* when both are lists.
        :param join_type: join type — ``'inner'`` (default), ``'left'``,
            ``'right'``, or ``'full'``.
        :return: a new :class:`TracedDataFrame` containing columns from both
            sides with a :class:`~yobx.sql.parse.JoinOp` recorded.

        Example::

            import numpy as np
            from yobx.xtracing.dataframe_trace import dataframe_to_onnx

            def transform(df1, df2):
                return df1.join(df2, left_key="cid", right_key="id")

            dtypes1 = {"cid": np.int64, "a": np.float32}
            dtypes2 = {"id": np.int64, "b": np.float32}
            artifact = dataframe_to_onnx(transform, [dtypes1, dtypes2])
        """

        # Normalise keys to lists.
        left_keys: List[str] = [left_key] if isinstance(left_key, str) else list(left_key)
        right_keys: List[str] = [right_key] if isinstance(right_key, str) else list(right_key)
        if len(left_keys) != len(right_keys):
            raise ValueError(
                f"left_key and right_key must have the same length, "
                f"got {len(left_keys)} and {len(right_keys)}"
            )

        # Build ColumnRef lists for both sides so that _populate_graph can
        # classify and type columns (including join-key columns that may not
        # appear in any SELECT expression) without requiring input_dtypes.
        def _col_refs_from(frame: "TracedDataFrame") -> "List[ColumnRef]":
            refs: List[ColumnRef] = []
            for _name in frame._source_columns:
                _ref = frame._find_ref(_name)
                _dtype = _ref.dtype if _ref is not None else 0
                refs.append(ColumnRef(column=_name, dtype=_dtype))
            return refs

        join_op = JoinOp(
            right_table="r",
            left_keys=left_keys,
            right_keys=right_keys,
            join_type=join_type,
            left_columns=_col_refs_from(self),
            right_columns=_col_refs_from(right),
        )
        # Merge columns: left columns take priority on name collision.
        merged_cols = dict(self._columns)
        for ref, series in right._columns.items():
            if not any(k.column == ref.column for k in merged_cols):
                merged_cols[ref] = series
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
    # Arithmetic operators (element-wise across all columns)
    # ------------------------------------------------------------------

    def _apply_elementwise(
        self, op: str, other: object, reversed: bool = False
    ) -> "TracedDataFrame":
        """Apply a binary arithmetic operation element-wise to every column.

        :param op: the operator string (``'+'``, ``'-'``, ``'*'``, or
            ``'/'``).
        :param other: a scalar (``int`` or ``float``) or another
            :class:`TracedDataFrame` with the same column names.
        :param reversed: when ``True`` the operands are swapped so that
            ``other op self`` is computed instead of ``self op other``.
        :return: a new :class:`TracedDataFrame` whose column expressions
            embed the arithmetic.
        """
        new_cols: Dict[ColumnRef, TracedSeries] = {}
        for ref, series in self._columns.items():
            if isinstance(other, TracedDataFrame):
                other_series = other._get_series_by_name(ref.column)
                if other_series is None:
                    raise KeyError(
                        f"Column {ref.column!r} not found in right DataFrame. "
                        f"Available columns: {[r.column for r in other._columns]}"
                    )
                right_expr = other_series._expr
            else:
                right_expr = _to_ast(other)
            if reversed:
                expr: object = BinaryExpr(right_expr, op, series._expr)
            else:
                expr = BinaryExpr(series._expr, op, right_expr)
            new_cols[ref] = TracedSeries(expr)
        return TracedDataFrame(new_cols, list(self._ops), list(self._source_columns))

    def __add__(self, other: object) -> "TracedDataFrame":
        """Return a new frame with *other* added to every column (``df + other``)."""
        return self._apply_elementwise("+", other)

    def __radd__(self, other: object) -> "TracedDataFrame":
        """Return a new frame with every column added to *other* (``other + df``)."""
        return self._apply_elementwise("+", other, reversed=True)

    def __sub__(self, other: object) -> "TracedDataFrame":
        """Return a new frame with *other* subtracted from every column (``df - other``)."""
        return self._apply_elementwise("-", other)

    def __rsub__(self, other: object) -> "TracedDataFrame":
        """Return a new frame with every column subtracted from *other* (``other - df``)."""
        return self._apply_elementwise("-", other, reversed=True)

    def __mul__(self, other: object) -> "TracedDataFrame":
        """Return a new frame with every column multiplied by *other* (``df * other``)."""
        return self._apply_elementwise("*", other)

    def __rmul__(self, other: object) -> "TracedDataFrame":
        """Return a new frame with *other* multiplied by every column (``other * df``)."""
        return self._apply_elementwise("*", other, reversed=True)

    def __truediv__(self, other: object) -> "TracedDataFrame":
        """Return a new frame with every column divided by *other* (``df / other``)."""
        return self._apply_elementwise("/", other)

    def __rtruediv__(self, other: object) -> "TracedDataFrame":
        """Return a new frame with *other* divided by every column (``other / df``)."""
        return self._apply_elementwise("/", other, reversed=True)

    # ------------------------------------------------------------------
    # Query assembly
    # ------------------------------------------------------------------

    def to_parsed_query(self) -> ParsedQuery:
        """Assemble a :class:`~yobx.sql.parse.ParsedQuery` from the recorded ops.

        If no ``SelectOp`` has been recorded yet (e.g. only a filter was
        applied, or element-wise arithmetic was applied to the whole frame),
        a ``SELECT`` is generated from the current column expressions.  This
        correctly captures any arithmetic embedded in the column expressions
        (e.g. from ``df + 1``).

        :return: a :class:`~yobx.sql.parse.ParsedQuery` ready for ONNX
            conversion via :func:`~yobx.sql.sql_convert.parsed_query_to_onnx`.
        """
        ops = list(self._ops)
        if not any(isinstance(op, SelectOp) for op in ops):
            items = [
                SelectItem(series._expr, alias=ref.column)
                for ref, series in self._columns.items()
            ]
            ops.append(SelectOp(items=items))
        columns = _collect_columns(ops)
        # Fill in missing dtypes for join-key columns (created by
        # _collect_columns with dtype=0 from JoinOp strings) using the
        # dtype stored in the ColumnRef objects in JoinOp.left_columns and
        # JoinOp.right_columns.
        join_col_dtypes: dict = {}
        for op in ops:
            if isinstance(op, JoinOp):
                for col_ref in op.left_columns:
                    if col_ref.dtype != 0:
                        join_col_dtypes[col_ref.column] = col_ref.dtype
                for col_ref in op.right_columns:
                    if col_ref.dtype != 0:
                        join_col_dtypes[col_ref.column] = col_ref.dtype
        if join_col_dtypes:
            columns = [
                (
                    ColumnRef(col_ref.column, col_ref.table, join_col_dtypes[col_ref.column])
                    if col_ref.dtype == 0 and col_ref.column in join_col_dtypes
                    else col_ref
                )
                for col_ref in columns
            ]
        return ParsedQuery(operations=ops, from_table="t", columns=columns)

    def __repr__(self) -> str:
        return f"TracedDataFrame(columns={[ref.column for ref in self._columns]!r})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def trace_dataframe(
    func: Callable,
    input_dtypes: Union[
        Dict[str, Union[np.dtype, type, str]], List[Dict[str, Union[np.dtype, type, str]]]
    ],
) -> ParsedQuery:
    """Trace *func* and return the equivalent :class:`~yobx.sql.parse.ParsedQuery`.

    Constructs one or more :class:`TracedDataFrame` objects whose columns
    correspond to *input_dtypes*, calls *func* with them, and converts the
    resulting frame's recorded operations into a
    :class:`~yobx.sql.parse.ParsedQuery`.

    :param func: a callable that accepts one or more :class:`TracedDataFrame`
        objects and returns a :class:`TracedDataFrame`.  The function may apply
        any combination of ``filter``, ``select``, arithmetic on columns,
        ``join``, and aggregations.
    :param input_dtypes: either

        * a single ``{column: dtype}`` mapping — *func* is called with one
          :class:`TracedDataFrame`; or
        * a **list** of ``{column: dtype}`` mappings — *func* is called with
          one :class:`TracedDataFrame` per mapping, in order.

        The dtypes are not used during tracing itself; they are used only when
        the returned query is subsequently compiled to ONNX.
    :return: a :class:`~yobx.sql.parse.ParsedQuery` representing the operations
        performed by *func*.

    Example — single dataframe::

        import numpy as np
        from yobx.xtracing.dataframe_trace import trace_dataframe

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
        from yobx.xtracing.dataframe_trace import trace_dataframe

        def transform(df1, df2):
            return df1.select([(df1["a"] + df2["b"]).alias("total")])

        pq = trace_dataframe(transform, [{"a": np.float32}, {"b": np.float32}])
    """
    if isinstance(input_dtypes, list):
        dfs = []
        for d in input_dtypes:
            cols: Dict[ColumnRef, TracedSeries] = {}
            for name in d:
                ref = ColumnRef(name, dtype=_to_tensor_proto_dtype(d[name]))
                cols[ref] = TracedSeries(ref)
            dfs.append(TracedDataFrame(cols, source_columns=list(d.keys())))
        result = func(*dfs)
    else:
        columns: Dict[ColumnRef, TracedSeries] = {}
        for name in input_dtypes:
            ref = ColumnRef(name, dtype=_to_tensor_proto_dtype(input_dtypes[name]))
            columns[ref] = TracedSeries(ref)
        df = TracedDataFrame(columns, source_columns=list(input_dtypes.keys()))
        result = func(df)
    if not isinstance(result, TracedDataFrame):
        raise TypeError(
            f"trace_dataframe: function must return a TracedDataFrame, "
            f"got {type(result).__name__!r}"
        )
    return result.to_parsed_query()
