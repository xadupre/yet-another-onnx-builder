"""
yobx.sql — SQL-to-ONNX converter.

This sub-package converts SQL queries into ONNX graphs.
Every column referenced in the query is treated as a **distinct 1-D ONNX
input** tensor, matching the convention used for tabular data.

The entry point is :func:`sql_to_onnx`.  Internally the query is first parsed
by :func:`~yobx.sql.parse.parse_sql` into a :class:`~yobx.sql.parse.ParsedQuery`
containing an ordered list of :class:`~yobx.sql.parse.SqlOperation` objects
(one per SQL clause), and the converter then emits ONNX nodes for each.

Public API
----------
* :func:`sql_to_onnx` — high-level entry point: SQL string →
  :class:`~yobx.container.ExportArtifact`
* :func:`sql_to_onnx_graph` — low-level entry point: SQL string → nodes added
  to an existing :class:`~yobx.typing.GraphBuilderProtocol`
* :func:`parsed_query_to_onnx` — high-level entry point: parsed query →
  :class:`~yobx.container.ExportArtifact`
* :func:`parsed_query_to_onnx_graph` — low-level entry point: parsed query →
  nodes added to an existing :class:`~yobx.typing.GraphBuilderProtocol`
* :func:`lazyframe_to_onnx` — high-level entry point: ``polars.LazyFrame`` →
  :class:`~yobx.container.ExportArtifact`
* :func:`dataframe_to_onnx` — high-level entry point: traced DataFrame
  function → :class:`~yobx.container.ExportArtifact`
* :func:`trace_dataframe` — trace a DataFrame function →
  :class:`~yobx.sql.parse.ParsedQuery`
* :func:`to_onnx` — unified entry point: SQL string, DataFrame-tracing callable,
  **or** ``polars.LazyFrame`` → :class:`~yobx.container.ExportArtifact`
* :func:`~yobx.sql.parse.parse_sql` — parse a SQL string into a
  :class:`~yobx.sql.parse.ParsedQuery`
* :class:`~yobx.sql.parse.ParsedQuery` — parsed query container
* :class:`~yobx.sql.parse.SelectOp` — SELECT clause operation
* :class:`~yobx.sql.parse.FilterOp` — WHERE / filter clause operation
* :class:`~yobx.sql.parse.GroupByOp` — GROUP BY clause operation
* :class:`~yobx.sql.parse.JoinOp` — JOIN clause operation
* :class:`~yobx.xtracing.dataframe_trace.TracedDataFrame` — proxy DataFrame for tracing
* :class:`~yobx.xtracing.dataframe_trace.TracedSeries` — proxy column series for tracing
* :class:`~yobx.xtracing.dataframe_trace.TracedCondition` — proxy boolean condition

Example
-------
::

    import numpy as np
    from yobx.sql import sql_to_onnx
    from yobx.reference import ExtendedReferenceEvaluator

    dtypes = {"a": np.float32, "b": np.float32}
    artifact = sql_to_onnx(
        "SELECT a + b AS total FROM t WHERE a > 0",
        dtypes,
    )

    ref = ExtendedReferenceEvaluator(artifact)
    a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    b = np.array([4.0,  5.0, 6.0], dtype=np.float32)
    (total,) = ref.run(None, {"a": a, "b": b})
    # total == array([5., 9.], dtype=float32)  (rows where a > 0)
"""

from .sql_convert import (
    sql_to_onnx,
    sql_to_onnx_graph,
    parsed_query_to_onnx,
    parsed_query_to_onnx_graph,
)
from .polars_convert import lazyframe_to_onnx
from .convert import to_onnx, dataframe_to_onnx, trace_numpy_to_onnx
from .coverage import get_sql_coverage, not_implemented_error
from yobx.xtracing.parse import (
    AggExpr,
    BinaryExpr,
    ColumnRef,
    Condition,
    FilterOp,
    FuncCallExpr,
    GroupByOp,
    JoinOp,
    Literal,
    ParsedQuery,
    PivotTableOp,
    SelectItem,
    SelectOp,
    parse_sql,
)

_DATAFRAME_TRACE_NAMES = frozenset(
    ["TracedCondition", "TracedDataFrame", "TracedGroupBy", "TracedSeries", "trace_dataframe"]
)


def __getattr__(name: str) -> object:
    if name in _DATAFRAME_TRACE_NAMES:
        from yobx.xtracing.dataframe_trace import (  # noqa: PLC0415
            TracedCondition,
            TracedDataFrame,
            TracedGroupBy,
            TracedSeries,
            trace_dataframe,
        )

        _symbols = {
            "TracedCondition": TracedCondition,
            "TracedDataFrame": TracedDataFrame,
            "TracedGroupBy": TracedGroupBy,
            "TracedSeries": TracedSeries,
            "trace_dataframe": trace_dataframe,
        }
        # Cache in module globals to avoid repeated __getattr__ calls.
        import sys as _sys

        _mod = _sys.modules[__name__]
        for _k, _v in _symbols.items():
            setattr(_mod, _k, _v)
        return _symbols[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AggExpr",
    "BinaryExpr",
    "ColumnRef",
    "Condition",
    "FilterOp",
    "FuncCallExpr",
    "GroupByOp",
    "JoinOp",
    "Literal",
    "ParsedQuery",
    "PivotTableOp",
    "SelectItem",
    "SelectOp",
    "dataframe_to_onnx",
    "get_sql_coverage",
    "lazyframe_to_onnx",
    "not_implemented_error",
    "parse_sql",
    "parsed_query_to_onnx",
    "parsed_query_to_onnx_graph",
    "sql_to_onnx",
    "sql_to_onnx_graph",
    "to_onnx",
    "trace_numpy_to_onnx",
]
