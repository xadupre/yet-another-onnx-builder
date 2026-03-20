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
* :func:`to_onnx` — convenience wrapper: ``polars.LazyFrame`` → ONNX model.
  The query is extracted automatically from the frame when the frame was
  produced by :meth:`polars.LazyFrame.sql`, or can be supplied explicitly
  as a second argument.
* :func:`polars_frame_to_sql` — extract a SQL string and dtype map from a
  ``polars.LazyFrame`` produced by ``.sql(...)`` (used internally by
  :func:`to_onnx`, also available as a standalone utility)
* :func:`polars_schema_to_input_dtypes` — extract a column-dtype mapping from
  a ``polars.LazyFrame`` or ``polars.DataFrame`` (used internally by
  :func:`to_onnx`, also available as a standalone utility)
* :func:`~yobx.sql.parse.parse_sql` — parse a SQL string into a
  :class:`~yobx.sql.parse.ParsedQuery`
* :class:`~yobx.sql.parse.ParsedQuery` — parsed query container
* :class:`~yobx.sql.parse.SelectOp` — SELECT clause operation
* :class:`~yobx.sql.parse.FilterOp` — WHERE / filter clause operation
* :class:`~yobx.sql.parse.GroupByOp` — GROUP BY clause operation
* :class:`~yobx.sql.parse.JoinOp` — JOIN clause operation

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

polars example
--------------
::

    import polars as pl
    from yobx.sql import to_onnx
    from yobx.reference import ExtendedReferenceEvaluator

    src = pl.LazyFrame({"a": pl.Series([1.0, -2.0, 3.0], dtype=pl.Float32),
                       "b": pl.Series([4.0,  5.0, 6.0], dtype=pl.Float32)})
    # Query embedded in the LazyFrame via .sql():
    onx = to_onnx(src.sql("SELECT a + b AS total FROM self WHERE a > 0"))
    # Or pass the query string explicitly:
    # onx = to_onnx(src, "SELECT a + b AS total FROM t WHERE a > 0")
"""

from .convert import sql_to_onnx, sql_to_onnx_graph, to_onnx
from ._polars_helper import polars_frame_to_sql, polars_schema_to_input_dtypes
from .parse import (
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
    SelectItem,
    SelectOp,
    parse_sql,
)

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
    "SelectItem",
    "SelectOp",
    "parse_sql",
    "polars_frame_to_sql",
    "polars_schema_to_input_dtypes",
    "sql_to_onnx",
    "sql_to_onnx_graph",
    "to_onnx",
]
