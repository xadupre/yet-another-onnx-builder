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
"""

from .convert import sql_to_onnx, sql_to_onnx_graph
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
    "sql_to_onnx",
    "sql_to_onnx_graph",
]
