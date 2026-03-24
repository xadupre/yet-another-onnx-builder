"""
Backward-compatibility shim — re-exports from :mod:`yobx.xtracing.dataframe_trace`.

The DataFrame tracer has moved to :mod:`yobx.xtracing.dataframe_trace`.
All public symbols are re-exported here so that existing code importing
from ``yobx.sql.dataframe_trace`` continues to work without modification.
"""

from ..xtracing.dataframe_trace import (  # noqa: F401
    TracedCondition,
    TracedDataFrame,
    TracedGroupBy,
    TracedSeries,
    _to_ast,
    dataframe_to_onnx,
    trace_dataframe,
)

__all__ = [
    "TracedCondition",
    "TracedDataFrame",
    "TracedGroupBy",
    "TracedSeries",
    "_to_ast",
    "dataframe_to_onnx",
    "trace_dataframe",
]
