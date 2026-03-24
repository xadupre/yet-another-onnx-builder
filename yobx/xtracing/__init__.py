"""
xtracing — Trace numpy and DataFrame operations and convert them to ONNX.

This sub-package provides lightweight mechanisms for exporting Python
functions to ONNX.

**Numpy tracing** — replace numpy array arguments with :class:`NumpyArray`
proxy objects: every arithmetic operation, ufunc call, or reduction performed
on a :class:`NumpyArray` is recorded as an ONNX node in an underlying
:class:`~yobx.xbuilder.GraphBuilder` rather than being executed.

**DataFrame tracing** — replace DataFrame arguments with
:class:`TracedDataFrame` proxy objects: every column access, arithmetic
operation, filter, select, groupby, or join is recorded as an AST node and
then compiled to ONNX via the SQL converter.

The resulting ONNX graph can be exported either as a standalone
:class:`onnx.ModelProto` or embedded into a larger graph.

Public API
----------
* :class:`NumpyArray` — proxy that records numpy ops as ONNX nodes
* :func:`trace_numpy_function` — converter-API function: trace a numpy function
  into an existing :class:`~yobx.xbuilder.GraphBuilder`
* :func:`trace_numpy_to_onnx` — high-level entry point: convert a numpy
  function to a standalone ONNX model
* :class:`TracedDataFrame` — proxy DataFrame that records SQL-like operations
* :class:`TracedSeries` — proxy for a single column or computed expression
* :class:`TracedCondition` — proxy for a boolean predicate
* :class:`TracedGroupBy` — result of :meth:`TracedDataFrame.groupby`
* :func:`trace_dataframe` — trace a DataFrame function →
  :class:`~yobx.sql.parse.ParsedQuery`
* :func:`dataframe_to_onnx` — high-level entry point: traced DataFrame
  function → :class:`~yobx.container.ExportArtifact`

Example (numpy)
---------------
::

    import numpy as np
    from yobx.xtracing import trace_numpy_to_onnx

    def my_func(X):
        return np.sqrt(np.abs(X) + 1)

    X_sample = np.random.randn(4, 3).astype(np.float32)
    onx = trace_numpy_to_onnx(my_func, X_sample)

Example (DataFrame)
-------------------
::

    import numpy as np
    from yobx.xtracing import dataframe_to_onnx

    def transform(df):
        df = df.filter(df["a"] > 0)
        return df.select([(df["a"] + df["b"]).alias("total")])

    artifact = dataframe_to_onnx(transform, {"a": np.float32, "b": np.float32})
"""

from .tracing import trace_numpy_function, trace_numpy_to_onnx
from .dataframe_trace import (
    TracedCondition,
    TracedDataFrame,
    TracedGroupBy,
    TracedSeries,
    dataframe_to_onnx,
    trace_dataframe,
)

__all__ = [
    "TracedCondition",
    "TracedDataFrame",
    "TracedGroupBy",
    "TracedSeries",
    "dataframe_to_onnx",
    "trace_dataframe",
    "trace_numpy_function",
    "trace_numpy_to_onnx",
]
