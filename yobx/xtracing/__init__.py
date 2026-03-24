"""
xtracing — Trace numpy operations and convert them to ONNX.

This sub-package provides a lightweight mechanism for exporting Python
functions that use numpy operations to ONNX.  The key idea is to replace
the numpy array arguments with :class:`NumpyArray` proxy objects: every
arithmetic operation, ufunc call, or reduction performed on a
:class:`NumpyArray` is recorded as an ONNX node in an underlying
:class:`~yobx.xbuilder.GraphBuilder` rather than being executed.

The resulting ONNX graph can be exported either as a standalone
:class:`onnx.ModelProto` (via :func:`trace_numpy_to_onnx`) or embedded into
a larger graph when converting a
:class:`~sklearn.preprocessing.FunctionTransformer`.

Public API
----------
* :class:`NumpyArray` — proxy that records numpy ops as ONNX nodes
* :func:`trace_numpy_function` — converter-API function: trace a numpy function
  into an existing :class:`~yobx.xbuilder.GraphBuilder`
* :func:`trace_numpy_to_onnx` — high-level entry point: convert a numpy
  function to a standalone ONNX model

Example
-------
::

    import numpy as np
    from yobx.xtracing import trace_numpy_to_onnx

    def my_func(X):
        return np.sqrt(np.abs(X) + 1)

    X_sample = np.random.randn(4, 3).astype(np.float32)
    onx = trace_numpy_to_onnx(my_func, X_sample)
"""

from .numpy_array import NumpyArray
from .tracing import trace_numpy_function, trace_numpy_to_onnx

__all__ = ["NumpyArray", "trace_numpy_function", "trace_numpy_to_onnx"]
