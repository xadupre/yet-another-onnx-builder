"""
Constants and helpers for the einsum decomposition.

Ported from
https://github.com/sdpython/onnx-extended/blob/main/onnx_extended/tools/einsum/einsum_config.py
(MIT licence).
"""

from typing import Any
import numpy
import onnx

DEFAULT_OPSET: int = min(18, onnx.defs.onnx_opset_version())
DEFAULT_IR_VERSION: int = 8


def guess_proto_dtype(dtype: Any) -> int:
    """
    Returns the ONNX :class:`onnx.TensorProto` element type for a numpy dtype.

    :param dtype: numpy dtype (e.g. ``numpy.float32``)
    :return: integer element type constant
    """
    if dtype == numpy.float32:
        return onnx.TensorProto.FLOAT
    if dtype == numpy.float64:
        return onnx.TensorProto.DOUBLE
    if dtype == numpy.int32:
        return onnx.TensorProto.INT32
    if dtype == numpy.int64:
        return onnx.TensorProto.INT64
    raise ValueError(f"Unexpected value for dtype {dtype!r}.")
