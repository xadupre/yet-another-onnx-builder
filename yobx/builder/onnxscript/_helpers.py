"""
Helper utilities for :class:`~yobx.builder.onnxscript.OnnxScriptGraphBuilder`.

These functions convert between the yobx / ONNX proto type system and the
``onnx-ir`` type system used by :class:`onnxscript._internal.builder.GraphBuilder`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from onnx import AttributeProto, TensorProto

import onnx_ir as ir


# Mapping from ONNX opset version to ONNX IR version, as per the ONNX spec.
_OPSET_TO_IR: Dict[int, int] = {
    1: 3,
    2: 3,
    3: 3,
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 4,
    9: 4,
    10: 5,
    11: 6,
    12: 7,
    13: 7,
    14: 7,
    15: 8,
    16: 8,
    17: 8,
    18: 8,
    19: 9,
    20: 9,
    21: 10,
    22: 10,
}


def to_ir_dtype(elem_type: Optional[int]) -> Optional[ir.DataType]:
    """Convert an ONNX ``TensorProto`` element-type integer to :class:`ir.DataType`.

    :param elem_type: ONNX element type (e.g. ``TensorProto.FLOAT == 1``),
        or ``None`` / 0 for *unknown*.
    :return: Corresponding :class:`ir.DataType`, or ``None`` when unknown.
    """
    if not elem_type:
        return None
    return ir.DataType(elem_type)


def to_ir_shape(
    shape: Optional[Sequence[Optional[Union[int, str]]]]
) -> Optional[ir.Shape]:
    """Convert a yobx-style shape tuple to :class:`ir.Shape`.

    :param shape: A sequence of dimension sizes.  Each element may be an
        ``int`` (static), a ``str`` (symbolic / dynamic), or ``None``
        (fully unknown dimension).
    :return: :class:`ir.Shape`, or ``None`` when *shape* itself is ``None``.
    """
    if shape is None:
        return None
    return ir.Shape(list(shape))


def value_to_ir_tensor(value: Any, name: str) -> ir.TensorProtocol:
    """Convert an initializer *value* to an :class:`ir.TensorProtocol`.

    Supported input types:

    * :class:`numpy.ndarray`
    * scalar Python ``int`` or ``float`` (promoted to 0-D arrays)
    * :class:`onnx.TensorProto` (converted via :func:`ir.from_proto`)
    * Any object already satisfying :class:`ir.TensorProtocol`

    :param value: The raw initializer value.
    :param name: Name that should be associated with the resulting tensor.
    :return: An :class:`ir.TensorProtocol` suitable for passing to
        :meth:`onnxscript._internal.builder.GraphBuilder.initializer`.
    :raises TypeError: When *value* has an unsupported type.
    """
    if isinstance(value, TensorProto):
        t = ir.from_proto(value)
        return t
    if isinstance(value, int):
        value = np.array(value, dtype=np.int64)
    elif isinstance(value, float):
        value = np.array(value, dtype=np.float32)
    if isinstance(value, np.ndarray):
        return ir.tensor(value, name=name)
    # Try to use it as-is (e.g. already an ir.TensorProtocol subclass)
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        return ir.tensor(np.array(value), name=name)
    raise TypeError(
        f"Cannot convert initializer {name!r} of type {type(value)} "
        "to an onnx-ir tensor.  Supported types: numpy.ndarray, int, float, "
        "onnx.TensorProto."
    )


def kwargs_to_ir_attrs(
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert a mixed dict of attribute values to a form accepted by
    :meth:`onnxscript._internal.builder.GraphBuilder.call_op`.

    :class:`onnx.AttributeProto` instances are converted to their Python
    equivalents via :func:`ir.from_proto`; all other values are passed
    through unchanged.

    :param kwargs: Raw keyword-argument dict as produced by yobx
        :meth:`~yobx.xbuilder.GraphBuilder.make_node`.
    :return: Filtered/converted dict suitable for ``call_op``.
    """
    result: Dict[str, Any] = {}
    for key, val in kwargs.items():
        if isinstance(val, AttributeProto):
            ir_attr = ir.from_proto(val)
            result[key] = ir_attr.value
        else:
            result[key] = val
    return result


def default_ir_version(opset_version: int) -> int:
    """Return a sensible ONNX IR version for a given opset version.

    :param opset_version: ONNX opset version.
    :return: Recommended IR version.
    """
    return _OPSET_TO_IR.get(opset_version, 10)
