"""Converters for TFLite shape / tensor-manipulation ops:
RESHAPE, SQUEEZE, EXPAND_DIMS, TRANSPOSE, CONCATENATION,
MEAN, SUM, REDUCE_MAX, REDUCE_MIN."""

from typing import Any, Dict, List

import numpy as np

from ..litert_helper import BuiltinOperator, TFLiteOperator
from ..register import register_litert_op_converter
from ...typing import GraphBuilderExtendedProtocol


@register_litert_op_converter(BuiltinOperator.RESHAPE)
def convert_reshape(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: TFLiteOperator
) -> str:
    """TFLite ``RESHAPE`` → ONNX ``Reshape``.

    The target shape comes from tensor input 1 (an int32 1-D tensor).
    """
    x = op.inputs[0]
    shape_tensor = op.inputs[1]
    # Cast to int64 as ONNX Reshape requires int64 shape.
    from onnx import TensorProto

    shape_i64 = g.op.Cast(shape_tensor, to=TensorProto.INT64, name="litert_reshape_shape_cast")
    return g.op.Reshape(x, shape_i64, outputs=outputs, name="litert_reshape")


@register_litert_op_converter(BuiltinOperator.SQUEEZE)
def convert_squeeze(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: TFLiteOperator
) -> str:
    """TFLite ``SQUEEZE`` → ONNX ``Squeeze``.

    The axes to squeeze come from ``builtin_options["squeeze_dims"]``; if
    absent, squeeze all size-1 dims.
    """
    axes = op.builtin_options.get("squeeze_dims")
    if axes:
        axes_arr = np.array(list(axes), dtype=np.int64)
        return g.op.Squeeze(op.inputs[0], axes_arr, outputs=outputs, name="litert_squeeze")
    return g.op.Squeeze(op.inputs[0], outputs=outputs, name="litert_squeeze_all")


@register_litert_op_converter(BuiltinOperator.EXPAND_DIMS)
def convert_expand_dims(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: TFLiteOperator
) -> str:
    """TFLite ``EXPAND_DIMS`` → ONNX ``Unsqueeze``.

    The axis index comes from tensor input 1.
    """
    from onnx import TensorProto

    axis_i64 = g.op.Cast(op.inputs[1], to=TensorProto.INT64, name="litert_expand_axis_cast")
    return g.op.Unsqueeze(op.inputs[0], axis_i64, outputs=outputs, name="litert_expand_dims")


@register_litert_op_converter(BuiltinOperator.TRANSPOSE)
def convert_transpose(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: TFLiteOperator
) -> str:
    """TFLite ``TRANSPOSE`` → ONNX ``Transpose``.

    The permutation is stored as tensor input 1.  We read it directly from
    the graph builder's initializer map because ONNX ``Transpose`` requires
    the ``perm`` attribute (not a runtime input).
    """
    perm_tensor = g.initializers_dict.get(op.inputs[1])
    if perm_tensor is not None:
        perm = [int(v) for v in perm_tensor.ravel()]
        return g.op.Transpose(op.inputs[0], perm=perm, outputs=outputs, name="litert_transpose")
    # If the perm tensor is not a static initializer we cannot inline it.
    # Emit Transpose without perm (reverses axes).
    return g.op.Transpose(op.inputs[0], outputs=outputs, name="litert_transpose_noperm")


@register_litert_op_converter(BuiltinOperator.CONCATENATION)
def convert_concatenation(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: TFLiteOperator
) -> str:
    """TFLite ``CONCATENATION`` → ONNX ``Concat``."""
    axis = op.builtin_options.get("axis", 0)
    return g.op.Concat(*op.inputs, axis=axis, outputs=outputs, name="litert_concat")


@register_litert_op_converter(BuiltinOperator.MEAN)
def convert_mean(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: TFLiteOperator
) -> str:
    """TFLite ``MEAN`` → ONNX ``ReduceMean``."""
    from onnx import TensorProto

    keep_dims = op.builtin_options.get("keep_dims", False)
    axes_i64 = g.op.Cast(op.inputs[1], to=TensorProto.INT64, name="litert_mean_axes")
    return g.op.ReduceMean(
        op.inputs[0], axes_i64, keepdims=int(keep_dims), outputs=outputs, name="litert_mean"
    )


@register_litert_op_converter(BuiltinOperator.SUM)
def convert_sum(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: TFLiteOperator
) -> str:
    """TFLite ``SUM`` → ONNX ``ReduceSum``."""
    from onnx import TensorProto

    keep_dims = op.builtin_options.get("keep_dims", False)
    axes_i64 = g.op.Cast(op.inputs[1], to=TensorProto.INT64, name="litert_sum_axes")
    return g.op.ReduceSum(
        op.inputs[0], axes_i64, keepdims=int(keep_dims), outputs=outputs, name="litert_sum"
    )


@register_litert_op_converter(BuiltinOperator.REDUCE_MAX)
def convert_reduce_max(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: TFLiteOperator
) -> str:
    """TFLite ``REDUCE_MAX`` → ONNX ``ReduceMax``."""
    from onnx import TensorProto

    keep_dims = op.builtin_options.get("keep_dims", False)
    axes_i64 = g.op.Cast(op.inputs[1], to=TensorProto.INT64, name="litert_reduce_max_axes")
    return g.op.ReduceMax(
        op.inputs[0], axes_i64, keepdims=int(keep_dims), outputs=outputs, name="litert_reduce_max"
    )


@register_litert_op_converter(BuiltinOperator.REDUCE_MIN)
def convert_reduce_min(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: TFLiteOperator
) -> str:
    """TFLite ``REDUCE_MIN`` → ONNX ``ReduceMin``."""
    from onnx import TensorProto

    keep_dims = op.builtin_options.get("keep_dims", False)
    axes_i64 = g.op.Cast(op.inputs[1], to=TensorProto.INT64, name="litert_reduce_min_axes")
    return g.op.ReduceMin(
        op.inputs[0], axes_i64, keepdims=int(keep_dims), outputs=outputs, name="litert_reduce_min"
    )
