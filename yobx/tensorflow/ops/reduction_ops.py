"""
Converters for TF reduction and cumulative ops.

Reduction
---------
``Sum`` (reduce_sum), ``Max`` (reduce_max), ``Min`` (reduce_min),
``Mean`` (reduce_mean), ``Prod`` (reduce_prod)

Variance / standard deviation
------------------------------
``tf.math.reduce_variance`` and ``tf.math.reduce_std`` are Python-level
TensorFlow functions that decompose into primitive graph ops (``Mean``,
``SquaredDifference``, ``Sqrt``, etc.).  Because every component op has a
registered converter, these high-level functions are automatically supported
without a dedicated single-op converter.

Cumulative
----------
``Cumsum``
"""

from typing import Any, Dict, List

import numpy as np
from onnx import TensorProto
import tensorflow as tf

from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol


def _axes_i64(g: GraphBuilderExtendedProtocol, op: tf.Operation) -> str:
    """Cast the second input (reduction axes, int32) to a 1-D int64 tensor for ONNX.

    TF may provide axes as a 0-D scalar (for a single axis) or a 1-D vector
    (for multiple axes).  ONNX reduction ops require a 1-D tensor, so a
    ``Reshape(..., [-1])`` is applied after casting to int64.
    """
    axes_i64 = g.op.Cast(op.inputs[1].name, to=TensorProto.INT64, name=f"{op.name}_axes_cast")
    return g.op.Reshape(
        axes_i64,
        np.array([-1], dtype=np.int64),
        name=f"{op.name}_axes_reshape",
    )


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------


@register_tf_op_converter("Sum")
def convert_reduce_sum(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Sum`` (tf.math.reduce_sum) → ONNX ``ReduceSum``."""
    keep_dims = int(op.get_attr("keep_dims"))
    axes = _axes_i64(g, op)
    return g.op.ReduceSumAnyOpset(
        op.inputs[0].name, axes, keepdims=keep_dims, outputs=outputs[:1], name=op.name
    )


@register_tf_op_converter("Max")
def convert_reduce_max(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Max`` (tf.math.reduce_max) → ONNX ``ReduceMax``."""
    keep_dims = int(op.get_attr("keep_dims"))
    axes = _axes_i64(g, op)
    return g.op.ReduceMaxAnyOpset(
        op.inputs[0].name, axes, keepdims=keep_dims, outputs=outputs[:1], name=op.name
    )


@register_tf_op_converter("Min")
def convert_reduce_min(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Min`` (tf.math.reduce_min) → ONNX ``ReduceMin``."""
    keep_dims = int(op.get_attr("keep_dims"))
    axes = _axes_i64(g, op)
    return g.op.ReduceMinAnyOpset(
        op.inputs[0].name, axes, keepdims=keep_dims, outputs=outputs[:1], name=op.name
    )


@register_tf_op_converter("Mean")
def convert_reduce_mean(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Mean`` (tf.math.reduce_mean) → ONNX ``ReduceMean``."""
    keep_dims = int(op.get_attr("keep_dims"))
    axes = _axes_i64(g, op)
    return g.op.ReduceMeanAnyOpset(
        op.inputs[0].name, axes, keepdims=keep_dims, outputs=outputs[:1], name=op.name
    )


@register_tf_op_converter("Prod")
def convert_reduce_prod(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Prod`` (tf.math.reduce_prod) → ONNX ``ReduceProd``."""
    keep_dims = int(op.get_attr("keep_dims"))
    axes = _axes_i64(g, op)
    return g.op.ReduceProdAnyOpset(
        op.inputs[0].name, axes, keepdims=keep_dims, outputs=outputs[:1], name=op.name
    )


# ---------------------------------------------------------------------------
# Cumulative
# ---------------------------------------------------------------------------


@register_tf_op_converter("Cumsum")
def convert_cumsum(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Cumsum`` (tf.math.cumsum) → ONNX ``CumSum``."""
    exclusive = int(op.get_attr("exclusive"))
    reverse = int(op.get_attr("reverse"))
    return g.op.CumSum(
        op.inputs[0].name,
        op.inputs[1].name,
        exclusive=exclusive,
        reverse=reverse,
        outputs=outputs[:1],
        name=op.name,
    )
