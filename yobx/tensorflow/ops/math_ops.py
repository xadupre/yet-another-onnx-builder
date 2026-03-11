"""
Converters for miscellaneous TF math ops.

Argument reduction
------------------
``ArgMax``, ``ArgMin``

Top-K
-----
``TopKV2``

Predicate / classification
--------------------------
``IsNan``, ``IsInf``, ``IsFinite``
"""

from typing import Any, Dict, List

import numpy as np
from onnx import TensorProto
import tensorflow as tf

from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol


# ---------------------------------------------------------------------------
# Argument reduction
# ---------------------------------------------------------------------------


@register_tf_op_converter("ArgMax")
def convert_argmax(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``ArgMax`` (tf.math.argmax) → ONNX ``ArgMax``."""
    axis = int(tf.make_ndarray(op.inputs[1].op.get_attr("value")))
    return g.op.ArgMax(op.inputs[0].name, axis=axis, keepdims=0, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("ArgMin")
def convert_argmin(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``ArgMin`` (tf.math.argmin) → ONNX ``ArgMin``."""
    axis = int(tf.make_ndarray(op.inputs[1].op.get_attr("value")))
    return g.op.ArgMin(op.inputs[0].name, axis=axis, keepdims=0, outputs=outputs[:1], name=op.name)


# ---------------------------------------------------------------------------
# Top-K
# ---------------------------------------------------------------------------


@register_tf_op_converter("TopKV2")
def convert_topk(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``TopKV2`` (tf.math.top_k) → ONNX ``TopK``.

    ONNX ``TopK`` requires *K* to be a 1-D int64 tensor; TF passes it as a
    0-D int32 scalar, so we cast and reshape before forwarding.
    """
    sorted_attr = int(op.get_attr("sorted"))
    k_i64 = g.op.Cast(op.inputs[1].name, to=TensorProto.INT64, name=f"{op.name}_k_cast")
    k_1d = g.op.Reshape(
        k_i64,
        np.array([1], dtype=np.int64),
        name=f"{op.name}_k_reshape",
    )
    return g.op.TopK(
        op.inputs[0].name,
        k_1d,
        axis=-1,
        largest=1,
        sorted=sorted_attr,
        outputs=outputs[:2],
        name=op.name,
    )


# ---------------------------------------------------------------------------
# Predicate / classification
# ---------------------------------------------------------------------------


@register_tf_op_converter("IsNan")
def convert_isnan(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``IsNan`` (tf.math.is_nan) → ONNX ``IsNaN``."""
    return g.op.IsNaN(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("IsInf")
def convert_isinf(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``IsInf`` (tf.math.is_inf) → ONNX ``IsInf``."""
    return g.op.IsInf(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("IsFinite")
def convert_isfinite(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``IsFinite`` (tf.math.is_finite) → ONNX ``Not(Or(IsNaN(x), IsInf(x)))``."""
    x = op.inputs[0].name
    is_nan = g.op.IsNaN(x, name=f"{op.name}_isnan")
    is_inf = g.op.IsInf(x, name=f"{op.name}_isinf")
    nan_or_inf = g.op.Or(is_nan, is_inf, name=f"{op.name}_or")
    return g.op.Not(nan_or_inf, outputs=outputs[:1], name=op.name)
