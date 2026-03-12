"""
Converters for ``tf.nn`` activation ops.

Activation functions
--------------------
``Elu``, ``Selu``, ``LeakyRelu``, ``LogSoftmax``
"""

from typing import Any, Dict, List

import tensorflow as tf

from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol


@register_tf_op_converter("Elu")
def convert_elu(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Elu`` (``tf.nn.elu``) → ONNX ``Elu``."""
    return g.op.Elu(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Selu")
def convert_selu(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Selu`` (``tf.nn.selu``) → ONNX ``Selu``.

    Both TensorFlow and ONNX use the same fixed coefficients:
    ``alpha = 1.6732632423543772`` and ``gamma = 1.0507009873554805``.
    """
    return g.op.Selu(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("LeakyRelu")
def convert_leaky_relu(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``LeakyRelu`` (``tf.nn.leaky_relu``) → ONNX ``LeakyRelu``.

    The ``alpha`` negative-slope value is read from the TF op attribute
    and forwarded to the ONNX node unchanged.
    """
    alpha = float(op.get_attr("alpha"))
    return g.op.LeakyRelu(op.inputs[0].name, alpha=alpha, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("LogSoftmax")
def convert_log_softmax(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``LogSoftmax`` (``tf.nn.log_softmax``) → ONNX ``LogSoftmax(axis=-1)``.

    TensorFlow applies log-softmax along the last axis; the ONNX
    ``LogSoftmax`` operator defaults to ``axis=1`` in older opsets, so
    the axis is set explicitly to ``-1`` to match TF semantics.
    """
    return g.op.LogSoftmax(op.inputs[0].name, axis=-1, outputs=outputs[:1], name=op.name)
