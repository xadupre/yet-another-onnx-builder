"""
Converters for TF activation ops:
``Relu``, ``Relu6``, ``Sigmoid``, ``Tanh``, ``Softmax``,
``Elu``, ``Selu``, ``LeakyRelu``.
"""

from typing import Any, Dict, List
import numpy as np
import tensorflow as tf
from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol


@register_tf_op_converter("Relu")
def convert_relu(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Relu`` → ONNX ``Relu``."""
    return g.op.Relu(op.inputs[0].name, outputs=outputs, name=op.name)


@register_tf_op_converter("Relu6")
def convert_relu6(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: tf.Operation,
    verbose: int = 0,
) -> str:
    """TF ``Relu6`` → ONNX ``Clip(min=0, max=6)``."""
    return g.op.Clip(
        op.inputs[0].name,
        np.array(0.0, dtype=np.float32),
        np.array(6.0, dtype=np.float32),
        outputs=outputs[:1],
        name=op.name,
    )


@register_tf_op_converter("Sigmoid")
def convert_sigmoid(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Sigmoid`` → ONNX ``Sigmoid``."""
    return g.op.Sigmoid(op.inputs[0].name, outputs=outputs, name=op.name)


@register_tf_op_converter("Tanh")
def convert_tanh(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Tanh`` → ONNX ``Tanh``."""
    return g.op.Tanh(op.inputs[0].name, outputs=outputs, name=op.name)


@register_tf_op_converter("Softmax")
def convert_softmax(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: tf.Operation,
    verbose: int = 0,
) -> str:
    """TF ``Softmax`` → ONNX ``Softmax(axis=-1)``."""
    return g.op.Softmax(op.inputs[0].name, axis=-1, outputs=outputs, name=op.name)


@register_tf_op_converter("Elu")
def convert_elu(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Elu`` → ONNX ``Elu``."""
    return g.op.Elu(op.inputs[0].name, outputs=outputs, name=op.name)


@register_tf_op_converter("Selu")
def convert_selu(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Selu`` → ONNX ``Selu``."""
    return g.op.Selu(op.inputs[0].name, outputs=outputs, name=op.name)


@register_tf_op_converter("LeakyRelu")
def convert_leaky_relu(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``LeakyRelu`` → ONNX ``LeakyRelu``."""
    alpha = float(op.get_attr("alpha"))
    return g.op.LeakyRelu(op.inputs[0].name, alpha=alpha, outputs=outputs, name=op.name)
