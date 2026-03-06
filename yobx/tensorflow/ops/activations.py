"""
Converters for TF activation ops:
``Relu``, ``Relu6``, ``Sigmoid``, ``Tanh``, ``Softmax``.
"""

import numpy as np
from ..register import register_tf_op_converter
from ...xbuilder import GraphBuilder


@register_tf_op_converter("Relu")
def convert_relu(g: GraphBuilder, sts: dict, outputs: list, op) -> str:
    """TF ``Relu`` → ONNX ``Relu``."""
    return g.op.Relu(op.inputs[0].name, outputs=outputs, name=op.name)


@register_tf_op_converter("Relu6")
def convert_relu6(g: GraphBuilder, sts: dict, outputs: list, op, verbose: int = 0) -> str:
    """TF ``Relu6`` → ONNX ``Clip(min=0, max=6)``."""
    return g.op.Clip(
        op.inputs[0].name,
        np.array(0.0, dtype=np.float32),
        np.array(6.0, dtype=np.float32),
        outputs=outputs[:1],
        name=op.name,
    )


@register_tf_op_converter("Sigmoid")
def convert_sigmoid(g: GraphBuilder, sts: dict, outputs: list, op) -> str:
    """TF ``Sigmoid`` → ONNX ``Sigmoid``."""
    return g.op.Sigmoid(op.inputs[0].name, outputs=outputs, name=op.name)


@register_tf_op_converter("Tanh")
def convert_tanh(g: GraphBuilder, sts: dict, outputs: list, op) -> str:
    """TF ``Tanh`` → ONNX ``Tanh``."""
    return g.op.Tanh(op.inputs[0].name, outputs=outputs, name=op.name)


@register_tf_op_converter("Softmax")
def convert_softmax(g: GraphBuilder, sts: dict, outputs: list, op, verbose: int = 0) -> str:
    """TF ``Softmax`` → ONNX ``Softmax(axis=-1)``."""
    return g.op.Softmax(op.inputs[0].name, axis=-1, outputs=outputs, name=op.name)
