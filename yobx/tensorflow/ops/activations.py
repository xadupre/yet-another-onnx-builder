"""
Converters for TF activation ops:
``Relu``, ``Relu6``, ``Sigmoid``, ``Tanh``, ``Softmax``.
"""

from ..register import register_tf_op_converter
from ..tensorflow_helper import sanitize_name
from ...xbuilder import GraphBuilder


@register_tf_op_converter("Relu")
def convert_relu(
    g: GraphBuilder, sts: dict, outputs: list, op, verbose: int = 0
) -> None:
    """TF ``Relu`` → ONNX ``Relu``."""
    a = sts.get(op.inputs[0].name)
    if a is None:
        return
    result = g.op.Relu(a, outputs=outputs[:1], name=sanitize_name(op.name))
    assert isinstance(result, str)
    sts[op.outputs[0].name] = result


@register_tf_op_converter("Relu6")
def convert_relu6(
    g: GraphBuilder, sts: dict, outputs: list, op, verbose: int = 0
) -> None:
    """TF ``Relu6`` → ONNX ``Clip(min=0, max=6)``."""
    a = sts.get(op.inputs[0].name)
    if a is None:
        return
    import numpy as np

    op_name = sanitize_name(op.name)
    result = g.op.Clip(
        a,
        np.array(0.0, dtype=np.float32),
        np.array(6.0, dtype=np.float32),
        outputs=outputs[:1],
        name=op_name,
    )
    assert isinstance(result, str)
    sts[op.outputs[0].name] = result


@register_tf_op_converter("Sigmoid")
def convert_sigmoid(
    g: GraphBuilder, sts: dict, outputs: list, op, verbose: int = 0
) -> None:
    """TF ``Sigmoid`` → ONNX ``Sigmoid``."""
    a = sts.get(op.inputs[0].name)
    if a is None:
        return
    result = g.op.Sigmoid(a, outputs=outputs[:1], name=sanitize_name(op.name))
    assert isinstance(result, str)
    sts[op.outputs[0].name] = result


@register_tf_op_converter("Tanh")
def convert_tanh(
    g: GraphBuilder, sts: dict, outputs: list, op, verbose: int = 0
) -> None:
    """TF ``Tanh`` → ONNX ``Tanh``."""
    a = sts.get(op.inputs[0].name)
    if a is None:
        return
    result = g.op.Tanh(a, outputs=outputs[:1], name=sanitize_name(op.name))
    assert isinstance(result, str)
    sts[op.outputs[0].name] = result


@register_tf_op_converter("Softmax")
def convert_softmax(
    g: GraphBuilder, sts: dict, outputs: list, op, verbose: int = 0
) -> None:
    """TF ``Softmax`` → ONNX ``Softmax(axis=-1)``."""
    a = sts.get(op.inputs[0].name)
    if a is None:
        return
    result = g.op.Softmax(a, axis=-1, outputs=outputs[:1], name=sanitize_name(op.name))
    assert isinstance(result, str)
    sts[op.outputs[0].name] = result
