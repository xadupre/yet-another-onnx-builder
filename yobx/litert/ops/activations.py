"""Converters for TFLite activation ops: RELU, TANH, LOGISTIC, ELU, LEAKY_RELU."""

from typing import Any, Dict, List

import numpy as np

from ..litert_helper import BuiltinOperator, TFLiteOperator
from ..register import register_litert_op_converter
from ...typing import GraphBuilderExtendedProtocol


@register_litert_op_converter(BuiltinOperator.RELU)
def convert_relu(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``RELU`` → ONNX ``Relu``."""
    return g.op.Relu(op.inputs[0], outputs=outputs, name="litert_relu")


@register_litert_op_converter(BuiltinOperator.RELU_N1_TO_1)
def convert_relu_n1_to1(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``RELU_N1_TO_1`` → ONNX ``Clip(min=-1, max=1)``."""
    return g.op.Clip(
        op.inputs[0],
        np.array(-1.0, dtype=np.float32),
        np.array(1.0, dtype=np.float32),
        outputs=outputs,
        name="litert_relu_n1_to1",
    )


@register_litert_op_converter(BuiltinOperator.TANH)
def convert_tanh(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``TANH`` → ONNX ``Tanh``."""
    return g.op.Tanh(op.inputs[0], outputs=outputs, name="litert_tanh")


@register_litert_op_converter(BuiltinOperator.SOFTMAX)
def convert_softmax(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``SOFTMAX`` → ONNX ``Softmax(axis=-1)``."""
    return g.op.Softmax(op.inputs[0], axis=-1, outputs=outputs, name="litert_softmax")


@register_litert_op_converter(BuiltinOperator.LOG_SOFTMAX)
def convert_log_softmax(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``LOG_SOFTMAX`` → ONNX ``LogSoftmax(axis=-1)``."""
    return g.op.LogSoftmax(op.inputs[0], axis=-1, outputs=outputs, name="litert_log_softmax")


@register_litert_op_converter(BuiltinOperator.ELU)
def convert_elu(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``ELU`` → ONNX ``Elu(alpha=1.0)``."""
    return g.op.Elu(op.inputs[0], alpha=1.0, outputs=outputs, name="litert_elu")


@register_litert_op_converter(BuiltinOperator.LEAKY_RELU)
def convert_leaky_relu(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``LEAKY_RELU`` → ONNX ``LeakyRelu``."""
    alpha = float(op.builtin_options.get("alpha", 0.2))
    return g.op.LeakyRelu(op.inputs[0], alpha=alpha, outputs=outputs, name="litert_leaky_relu")


@register_litert_op_converter(BuiltinOperator.HARD_SWISH)
def convert_hard_swish(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``HARD_SWISH`` → ONNX ``HardSwish``."""
    return g.op.HardSwish(op.inputs[0], outputs=outputs, name="litert_hard_swish")


@register_litert_op_converter(BuiltinOperator.GELU)
def convert_gelu(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``GELU`` → ONNX ``Gelu``."""
    return g.op.Gelu(op.inputs[0], outputs=outputs, name="litert_gelu")
