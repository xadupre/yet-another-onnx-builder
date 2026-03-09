"""
Converters for TF unary (element-wise) math ops.

Exponential / Logarithm
-----------------------
``Exp``, ``Log``

Trigonometric
-------------
``Cos``, ``Sin``, ``Tan``, ``Acos``, ``Asin``, ``Atan``

Hyperbolic
----------
``Cosh``, ``Sinh``

Rounding / Magnitude
--------------------
``Abs``, ``Neg``, ``Sign``, ``Floor``, ``Ceil``, ``Round``

Square-root family
------------------
``Sqrt``, ``Rsqrt``, ``Square``, ``Reciprocal``
"""

from typing import Any, Dict, List
import tensorflow as tf
from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol

# ---------------------------------------------------------------------------
# Exponential / Logarithm
# ---------------------------------------------------------------------------


@register_tf_op_converter("Exp")
def convert_exp(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Exp`` → ONNX ``Exp``."""
    return g.op.Exp(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Log")
def convert_log(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Log`` → ONNX ``Log``."""
    return g.op.Log(op.inputs[0].name, outputs=outputs[:1], name=op.name)


# ---------------------------------------------------------------------------
# Trigonometric
# ---------------------------------------------------------------------------


@register_tf_op_converter("Cos")
def convert_cos(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Cos`` → ONNX ``Cos``."""
    return g.op.Cos(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Sin")
def convert_sin(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Sin`` → ONNX ``Sin``."""
    return g.op.Sin(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Tan")
def convert_tan(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Tan`` → ONNX ``Tan``."""
    return g.op.Tan(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Acos")
def convert_acos(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Acos`` → ONNX ``Acos``."""
    return g.op.Acos(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Asin")
def convert_asin(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Asin`` → ONNX ``Asin``."""
    return g.op.Asin(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Atan")
def convert_atan(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Atan`` → ONNX ``Atan``."""
    return g.op.Atan(op.inputs[0].name, outputs=outputs[:1], name=op.name)


# ---------------------------------------------------------------------------
# Hyperbolic
# ---------------------------------------------------------------------------


@register_tf_op_converter("Cosh")
def convert_cosh(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Cosh`` → ONNX ``Cosh``."""
    return g.op.Cosh(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Sinh")
def convert_sinh(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Sinh`` → ONNX ``Sinh``."""
    return g.op.Sinh(op.inputs[0].name, outputs=outputs[:1], name=op.name)


# ---------------------------------------------------------------------------
# Rounding / Magnitude
# ---------------------------------------------------------------------------


@register_tf_op_converter("Abs")
def convert_abs(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Abs`` → ONNX ``Abs``."""
    return g.op.Abs(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Neg")
def convert_neg(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Neg`` → ONNX ``Neg``."""
    return g.op.Neg(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Sign")
def convert_sign(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Sign`` → ONNX ``Sign``."""
    return g.op.Sign(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Floor")
def convert_floor(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Floor`` → ONNX ``Floor``."""
    return g.op.Floor(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Ceil")
def convert_ceil(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Ceil`` → ONNX ``Ceil``."""
    return g.op.Ceil(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Round")
def convert_round(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Round`` → ONNX ``Round``."""
    return g.op.Round(op.inputs[0].name, outputs=outputs[:1], name=op.name)


# ---------------------------------------------------------------------------
# Square-root family
# ---------------------------------------------------------------------------


@register_tf_op_converter("Sqrt")
def convert_sqrt(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Sqrt`` → ONNX ``Sqrt``."""
    return g.op.Sqrt(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Rsqrt")
def convert_rsqrt(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Rsqrt`` → ONNX ``Reciprocal(Sqrt(x))``."""
    sqrt = g.op.Sqrt(op.inputs[0].name, name=f"{op.name}_sqrt")
    return g.op.Reciprocal(sqrt, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Square")
def convert_square(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Square`` → ONNX ``Mul(x, x)``."""
    x = op.inputs[0].name
    return g.op.Mul(x, x, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Reciprocal")
def convert_reciprocal(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Reciprocal`` → ONNX ``Reciprocal``."""
    return g.op.Reciprocal(op.inputs[0].name, outputs=outputs[:1], name=op.name)
