"""
Converters for TF unary (element-wise) math ops.

Exponential / Logarithm
-----------------------
``Exp``, ``Log``, ``Log1p``, ``Expm1``

Trigonometric
-------------
``Cos``, ``Sin``, ``Tan``, ``Acos``, ``Asin``, ``Atan``

Hyperbolic
----------
``Cosh``, ``Sinh``, ``Acosh``, ``Asinh``, ``Atanh``

Special functions
-----------------
``Erf``, ``Erfc``, ``Softplus``, ``Softsign``

Rounding / Magnitude
--------------------
``Abs``, ``Neg``, ``Sign``, ``Floor``, ``Ceil``, ``Round``, ``Rint``

Square-root family
------------------
``Sqrt``, ``Rsqrt``, ``Square``, ``Reciprocal``
"""

from typing import Any, Dict, List
import numpy as np
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


@register_tf_op_converter("Log1p")
def convert_log1p(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Log1p`` → ONNX ``Log(Add(x, 1))``."""
    x = op.inputs[0].name
    dtype = op.inputs[0].dtype.as_numpy_dtype
    one = np.array(1, dtype=dtype)
    xp1 = g.op.Add(x, one, name=f"{op.name}_add")
    return g.op.Log(xp1, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Expm1")
def convert_expm1(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Expm1`` → ONNX ``Sub(Exp(x), 1)``."""
    x = op.inputs[0].name
    dtype = op.inputs[0].dtype.as_numpy_dtype
    one = np.array(1, dtype=dtype)
    exp_x = g.op.Exp(x, name=f"{op.name}_exp")
    return g.op.Sub(exp_x, one, outputs=outputs[:1], name=op.name)


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


@register_tf_op_converter("Acosh")
def convert_acosh(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Acosh`` → ONNX ``Acosh``."""
    return g.op.Acosh(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Asinh")
def convert_asinh(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Asinh`` → ONNX ``Asinh``."""
    return g.op.Asinh(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Atanh")
def convert_atanh(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Atanh`` → ONNX ``Atanh``."""
    return g.op.Atanh(op.inputs[0].name, outputs=outputs[:1], name=op.name)


# ---------------------------------------------------------------------------
# Special functions
# ---------------------------------------------------------------------------


@register_tf_op_converter("Erf")
def convert_erf(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Erf`` → ONNX ``Erf``."""
    return g.op.Erf(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Erfc")
def convert_erfc(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Erfc`` → ONNX ``Sub(1, Erf(x))``."""
    x = op.inputs[0].name
    dtype = op.inputs[0].dtype.as_numpy_dtype
    one = np.array(1, dtype=dtype)
    erf_x = g.op.Erf(x, name=f"{op.name}_erf")
    return g.op.Sub(one, erf_x, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Softplus")
def convert_softplus(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Softplus`` → ONNX ``Softplus``."""
    return g.op.Softplus(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Softsign")
def convert_softsign(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Softsign`` → ONNX ``Softsign``."""
    return g.op.Softsign(op.inputs[0].name, outputs=outputs[:1], name=op.name)


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


@register_tf_op_converter("Rint")
def convert_rint(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Rint`` → ONNX ``Round`` (both use round-half-to-even)."""
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
