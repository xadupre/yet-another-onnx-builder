"""
Converters for TF binary (element-wise) ops.

Arithmetic
----------
``Sub``, ``Mul``, ``RealDiv``, ``FloorDiv``, ``Minimum``, ``Maximum``,
``Pow``, ``SquaredDifference``, ``FloorMod``, ``TruncateMod``

Comparison
----------
``Equal``, ``NotEqual``, ``Greater``, ``GreaterEqual``, ``Less``,
``LessEqual``

Logical
-------
``LogicalAnd``, ``LogicalNot``, ``LogicalOr``, ``LogicalXor``
"""

from typing import Any, Dict, List
import tensorflow as tf
from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


@register_tf_op_converter("Sub")
def convert_sub(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Sub`` → ONNX ``Sub``."""
    return g.op.Sub(op.inputs[0].name, op.inputs[1].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Mul")
def convert_mul(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Mul`` → ONNX ``Mul``."""
    return g.op.Mul(op.inputs[0].name, op.inputs[1].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("RealDiv")
def convert_real_div(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``RealDiv`` → ONNX ``Div``."""
    return g.op.Div(op.inputs[0].name, op.inputs[1].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("FloorDiv")
def convert_floor_div(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``FloorDiv`` → ONNX ``Floor(Div(a, b))``."""
    div = g.op.Div(op.inputs[0].name, op.inputs[1].name, name=f"{op.name}_div")
    return g.op.Floor(div, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Minimum")
def convert_minimum(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Minimum`` → ONNX ``Min``."""
    return g.op.Min(
        op.inputs[0].name, op.inputs[1].name, outputs=outputs[:1], name=op.name
    )


@register_tf_op_converter("Maximum")
def convert_maximum(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Maximum`` → ONNX ``Max``."""
    return g.op.Max(
        op.inputs[0].name, op.inputs[1].name, outputs=outputs[:1], name=op.name
    )


@register_tf_op_converter("Pow")
def convert_pow(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Pow`` → ONNX ``Pow``."""
    return g.op.Pow(op.inputs[0].name, op.inputs[1].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("SquaredDifference")
def convert_squared_difference(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``SquaredDifference`` → ONNX ``Mul(Sub(a, b), Sub(a, b))``."""
    diff = g.op.Sub(op.inputs[0].name, op.inputs[1].name, name=f"{op.name}_sub")
    return g.op.Mul(diff, diff, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("FloorMod")
def convert_floor_mod(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``FloorMod`` → ONNX ``Mod(fmod=0)``."""
    return g.op.Mod(
        op.inputs[0].name, op.inputs[1].name, fmod=0, outputs=outputs[:1], name=op.name
    )


@register_tf_op_converter("TruncateMod")
def convert_truncate_mod(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``TruncateMod`` → ONNX ``Mod(fmod=1)``."""
    return g.op.Mod(
        op.inputs[0].name, op.inputs[1].name, fmod=1, outputs=outputs[:1], name=op.name
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@register_tf_op_converter("Equal")
def convert_equal(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Equal`` → ONNX ``Equal``."""
    return g.op.Equal(op.inputs[0].name, op.inputs[1].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("NotEqual")
def convert_not_equal(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``NotEqual`` → ONNX ``Not(Equal(a, b))``."""
    eq = g.op.Equal(op.inputs[0].name, op.inputs[1].name, name=f"{op.name}_eq")
    return g.op.Not(eq, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("Greater")
def convert_greater(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Greater`` → ONNX ``Greater``."""
    return g.op.Greater(
        op.inputs[0].name, op.inputs[1].name, outputs=outputs[:1], name=op.name
    )


@register_tf_op_converter("GreaterEqual")
def convert_greater_equal(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``GreaterEqual`` → ONNX ``GreaterOrEqual``."""
    return g.op.GreaterOrEqual(
        op.inputs[0].name, op.inputs[1].name, outputs=outputs[:1], name=op.name
    )


@register_tf_op_converter("Less")
def convert_less(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``Less`` → ONNX ``Less``."""
    return g.op.Less(op.inputs[0].name, op.inputs[1].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("LessEqual")
def convert_less_equal(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``LessEqual`` → ONNX ``LessOrEqual``."""
    return g.op.LessOrEqual(
        op.inputs[0].name, op.inputs[1].name, outputs=outputs[:1], name=op.name
    )


# ---------------------------------------------------------------------------
# Logical
# ---------------------------------------------------------------------------


@register_tf_op_converter("LogicalAnd")
def convert_logical_and(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``LogicalAnd`` → ONNX ``And``."""
    return g.op.And(op.inputs[0].name, op.inputs[1].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("LogicalNot")
def convert_logical_not(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``LogicalNot`` → ONNX ``Not``."""
    return g.op.Not(op.inputs[0].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("LogicalOr")
def convert_logical_or(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``LogicalOr`` → ONNX ``Or``."""
    return g.op.Or(op.inputs[0].name, op.inputs[1].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("LogicalXor")
def convert_logical_xor(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """TF ``LogicalXor`` → ONNX ``Xor``."""
    return g.op.Xor(op.inputs[0].name, op.inputs[1].name, outputs=outputs[:1], name=op.name)
