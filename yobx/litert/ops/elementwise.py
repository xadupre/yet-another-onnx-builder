"""Converters for TFLite element-wise binary and unary ops."""

from typing import Any, Dict, List

from ..litert_helper import BuiltinOperator, TFLiteOperator
from ..register import register_litert_op_converter
from ...typing import GraphBuilderExtendedProtocol


@register_litert_op_converter(BuiltinOperator.ADD)
def convert_add(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``ADD`` → ONNX ``Add``."""
    return g.op.Add(op.inputs[0], op.inputs[1], outputs=outputs, name="litert_add")


@register_litert_op_converter(BuiltinOperator.SUB)
def convert_sub(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``SUB`` → ONNX ``Sub``."""
    return g.op.Sub(op.inputs[0], op.inputs[1], outputs=outputs, name="litert_sub")


@register_litert_op_converter(BuiltinOperator.MUL)
def convert_mul(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``MUL`` → ONNX ``Mul``."""
    return g.op.Mul(op.inputs[0], op.inputs[1], outputs=outputs, name="litert_mul")


@register_litert_op_converter(BuiltinOperator.DIV)
def convert_div(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``DIV`` → ONNX ``Div``."""
    return g.op.Div(op.inputs[0], op.inputs[1], outputs=outputs, name="litert_div")


@register_litert_op_converter(BuiltinOperator.NEG)
def convert_neg(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``NEG`` → ONNX ``Neg``."""
    return g.op.Neg(op.inputs[0], outputs=outputs, name="litert_neg")


@register_litert_op_converter(BuiltinOperator.ABS)
def convert_abs(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``ABS`` → ONNX ``Abs``."""
    return g.op.Abs(op.inputs[0], outputs=outputs, name="litert_abs")


@register_litert_op_converter(BuiltinOperator.FLOOR)
def convert_floor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``FLOOR`` → ONNX ``Floor``."""
    return g.op.Floor(op.inputs[0], outputs=outputs, name="litert_floor")


@register_litert_op_converter(BuiltinOperator.CEIL)
def convert_ceil(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``CEIL`` → ONNX ``Ceil``."""
    return g.op.Ceil(op.inputs[0], outputs=outputs, name="litert_ceil")


@register_litert_op_converter(BuiltinOperator.ROUND)
def convert_round(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``ROUND`` → ONNX ``Round``."""
    return g.op.Round(op.inputs[0], outputs=outputs, name="litert_round")


@register_litert_op_converter(BuiltinOperator.SQRT)
def convert_sqrt(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``SQRT`` → ONNX ``Sqrt``."""
    return g.op.Sqrt(op.inputs[0], outputs=outputs, name="litert_sqrt")


@register_litert_op_converter(BuiltinOperator.RSQRT)
def convert_rsqrt(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``RSQRT`` → ONNX ``Reciprocal(Sqrt(x))``."""
    sqrt = g.op.Sqrt(op.inputs[0], name="litert_rsqrt_sqrt")
    return g.op.Reciprocal(sqrt, outputs=outputs, name="litert_rsqrt")


@register_litert_op_converter(BuiltinOperator.EXP)
def convert_exp(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``EXP`` → ONNX ``Exp``."""
    return g.op.Exp(op.inputs[0], outputs=outputs, name="litert_exp")


@register_litert_op_converter(BuiltinOperator.LOG)
def convert_log(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``LOG`` → ONNX ``Log``."""
    return g.op.Log(op.inputs[0], outputs=outputs, name="litert_log")


@register_litert_op_converter(BuiltinOperator.SIN)
def convert_sin(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``SIN`` → ONNX ``Sin``."""
    return g.op.Sin(op.inputs[0], outputs=outputs, name="litert_sin")


@register_litert_op_converter(BuiltinOperator.POW)
def convert_pow(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``POW`` → ONNX ``Pow``."""
    return g.op.Pow(op.inputs[0], op.inputs[1], outputs=outputs, name="litert_pow")


@register_litert_op_converter(BuiltinOperator.SQUARED_DIFFERENCE)
def convert_squared_difference(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``SQUARED_DIFFERENCE`` → ONNX ``Pow(Sub(a, b), 2)``."""
    diff = g.op.Sub(op.inputs[0], op.inputs[1], name="litert_sq_diff_sub")
    return g.op.Mul(diff, diff, outputs=outputs, name="litert_sq_diff")


@register_litert_op_converter(BuiltinOperator.FLOOR_DIV)
def convert_floor_div(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``FLOOR_DIV`` → ONNX ``Floor(Div(a, b))``."""
    div = g.op.Div(op.inputs[0], op.inputs[1], name="litert_floor_div_div")
    return g.op.Floor(div, outputs=outputs, name="litert_floor_div")


@register_litert_op_converter(BuiltinOperator.LOGICAL_OR)
def convert_logical_or(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``LOGICAL_OR`` → ONNX ``Or``."""
    return g.op.Or(op.inputs[0], op.inputs[1], outputs=outputs, name="litert_or")


@register_litert_op_converter(BuiltinOperator.LOGICAL_AND)
def convert_logical_and(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``LOGICAL_AND`` → ONNX ``And``."""
    return g.op.And(op.inputs[0], op.inputs[1], outputs=outputs, name="litert_and")


@register_litert_op_converter(BuiltinOperator.LOGICAL_NOT)
def convert_logical_not(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``LOGICAL_NOT`` → ONNX ``Not``."""
    return g.op.Not(op.inputs[0], outputs=outputs, name="litert_not")
