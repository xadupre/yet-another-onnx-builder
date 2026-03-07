"""
Converter for the TF ``MatMul`` / ``BatchMatMulV2`` op → ONNX ``MatMul``.
"""

from typing import Any, Dict, List
import tensorflow as tf
from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol


@register_tf_op_converter(("MatMul", "BatchMatMulV2", "BatchMatMul"))
def convert_matmul(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """
    Converts TF ``MatMul`` / ``BatchMatMulV2`` to ONNX ``MatMul``.

    TF's transpose flags (``transpose_a`` / ``transpose_b``) are honoured by
    inserting ONNX ``Transpose`` nodes when needed.
    """
    # Honour optional transpose flags (present on MatMul and BatchMatMulV2).
    attr_names = {attr.name for attr in op.op_def.attr} if op.op_def else set()
    a = (
        g.op.Transpose(sts[op.inputs[0].name], perm=[-1, -2], name=f"{op.name}_tA")
        if "transpose_a" in attr_names and op.get_attr("transpose_a")
        else sts[op.inputs[0].name]
    )
    b = (
        g.op.Transpose(sts[op.inputs[1].name], perm=[-1, -2], name=f"{op.name}_tB")
        if "transpose_b" in attr_names and op.get_attr("transpose_b")
        else sts[op.inputs[1].name]
    )
    return g.op.MatMul(a, b, outputs=outputs, name=op.name)
