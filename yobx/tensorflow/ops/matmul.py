"""
Converter for the TF ``MatMul`` / ``BatchMatMulV2`` op → ONNX ``MatMul``.
"""

from ..register import register_tf_op_converter
from ..tensorflow_helper import sanitize_name
from ...xbuilder import GraphBuilder


@register_tf_op_converter(("MatMul", "BatchMatMulV2", "BatchMatMul"))
def convert_matmul(g: GraphBuilder, op, ctx: dict, verbose: int = 0) -> None:
    """Converts TF ``MatMul`` / ``BatchMatMulV2`` to ONNX ``MatMul``.

    TF's transpose flags (``transpose_a`` / ``transpose_b``) are honoured by
    inserting ONNX ``Transpose`` nodes when needed.
    """
    a = ctx.get(op.inputs[0].name)
    b = ctx.get(op.inputs[1].name)
    if a is None or b is None:
        if verbose:
            print(f"[MatMul] missing input(s) for op {op.name!r}")
        return

    op_name = sanitize_name(op.name)

    # Honour optional transpose flags (present on MatMul and BatchMatMulV2).
    attr_names = {attr.name for attr in op.op_def.attr} if op.op_def else set()
    if "transpose_a" in attr_names and op.get_attr("transpose_a"):
        a = g.op.Transpose(a, perm=[-1, -2], name=f"{op_name}_tA")
    if "transpose_b" in attr_names and op.get_attr("transpose_b"):
        b = g.op.Transpose(b, perm=[-1, -2], name=f"{op_name}_tB")

    result = g.op.MatMul(a, b, name=op_name)
    assert isinstance(result, str)
    ctx[op.outputs[0].name] = result
