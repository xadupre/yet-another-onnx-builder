"""
Converter for the TF ``MatMul`` / ``BatchMatMulV2`` op → ONNX ``MatMul``.
"""

from typing import Any, Dict, List
import numpy as np
import tensorflow as tf
from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol


def _transpose_last_two(
    g: GraphBuilderExtendedProtocol, name: str, inp: tf.Tensor, tag: str
) -> str:
    """Insert an ONNX Transpose that swaps the last two dimensions of *inp*."""
    rank = len(inp.shape)
    if rank < 2:
        raise ValueError(
            f"Cannot transpose last two dims of a tensor with rank {rank}."
        )
    perm = list(range(rank))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return g.op.Transpose(inp.name, perm=perm, name=f"{name}_{tag}")


@register_tf_op_converter(("MatMul", "BatchMatMulV2", "BatchMatMul"))
def convert_matmul(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """
    Converts TF ``MatMul`` / ``BatchMatMulV2`` to ONNX ``MatMul``.

    TF's transpose flags (``transpose_a`` / ``transpose_b``) and adjoint flags
    (``adj_x`` / ``adj_y`` used by ``BatchMatMulV2``) are honoured by inserting
    ONNX ``Transpose`` nodes when needed.  For real-valued tensors, adjoint is
    equivalent to transpose.
    """
    attr_names = {attr.name for attr in op.op_def.attr} if op.op_def else set()

    def _is_set(attr_a: str, attr_b: str) -> bool:
        """Return True when either attribute alias is present and truthy."""
        for attr in (attr_a, attr_b):
            if attr in attr_names:
                try:
                    if op.get_attr(attr):
                        return True
                except (ValueError, tf.errors.NotFoundError):
                    pass
        return False

    a = (
        _transpose_last_two(g, op.name, op.inputs[0], "tA")
        if _is_set("transpose_a", "adj_x")
        else op.inputs[0].name
    )
    b = (
        _transpose_last_two(g, op.name, op.inputs[1], "tB")
        if _is_set("transpose_b", "adj_y")
        else op.inputs[1].name
    )
    return g.op.MatMul(a, b, outputs=outputs, name=op.name)
