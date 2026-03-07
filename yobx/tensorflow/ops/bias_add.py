"""
Converter for the TF ``BiasAdd`` op → ONNX ``Add``.
"""

from typing import Any, Dict, List
from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol


@register_tf_op_converter("BiasAdd")
def convert_bias_add(g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op) -> str:
    """Converts TF ``BiasAdd`` to ONNX ``Add``.

    TF's ``BiasAdd`` is semantically equivalent to adding a 1-D bias along the
    last dimension of the input, which maps directly to ONNX ``Add`` with
    numpy-style broadcasting.
    """
    return g.op.Add(op.inputs[0].name, op.inputs[1].name, outputs=outputs[:1], name=op.name)
