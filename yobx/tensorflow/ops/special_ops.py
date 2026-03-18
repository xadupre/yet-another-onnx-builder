from typing import Any, Dict, List
import tensorflow as tf
from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol


@register_tf_op_converter("PreventGradient")
def convert_exp(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """PreventGradient -> Identity"""
    return g.op.Identity(op.inputs[0].name, outputs=outputs[:1], name=op.name)
