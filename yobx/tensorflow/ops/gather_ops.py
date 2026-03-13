"""
Converter for TF ``GatherV2`` op → ONNX ``Gather``.

Indexing / lookup
-----------------
``GatherV2``
"""

from typing import Any, Dict, List

import tensorflow as tf

from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol


@register_tf_op_converter("GatherV2")
def convert_gather_v2(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: tf.Operation,
) -> str:
    """
    Converts TF ``GatherV2`` (``tf.gather``) → ONNX ``Gather``.

    TF ``GatherV2`` takes three inputs: ``params``, ``indices``, and a scalar
    ``axis`` constant.  The axis value is read from that constant and forwarded
    as the ``axis`` attribute of the ONNX ``Gather`` node.
    """
    axis_op = op.inputs[2].op
    axis = int(tf.make_ndarray(axis_op.get_attr("value")))
    return g.op.Gather(
        op.inputs[0].name,
        op.inputs[1].name,
        axis=axis,
        outputs=outputs[:1],
        name=op.name,
    )
