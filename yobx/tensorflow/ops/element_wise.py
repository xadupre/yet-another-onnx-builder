"""
Converters for element-wise TF ops: ``AddV2``, ``BiasAdd``, ``ConcatV2``.

Addition / bias
---------------
``AddV2``, ``BiasAdd``

Concatenation
-------------
``ConcatV2``
"""

from typing import Any, Dict, List
import tensorflow as tf
from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol


@register_tf_op_converter(("AddV2", "BiasAdd"))
def convert_element_wise(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """
    Converts TF element-wise addition ops (``AddV2``, ``BiasAdd``) to ONNX ``Add``.

    TF's ``BiasAdd`` is semantically equivalent to adding a 1-D bias along the
    last dimension of the input, which maps directly to ONNX ``Add`` with
    numpy-style broadcasting.
    """
    return g.op.Add(op.inputs[0].name, op.inputs[1].name, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("ConcatV2")
def convert_concat_v2(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """
    Converts TF ``ConcatV2`` → ONNX ``Concat``.

    TF ``ConcatV2`` takes *N* tensor inputs followed by a scalar ``axis``
    constant as the last input.  The axis value is read from that constant
    and forwarded as the ``axis`` attribute of the ONNX ``Concat`` node.
    """
    # The last input is the axis scalar constant.
    axis_op = op.inputs[-1].op
    axis = int(tf.make_ndarray(axis_op.get_attr("value")))
    tensor_inputs = [inp.name for inp in op.inputs[:-1]]
    return g.op.Concat(*tensor_inputs, axis=axis, outputs=outputs[:1], name=op.name)
