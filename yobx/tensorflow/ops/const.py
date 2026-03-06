"""
Converters for TF ops that act as pass-throughs or supply constant values:
``ReadVariableOp``, ``Const``, ``Identity``, and ``NoOp``.
"""

import numpy as np
import tensorflow as tf
from ..register import register_tf_op_converter
from ...xbuilder import GraphBuilder


@register_tf_op_converter("ReadVariableOp")
def convert_read_variable_op(g: GraphBuilder, sts: dict, outputs: list, op) -> str:
    """Propagates a captured-variable value to the op's output tensor.

    In TF's graph, variables are captured as resource handles.  A
    ``ReadVariableOp`` reads the current value from such a handle.  Because
    we have already seeded ``sts`` with the numpy values of all captured
    variables, this converter simply maps the op's output tensor name to the
    same numpy array that was stored under the input (handle) tensor name.
    """
    return g.op.Identity(op.inputs[0].name, outputs=outputs, name=op.name)


@register_tf_op_converter("Const")
def convert_const(g: GraphBuilder, sts: dict, outputs: list, op) -> str:
    """Materialises a TF ``Const`` op as a numpy array in the context.

    The constant value is extracted from the op's ``"value"`` attribute and
    converted to a numpy array.  It will be embedded as an ONNX initializer
    when first consumed by a downstream op.
    """
    value = tf.constant(op.get_attr("value")).numpy()
    return g.op.Identity(value, outputs=outputs, name=op.name)


@register_tf_op_converter("Identity")
def convert_identity(g: GraphBuilder, sts: dict, outputs: list, op) -> str:
    """Propagates the input tensor name / value without emitting any ONNX node.

    ``Identity`` is used heavily as a graph-internal pass-through (e.g. to
    attach a name to an output).  We simply forward whatever value or ONNX name
    the input has to the output, so downstream converters can use it directly.
    """
    return g.op.Identity(op.inputs[0].name, outputs=outputs, name=op.name)


@register_tf_op_converter("NoOp")
def convert_noop(g: GraphBuilder, sts: dict, outputs: list, op) -> None:
    """``NoOp`` — nothing to emit."""
    assert (
        not op.inputs and not op.outputs
    ), f"This operator does something {op=}{g.get_debug_msg()}"
