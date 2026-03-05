"""
Converters for TF ops that act as pass-throughs or supply constant values:
``ReadVariableOp``, ``Const``, ``Identity``, and ``NoOp``.
"""

import numpy as np

from ..register import register_tf_op_converter
from ...xbuilder import GraphBuilder


@register_tf_op_converter("ReadVariableOp")
def convert_read_variable_op(
    g: GraphBuilder, op, ctx: dict, verbose: int = 0
) -> None:
    """Propagates a captured-variable value to the op's output tensor.

    In TF's graph, variables are captured as resource handles.  A
    ``ReadVariableOp`` reads the current value from such a handle.  Because
    we have already seeded ``ctx`` with the numpy values of all captured
    variables, this converter simply maps the op's output tensor name to the
    same numpy array that was stored under the input (handle) tensor name.
    """
    if not op.inputs:
        return
    src = ctx.get(op.inputs[0].name)
    if src is not None:
        ctx[op.outputs[0].name] = src


@register_tf_op_converter("Const")
def convert_const(g: GraphBuilder, op, ctx: dict, verbose: int = 0) -> None:
    """Materialises a TF ``Const`` op as a numpy array in the context.

    The constant value is extracted from the op's ``"value"`` attribute and
    converted to a numpy array.  It will be embedded as an ONNX initializer
    when first consumed by a downstream op.
    """
    try:
        import tensorflow as tf

        value = tf.constant(op.get_attr("value")).numpy()
        ctx[op.outputs[0].name] = value
    except Exception:
        if verbose:
            print(f"[Const] could not extract value for op {op.name!r}")


@register_tf_op_converter("Identity")
def convert_identity(g: GraphBuilder, op, ctx: dict, verbose: int = 0) -> None:
    """Propagates the input tensor name / value without emitting any ONNX node.

    ``Identity`` is used heavily as a graph-internal pass-through (e.g. to
    attach a name to an output).  We simply forward whatever value or ONNX name
    the input has to the output, so downstream converters can use it directly.
    """
    if not op.inputs:
        return
    src = ctx.get(op.inputs[0].name)
    if src is not None:
        ctx[op.outputs[0].name] = src


@register_tf_op_converter("NoOp")
def convert_noop(g: GraphBuilder, op, ctx: dict, verbose: int = 0) -> None:
    """``NoOp`` — nothing to emit."""
    pass
