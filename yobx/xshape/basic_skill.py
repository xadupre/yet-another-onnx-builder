"""
A basic skill template for registering custom ONNX operator shape inference.

This module provides:

* :func:`register_custom_shape_function` -- a helper that inserts a shape
  inference function into the registry used by
  :class:`~yobx.xshape.shape_builder_impl.BasicShapeBuilder`.
* ``BasicSkill`` -- a minimal example of a custom operator whose output has
  the same shape and element type as its first input (a *unary* operator).

**How to create your own skill**

1. Define a shape inference function with signature
   ``(g: ShapeBuilder, node: onnx.NodeProto) -> ...``.
2. Call :func:`register_custom_shape_function` with the desired ``op_type``
   string and your function.
3. Build ONNX graphs that use nodes with that ``op_type`` (any non-standard
   domain is fine).

The ``BasicSkill`` example below is a good starting point - copy and
modify it as needed.
"""

from typing import Callable

import onnx

from .shape_builder import ShapeBuilder
from .shape_type_compute import _set_shape_type_op_any_custom, set_type_shape_unary_op

#: Domain used by the built-in ``BasicSkill`` example operator.
BASIC_SKILL_DOMAIN = "custom"

#: Operator type name of the built-in ``BasicSkill`` example operator.
BASIC_SKILL_OP_TYPE = "BasicSkill"


def register_custom_shape_function(op_type: str, fn: Callable) -> None:
    """
    Registers a shape inference function for a custom operator.

    After registration the function is called automatically by
    :class:`~yobx.xshape.shape_builder_impl.BasicShapeBuilder` whenever it
    encounters a node whose ``op_type`` matches *op_type*, regardless of the
    node's domain.

    :param op_type: ONNX operator type name (the ``op_type`` field on the node).
    :param fn: callable with signature
        ``(g: ShapeBuilder, node: onnx.NodeProto) -> ...`` that sets the
        output shapes and types on *g* for the given *node*.

    Example::

        from yobx.xshape.basic_skill import register_custom_shape_function
        from yobx.xshape.shape_type_compute import set_type_shape_unary_op

        register_custom_shape_function(
            "MyCustomRelu",
            lambda g, node: set_type_shape_unary_op(g, node.output[0], node.input[0]),
        )
    """
    _set_shape_type_op_any_custom[op_type] = fn


def _basic_skill_shape_type(g: ShapeBuilder, node: onnx.NodeProto):
    """
    Shape inference for the ``BasicSkill`` example operator.

    The single output inherits the shape and element type of the first input.
    Adapt this function to implement your own shape inference logic.

    :param g: shape builder - use it to query input shapes/types and set
        output shapes/types.
    :param node: ONNX node currently being processed.
    :return: the inferred output shape, or ``None`` if it cannot be determined.
    """
    return set_type_shape_unary_op(g, node.output[0], node.input[0])


# Register the BasicSkill operator so that BasicShapeBuilder handles it
# automatically when it appears in a model.
register_custom_shape_function(BASIC_SKILL_OP_TYPE, _basic_skill_shape_type)
