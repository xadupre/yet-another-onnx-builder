"""
Converters for TF random-sampling ops.

Random normal
-------------
``RandomStandardNormal``
"""

from typing import Any, Dict, List

from onnx import TensorProto
import onnx.helper as onh
import tensorflow as tf

from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol

# Mapping from TF dtype enum to ONNX TensorProto type.
_TF_DTYPE_TO_ONNX = {
    tf.float16.as_datatype_enum: TensorProto.FLOAT16,
    tf.float32.as_datatype_enum: TensorProto.FLOAT,
    tf.float64.as_datatype_enum: TensorProto.DOUBLE,
}


@register_tf_op_converter("RandomStandardNormal")
def convert_random_standard_normal(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """
    Converts TF ``RandomStandardNormal`` → ONNX ``ConstantOfShape`` +
    ``RandomNormalLike``.

    TF ``RandomStandardNormal`` draws samples from a standard normal
    distribution given a dynamic *shape* tensor.  ONNX does not have an op
    that accepts a dynamic shape for random generation, but ``RandomNormalLike``
    generates standard-normal values that match the shape and dtype of its
    input tensor.  A zero-filled ``ConstantOfShape`` is therefore first
    created with the requested shape, and then ``RandomNormalLike`` is
    applied to it.

    The output dtype is taken from the TF op's ``dtype`` attribute.  The
    output shape is inferred from the TF op's output shape metadata (which
    always contains the correct rank even when individual dimensions are
    unknown) and recorded in the graph builder so that downstream shape
    assertions are satisfied.
    """
    shape_input = op.inputs[0].name

    # Ensure the shape tensor is int64 (ConstantOfShape requires int64).
    shape_i64 = g.op.Cast(shape_input, to=TensorProto.INT64, name=f"{op.name}_shape_cast")

    # Create a zero-filled tensor with the requested dtype and shape.
    tf_dtype = op.get_attr("dtype").as_datatype_enum
    onnx_dtype = _TF_DTYPE_TO_ONNX.get(tf_dtype, TensorProto.FLOAT)
    zero_tensor = onh.make_tensor("value", onnx_dtype, [1], [0.0])
    zeros = g.op.ConstantOfShape(shape_i64, value=zero_tensor, name=f"{op.name}_zeros")

    # Draw standard-normal samples with matching shape and dtype.
    result = g.op.RandomNormalLike(zeros, outputs=outputs[:1], name=op.name)

    # Propagate shape/type to satisfy the framework's post-converter assertions.
    # TF always knows the rank of the output even if individual dims are dynamic.
    if not g.has_shape(outputs[0]):
        tf_dims = op.outputs[0].shape.as_list()
        if tf_dims is not None:
            # Replace None (dynamic) dims with symbolic names.
            onnx_shape = tuple(d if d is not None else f"d{i}" for i, d in enumerate(tf_dims))
            g.set_shape(outputs[0], onnx_shape)
            g.set_type(outputs[0], onnx_dtype)

    return result
