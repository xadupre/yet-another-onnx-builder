"""
Converters for TF shape-manipulation ops: ``Reshape``, ``Squeeze``.
"""

from typing import Any, Dict, List

import numpy as np
from onnx import TensorProto
import tensorflow as tf

from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol


@register_tf_op_converter("Reshape")
def convert_reshape(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: tf.Operation,
) -> str:
    """
    Converts TF ``Reshape`` → ONNX ``Reshape``.

    The TF ``shape`` input (second input) may be int32; it is cast to int64
    as required by the ONNX ``Reshape`` specification.
    """
    shape_i64 = g.op.Cast(
        op.inputs[1].name, to=TensorProto.INT64, name=f"{op.name}_shape_cast"
    )
    return g.op.Reshape(
        op.inputs[0].name, shape_i64, outputs=outputs[:1], name=op.name
    )


@register_tf_op_converter("Squeeze")
def convert_squeeze(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: tf.Operation,
) -> str:
    """
    Converts TF ``Squeeze`` → ONNX ``Squeeze``.

    When ``squeeze_dims`` is empty all size-1 dimensions are removed (matching
    TF's default behaviour).  When specific axes are given they are passed as
    an int64 tensor input to the ONNX node (opset ≥ 13 API).
    """
    squeeze_dims = list(op.get_attr("squeeze_dims"))
    if squeeze_dims:
        axes = np.array(squeeze_dims, dtype=np.int64)
        return g.op.Squeeze(
            op.inputs[0].name, axes, outputs=outputs[:1], name=op.name
        )
    return g.op.Squeeze(op.inputs[0].name, outputs=outputs[:1], name=op.name)
