"""
Converters for TF padding ops: ``Pad``, ``PadV2``, ``MirrorPad``.

TensorFlow ``Pad`` / ``PadV2`` store padding amounts as a 2-D tensor of shape
``[N, 2]`` with layout ``[[begin0, end0], [begin1, end1], ...]``.
ONNX ``Pad`` (opset ≥ 11) expects a 1-D int64 tensor of shape ``[2N]`` with
layout ``[begin0, begin1, ..., beginN-1, end0, end1, ..., endN-1]``.

The converters transpose the TF paddings tensor from ``[N, 2]`` to ``[2, N]``
and then flatten it to produce the ONNX-compatible layout.
"""

from typing import Any, Dict, List

import numpy as np
from onnx import TensorProto
import tensorflow as tf

from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol


def _reorder_paddings(g: GraphBuilderExtendedProtocol, paddings_name: str, op_name: str) -> str:
    """
    Reorders a TF paddings tensor from ``[N, 2]`` (TF layout) to ``[2N]``
    (ONNX layout) and casts the result to int64.

    * Transpose ``[N, 2]`` → ``[2, N]``
    * Reshape to ``[2N]``
    * Cast to int64 (ONNX ``Pad`` requires int64 pads)
    """
    pads_t = g.op.Transpose(paddings_name, perm=[1, 0], name=f"{op_name}_pads_t")
    pads_flat = g.op.Reshape(
        pads_t,
        np.array([-1], dtype=np.int64),
        name=f"{op_name}_pads_flat",
    )
    return g.op.Cast(pads_flat, to=TensorProto.INT64, name=f"{op_name}_pads_i64")


@register_tf_op_converter(("Pad", "PadV2"))
def convert_pad(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: tf.Operation,
) -> str:
    """
    Converts TF ``Pad`` / ``PadV2`` → ONNX ``Pad``.

    ``PadV2`` carries an explicit ``constant_values`` input (third input) which
    is forwarded to the ONNX ``Pad`` node as the optional constant value.
    ``Pad`` pads with zeros and does not need that input.
    """
    data = op.inputs[0].name
    pads_i64 = _reorder_paddings(g, op.inputs[1].name, op.name)

    if op.type == "PadV2":
        constant_value = op.inputs[2].name
        return g.op.Pad(data, pads_i64, constant_value, outputs=outputs[:1], name=op.name)
    return g.op.Pad(data, pads_i64, outputs=outputs[:1], name=op.name)


@register_tf_op_converter("MirrorPad")
def convert_mirror_pad(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: tf.Operation,
) -> str:
    """
    Converts TF ``MirrorPad`` → ONNX ``Pad`` with ``mode`` attribute.

    TF ``mode`` attribute values:

    * ``"REFLECT"``   → ONNX ``mode="reflect"`` (excludes the border value)
    * ``"SYMMETRIC"`` → ONNX ``mode="edge"`` (includes the border value)
    """
    mode_raw = op.get_attr("mode")
    mode_str = mode_raw.decode() if isinstance(mode_raw, bytes) else mode_raw
    onnx_mode = "reflect" if mode_str == "REFLECT" else "edge"

    data = op.inputs[0].name
    pads_i64 = _reorder_paddings(g, op.inputs[1].name, op.name)
    return g.op.Pad(data, pads_i64, mode=onnx_mode, outputs=outputs[:1], name=op.name)
