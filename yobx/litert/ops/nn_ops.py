"""Converters for TFLite neural network ops: FULLY_CONNECTED, CONV_2D,
DEPTHWISE_CONV_2D, AVERAGE_POOL_2D, MAX_POOL_2D, BATCH_MATMUL."""

from typing import Any, Dict, List

import numpy as np

from ..litert_helper import BuiltinOperator, Padding, TFLiteOperator
from ..register import register_litert_op_converter
from ...typing import GraphBuilderExtendedProtocol


def _padding_to_auto_pad(padding: int) -> str:
    """Map TFLite ``Padding`` enum to ONNX ``auto_pad`` string."""
    if padding == Padding.SAME:
        return "SAME_UPPER"
    return "VALID"


@register_litert_op_converter(BuiltinOperator.FULLY_CONNECTED)
def convert_fully_connected(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``FULLY_CONNECTED`` → ONNX ``MatMul`` (+ optional ``Add`` bias).

    TFLite stores the weight matrix in (out_features, in_features) order
    so we transpose it to get the standard ONNX layout.
    """
    x = op.inputs[0]
    w = op.inputs[1]
    # Transpose weights: TFLite is (out, in), ONNX MatMul expects (in, out).
    wT = g.op.Transpose(w, perm=[1, 0], name="litert_fc_wT")
    matmul = g.op.MatMul(x, wT, name="litert_fc_matmul")

    if len(op.inputs) > 2 and op.inputs[2] >= 0:
        bias = op.inputs[2]
        result = g.op.Add(matmul, bias, outputs=outputs, name="litert_fc_bias")
    else:
        result = g.op.Identity(matmul, outputs=outputs, name="litert_fc_out")

    return result


@register_litert_op_converter(BuiltinOperator.BATCH_MATMUL)
def convert_batch_matmul(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``BATCH_MATMUL`` → ONNX ``MatMul`` with optional transposes."""
    x = op.inputs[0]
    y = op.inputs[1]
    adj_x = op.builtin_options.get("adj_x", False)
    adj_y = op.builtin_options.get("adj_y", False)

    if adj_x:
        x = g.op.Transpose(x, perm=[0, 2, 1], name="litert_bmm_xT")
    if adj_y:
        y = g.op.Transpose(y, perm=[0, 2, 1], name="litert_bmm_yT")

    return g.op.MatMul(x, y, outputs=outputs, name="litert_batch_matmul")


@register_litert_op_converter(BuiltinOperator.CONV_2D)
def convert_conv2d(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``CONV_2D`` → ONNX ``Conv``.

    TFLite weight layout: ``(out_ch, kH, kW, in_ch)`` — convert to ONNX
    ``(out_ch, in_ch, kH, kW)`` via a ``Transpose``.
    TFLite input/output layout: ``NHWC`` — wrapped with ``Transpose``
    nodes to present as ``NCHW`` to the ONNX ``Conv`` node.
    """
    opts = op.builtin_options
    padding = opts.get("padding", Padding.VALID)
    stride_w = opts.get("stride_w", 1)
    stride_h = opts.get("stride_h", 1)
    dilation_w = opts.get("dilation_w_factor", 1)
    dilation_h = opts.get("dilation_h_factor", 1)

    # NHWC → NCHW
    x_nchw = g.op.Transpose(op.inputs[0], perm=[0, 3, 1, 2], name="litert_conv_x_nhwc2nchw")
    # Weight: (out, kH, kW, in) → (out, in, kH, kW)
    w_oikk = g.op.Transpose(op.inputs[1], perm=[0, 3, 1, 2], name="litert_conv_w_oihw")

    conv_kwargs: Dict[str, Any] = dict(
        strides=[stride_h, stride_w],
        dilations=[dilation_h, dilation_w],
        auto_pad=_padding_to_auto_pad(padding),
        name="litert_conv",
    )

    if len(op.inputs) > 2 and op.inputs[2] >= 0:
        conv_nchw = g.op.Conv(x_nchw, w_oikk, op.inputs[2], **conv_kwargs)
    else:
        conv_nchw = g.op.Conv(x_nchw, w_oikk, **conv_kwargs)

    # NCHW → NHWC
    return g.op.Transpose(conv_nchw, perm=[0, 2, 3, 1], outputs=outputs, name="litert_conv_out")


@register_litert_op_converter(BuiltinOperator.DEPTHWISE_CONV_2D)
def convert_depthwise_conv2d(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``DEPTHWISE_CONV_2D`` → ONNX ``Conv`` with ``group=in_channels``.

    TFLite weight layout: ``(1, kH, kW, out_ch)`` — for a depthwise conv
    with one filter per input channel, the number of groups equals the number
    of input channels.  ONNX expects ``(out_ch, 1, kH, kW)`` for a grouped
    conv.
    """
    opts = op.builtin_options
    padding = opts.get("padding", Padding.VALID)
    stride_w = opts.get("stride_w", 1)
    stride_h = opts.get("stride_h", 1)
    dilation_w = opts.get("dilation_w_factor", 1)
    dilation_h = opts.get("dilation_h_factor", 1)

    # NHWC → NCHW
    x_nchw = g.op.Transpose(op.inputs[0], perm=[0, 3, 1, 2], name="litert_dw_x_nhwc2nchw")
    # TFLite dw weight: (1, kH, kW, in_ch * depth_mult) → (in_ch*dm, 1, kH, kW)
    w_oikk = g.op.Transpose(op.inputs[1], perm=[3, 0, 1, 2], name="litert_dw_w_oihw")

    # The number of groups is the number of input channels.  We derive it
    # from the shape of the transposed input.
    try:
        x_shape = g.get_shape(x_nchw) if g.has_name(x_nchw) and x_nchw in getattr(g, "_known_shapes", {}) else None
    except (AssertionError, AttributeError):
        x_shape = None
    if x_shape is not None and len(x_shape) >= 2 and isinstance(x_shape[1], int):
        groups = int(x_shape[1])
    else:
        # Fall back to 1 (incorrect but allows the graph to be built).
        groups = 1

    conv_kwargs: Dict[str, Any] = dict(
        strides=[stride_h, stride_w],
        dilations=[dilation_h, dilation_w],
        group=groups,
        auto_pad=_padding_to_auto_pad(padding),
        name="litert_dw_conv",
    )

    if len(op.inputs) > 2 and op.inputs[2] >= 0:
        conv_nchw = g.op.Conv(x_nchw, w_oikk, op.inputs[2], **conv_kwargs)
    else:
        conv_nchw = g.op.Conv(x_nchw, w_oikk, **conv_kwargs)

    # NCHW → NHWC
    return g.op.Transpose(conv_nchw, perm=[0, 2, 3, 1], outputs=outputs, name="litert_dw_out")


@register_litert_op_converter(BuiltinOperator.AVERAGE_POOL_2D)
def convert_avg_pool2d(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``AVERAGE_POOL_2D`` → ONNX ``AveragePool``."""
    opts = op.builtin_options
    # NHWC → NCHW
    x_nchw = g.op.Transpose(op.inputs[0], perm=[0, 3, 1, 2], name="litert_avgpool_x")
    pool = g.op.AveragePool(
        x_nchw,
        kernel_shape=[opts.get("filter_height", 1), opts.get("filter_width", 1)],
        strides=[opts.get("stride_h", 1), opts.get("stride_w", 1)],
        auto_pad=_padding_to_auto_pad(opts.get("padding", 1)),
        name="litert_avgpool",
    )
    # NCHW → NHWC
    return g.op.Transpose(pool, perm=[0, 2, 3, 1], outputs=outputs, name="litert_avgpool_out")


@register_litert_op_converter(BuiltinOperator.MAX_POOL_2D)
def convert_max_pool2d(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: TFLiteOperator,
) -> str:
    """TFLite ``MAX_POOL_2D`` → ONNX ``MaxPool``."""
    opts = op.builtin_options
    # NHWC → NCHW
    x_nchw = g.op.Transpose(op.inputs[0], perm=[0, 3, 1, 2], name="litert_maxpool_x")
    pool = g.op.MaxPool(
        x_nchw,
        kernel_shape=[opts.get("filter_height", 1), opts.get("filter_width", 1)],
        strides=[opts.get("stride_h", 1), opts.get("stride_w", 1)],
        auto_pad=_padding_to_auto_pad(opts.get("padding", 1)),
        name="litert_maxpool",
    )
    # NCHW → NHWC
    return g.op.Transpose(pool, perm=[0, 2, 3, 1], outputs=outputs, name="litert_maxpool_out")
