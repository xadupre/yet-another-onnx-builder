"""
Converters for TF 2-D pooling ops: ``MaxPool``, ``AvgPool``.

TensorFlow uses **NHWC** (batch, height, width, channels) data format by
default.  ONNX ``MaxPool`` and ``AveragePool`` expect **NCHW** input.  The
converters therefore insert ``Transpose`` nodes:

* Before pooling: NHWC → NCHW  (``perm=[0, 3, 1, 2]``)
* After  pooling: NCHW → NHWC  (``perm=[0, 2, 3, 1]``)

Padding is always expressed as explicit ``pads`` rather than ``auto_pad`` so
that ONNX shape inference can propagate output shapes even when the batch
dimension is dynamic.
"""

import math
from typing import Any, Dict, List, Optional

import tensorflow as tf

from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _explicit_pads(
    padding: str,
    kernel_shape: List[int],
    strides: List[int],
    input_spatial: List[Optional[int]],
) -> List[int]:
    """
    Returns ONNX-style explicit ``pads`` ``[h_beg, w_beg, h_end, w_end]``.

    For ``"VALID"`` padding this is always all-zeros.  For ``"SAME"`` padding
    the values are computed using TF's *SAME_UPPER* formula (round extra
    padding towards the end).  If a spatial input dimension is unknown (``None``)
    the caller must fall back to ``auto_pad="SAME_UPPER"`` instead.
    """
    if padding == "VALID":
        return [0] * (len(kernel_shape) * 2)

    # SAME_UPPER
    begins, ends = [], []
    for in_size, k, s in zip(input_spatial, kernel_shape, strides):
        if in_size is None:
            raise ValueError(
                "Cannot compute explicit SAME pads for a dynamic spatial dimension."
            )
        out_size = math.ceil(in_size / s)
        total = max(0, (out_size - 1) * s + k - in_size)
        begins.append(total // 2)
        ends.append(total - total // 2)
    return begins + ends


def _set_pool_nchw_shape(
    g: GraphBuilderExtendedProtocol,
    op: "tf.Operation",
    x_nchw: str,
    pool_out: str,
) -> None:
    """
    Manually register the NCHW output shape of a pooling node.

    ONNX shape inference for ``AveragePool`` can fail when the batch dimension
    is a symbolic string.  This helper reads TF's already-inferred output shape
    (NHWC: ``[N, H_out, W_out, C]``) and converts it to the NCHW layout that
    the ONNX pooling node produces (``[N, C, H_out, W_out]``), then registers
    it in the GraphBuilder so that downstream nodes can use it.
    """
    if not g.has_shape(x_nchw):
        return
    nchw_in = g.get_shape(x_nchw)      # (batch, C_in, H_in, W_in)
    batch_dim = nchw_in[0]              # may be an int or a symbolic string

    tf_out = op.outputs[0].shape.as_list()  # NHWC: [N, H_out, W_out, C]
    if len(tf_out) != 4:
        return
    _, h_out, w_out, c = tf_out
    if any(v is None for v in (h_out, w_out, c)):
        return

    g.set_shape(pool_out, (batch_dim, c, h_out, w_out))
    g.set_type(pool_out, g.get_type(x_nchw))


@register_tf_op_converter("MaxPool")
def convert_max_pool(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: tf.Operation,
) -> str:
    """
    Converts TF ``MaxPool`` (NHWC) → ONNX ``MaxPool`` (NCHW).

    TF ``ksize`` / ``strides`` attributes are 4-element lists in NHWC order
    ``[batch, H, W, C]``; only the spatial dimensions ``[H, W]`` are forwarded
    to the ONNX node.  Padding is represented as explicit ``pads`` to enable
    ONNX shape inference regardless of whether the batch dimension is dynamic.
    """
    data_format = op.get_attr("data_format").decode()
    if data_format != "NHWC":
        raise NotImplementedError(
            f"MaxPool with data_format={data_format!r} is not supported."
        )

    ksize = list(op.get_attr("ksize"))      # [1, kH, kW, 1]
    strides = list(op.get_attr("strides"))  # [1, sH, sW, 1]
    padding = _decode(op.get_attr("padding"))

    kernel_shape = ksize[1:3]
    onnx_strides = strides[1:3]

    nhwc_shape = op.inputs[0].shape.as_list()  # [N, H, W, C]
    input_spatial = nhwc_shape[1:3]            # [H, W]

    try:
        pads = _explicit_pads(padding, kernel_shape, onnx_strides, input_spatial)
        pool_kwargs: Dict[str, Any] = dict(kernel_shape=kernel_shape, strides=onnx_strides,
                                           pads=pads)
    except ValueError:
        # Dynamic spatial dims — fall back to auto_pad
        auto_pad = "SAME_UPPER" if padding == "SAME" else "VALID"
        pool_kwargs = dict(kernel_shape=kernel_shape, strides=onnx_strides,
                           auto_pad=auto_pad)

    # NHWC → NCHW
    x_nchw = g.op.Transpose(
        op.inputs[0].name, perm=[0, 3, 1, 2], name=f"{op.name}_in_t"
    )
    # Apply MaxPool in NCHW; request exactly one output to avoid returning indices
    pool_out = f"{op.name}_nchw_out"
    g.op.MaxPool(x_nchw, outputs=[pool_out], name=f"{op.name}_pool", **pool_kwargs)
    _set_pool_nchw_shape(g, op, x_nchw, pool_out)
    # NCHW → NHWC
    return g.op.Transpose(pool_out, perm=[0, 2, 3, 1], outputs=outputs[:1], name=op.name)


@register_tf_op_converter("AvgPool")
def convert_avg_pool(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: tf.Operation,
) -> str:
    """
    Converts TF ``AvgPool`` (NHWC) → ONNX ``AveragePool`` (NCHW).

    TF ``ksize`` / ``strides`` attributes are 4-element lists in NHWC order
    ``[batch, H, W, C]``; only the spatial dimensions ``[H, W]`` are forwarded
    to the ONNX node.  Padding is represented as explicit ``pads`` to enable
    ONNX shape inference regardless of whether the batch dimension is dynamic.
    """
    data_format = op.get_attr("data_format").decode()
    if data_format != "NHWC":
        raise NotImplementedError(
            f"AvgPool with data_format={data_format!r} is not supported."
        )

    ksize = list(op.get_attr("ksize"))      # [1, kH, kW, 1]
    strides = list(op.get_attr("strides"))  # [1, sH, sW, 1]
    padding = _decode(op.get_attr("padding"))

    kernel_shape = ksize[1:3]
    onnx_strides = strides[1:3]

    nhwc_shape = op.inputs[0].shape.as_list()  # [N, H, W, C]
    input_spatial = nhwc_shape[1:3]            # [H, W]

    try:
        pads = _explicit_pads(padding, kernel_shape, onnx_strides, input_spatial)
        pool_kwargs: Dict[str, Any] = dict(kernel_shape=kernel_shape, strides=onnx_strides,
                                           pads=pads)
    except ValueError:
        auto_pad = "SAME_UPPER" if padding == "SAME" else "VALID"
        pool_kwargs = dict(kernel_shape=kernel_shape, strides=onnx_strides,
                           auto_pad=auto_pad)

    # NHWC → NCHW
    x_nchw = g.op.Transpose(
        op.inputs[0].name, perm=[0, 3, 1, 2], name=f"{op.name}_in_t"
    )
    # Apply AveragePool in NCHW
    pool_out = f"{op.name}_nchw_out"
    g.op.AveragePool(x_nchw, outputs=[pool_out], name=f"{op.name}_pool", **pool_kwargs)
    # ONNX AveragePool shape inference may not propagate reliably when the batch
    # dimension is dynamic; manually register the output shape from TF's known shape.
    _set_pool_nchw_shape(g, op, x_nchw, pool_out)
    # NCHW → NHWC
    return g.op.Transpose(pool_out, perm=[0, 2, 3, 1], outputs=outputs[:1], name=op.name)
