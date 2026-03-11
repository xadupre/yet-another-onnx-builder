"""
Converter for the TF ``Conv2D`` op → ONNX ``Conv``.

TensorFlow ``Conv2D`` uses:

* **Input layout**: NHWC (batch, height, width, channels)
* **Filter layout**: HWIO (kernel_h, kernel_w, in_channels, out_channels)

ONNX ``Conv`` expects:

* **X layout**: NCHW (batch, channels, height, width)
* **W layout**: OIHW (out_channels, in_channels, kernel_h, kernel_w)

The converter inserts ``Transpose`` nodes to bridge the two conventions:

1. Input:  NHWC → NCHW  (``perm=[0, 3, 1, 2]``)
2. Filter: HWIO → OIHW  (``perm=[3, 2, 0, 1]``)
3. ONNX ``Conv`` is emitted.
4. Output: NCHW → NHWC  (``perm=[0, 2, 3, 1]``)

Padding is expressed as explicit ``pads`` rather than ``auto_pad`` so that
ONNX shape inference can propagate output shapes even when the batch dimension
is dynamic.
"""

import math
from typing import Any, Dict, List, Optional

import tensorflow as tf

from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol


def _explicit_pads(
    padding: str,
    kernel_shape: List[int],
    strides: List[int],
    dilations: List[int],
    input_spatial: List[Optional[int]],
) -> List[int]:
    """
    Returns ONNX-style explicit ``pads`` ``[h_beg, w_beg, h_end, w_end]``.

    For ``"VALID"`` all pads are zero.  For ``"SAME"`` pads are computed using
    TF's *SAME_UPPER* rule; raises ``ValueError`` when a spatial dim is unknown.
    """
    if padding == "VALID":
        return [0] * (len(kernel_shape) * 2)

    # SAME_UPPER
    begins, ends = [], []
    for in_size, k, s, d in zip(input_spatial, kernel_shape, strides, dilations):
        if in_size is None:
            raise ValueError(
                "Cannot compute explicit SAME pads for a dynamic spatial dimension."
            )
        effective_k = (k - 1) * d + 1
        out_size = math.ceil(in_size / s)
        total = max(0, (out_size - 1) * s + effective_k - in_size)
        begins.append(total // 2)
        ends.append(total - total // 2)
    return begins + ends


@register_tf_op_converter("Conv2D")
def convert_conv2d(
    g: GraphBuilderExtendedProtocol,
    sts: Dict[str, Any],
    outputs: List[str],
    op: tf.Operation,
) -> str:
    """
    Converts TF ``Conv2D`` (NHWC / HWIO) → ONNX ``Conv`` (NCHW / OIHW).

    Dilations and strides are extracted from the 4-element NHWC-ordered
    attribute lists; only the spatial components (indices 1 and 2) are
    forwarded to the ONNX node.  Padding is expressed as explicit pads so
    that ONNX shape inference works with a dynamic batch dimension.
    """
    data_format = op.get_attr("data_format").decode()
    if data_format != "NHWC":
        raise NotImplementedError(
            f"Conv2D with data_format={data_format!r} is not supported."
        )

    strides = list(op.get_attr("strides"))      # [1, sH, sW, 1]
    dilations = list(op.get_attr("dilations"))  # [1, dH, dW, 1]
    padding_raw = op.get_attr("padding")
    padding = padding_raw.decode() if isinstance(padding_raw, bytes) else padding_raw

    onnx_strides = strides[1:3]
    onnx_dilations = dilations[1:3]

    # Filter shape is HWIO: [kH, kW, in_ch, out_ch]
    filter_shape = op.inputs[1].shape.as_list()
    kernel_shape = filter_shape[:2]  # [kH, kW]

    nhwc_shape = op.inputs[0].shape.as_list()  # [N, H, W, C]
    input_spatial = nhwc_shape[1:3]            # [H, W]

    try:
        pads = _explicit_pads(padding, kernel_shape, onnx_strides, onnx_dilations,
                               input_spatial)
        conv_kwargs: Dict[str, Any] = dict(strides=onnx_strides, dilations=onnx_dilations,
                                           pads=pads)
    except ValueError:
        auto_pad = "SAME_UPPER" if padding == "SAME" else "VALID"
        conv_kwargs = dict(strides=onnx_strides, dilations=onnx_dilations,
                           auto_pad=auto_pad)

    # Input: NHWC → NCHW
    x_nchw = g.op.Transpose(
        op.inputs[0].name, perm=[0, 3, 1, 2], name=f"{op.name}_in_t"
    )
    # Filter: HWIO → OIHW
    w_oihw = g.op.Transpose(
        op.inputs[1].name, perm=[3, 2, 0, 1], name=f"{op.name}_w_t"
    )
    # ONNX Conv (NCHW)
    y_nchw = g.op.Conv(
        x_nchw,
        w_oihw,
        name=f"{op.name}_conv",
        **conv_kwargs,
    )
    # Output: NCHW → NHWC
    return g.op.Transpose(y_nchw, perm=[0, 2, 3, 1], outputs=outputs[:1], name=op.name)
