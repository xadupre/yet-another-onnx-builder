from typing import Dict, List
import numpy as np
from tensorflow.keras.layers import Dense  # type: ignore[import]
from ..register import register_tensorflow_converter
from ...xbuilder import GraphBuilder


@register_tensorflow_converter(Dense)
def tensorflow_dense(
    g: GraphBuilder,
    sts: Dict,
    outputs: List[str],
    layer: Dense,
    X: str,
    name: str = "dense",
) -> str:
    """
    Converts a :class:`tensorflow.keras.layers.Dense` layer into ONNX.

    A ``Dense`` layer computes ``output = activation(X @ kernel + bias)``.
    The kernel and bias are extracted from the layer weights.

    Without activation (``linear``):

    .. code-block:: text

        X  ──MatMul(kernel)──Add(bias)──►  output

    With activation (e.g. ``relu``):

    .. code-block:: text

        X  ──MatMul(kernel)──Add(bias)──Relu──►  output

    :param g: the graph builder to add nodes to
    :param sts: shapes dictionary (unused, kept for API consistency)
    :param outputs: desired names for the output tensors
    :param layer: a built (fitted) Keras ``Dense`` layer
    :param X: name of the input tensor
    :param name: prefix used for names of added nodes
    :return: name of the output tensor
    """
    assert isinstance(layer, Dense), f"Unexpected type {type(layer)} for layer."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = g.onnx_dtype_to_np_dtype(itype)

    weights = layer.get_weights()
    if layer.use_bias:
        if len(weights) < 2:
            raise ValueError(
                f"Dense layer has use_bias=True but only {len(weights)} weight arrays "
                f"were returned by get_weights()."
            )
        kernel = weights[0].astype(dtype)
        bias = weights[1].astype(dtype)
    else:
        kernel = weights[0].astype(dtype)
        bias = None

    pre_act = g.op.MatMul(X, kernel, name=name)
    if bias is not None:
        pre_act = g.op.Add(pre_act, bias, name=name)

    activation = layer.activation.__name__ if hasattr(layer.activation, "__name__") else None
    if activation in (None, "linear"):
        res = g.op.Identity(pre_act, name=name, outputs=outputs)
    elif activation == "relu":
        res = g.op.Relu(pre_act, name=name, outputs=outputs)
    elif activation == "sigmoid":
        res = g.op.Sigmoid(pre_act, name=name, outputs=outputs)
    elif activation == "tanh":
        res = g.op.Tanh(pre_act, name=name, outputs=outputs)
    elif activation == "softmax":
        res = g.op.Softmax(pre_act, axis=-1, name=name, outputs=outputs)
    else:
        raise NotImplementedError(
            f"Activation {activation!r} is not yet supported by the Dense converter."
        )

    assert isinstance(res, str)  # type happiness
    # Only propagate type/device metadata when the shape-type propagation pass
    # (``sts``) has not already populated them.
    if not sts:
        g.set_type(res, g.get_type(X))
        if g.has_device(X):
            g.set_device(res, g.get_device(X))
    return res
