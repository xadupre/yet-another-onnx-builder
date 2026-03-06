from typing import Tuple, Dict, List
import numpy as np
import onnx
from sklearn.neural_network import MLPClassifier, MLPRegressor
from ..register import register_sklearn_converter
from ...xbuilder import GraphBuilder


def _apply_activation(g: GraphBuilder, x: str, activation: str, name: str) -> str:
    """
    Applies an activation function in-place and returns the output tensor name.

    Supported activations are ``'identity'``, ``'logistic'``, ``'tanh'``,
    and ``'relu'``.

    :param g: graph builder
    :param x: input tensor name
    :param activation: activation function name (sklearn convention)
    :param name: node name prefix
    :return: output tensor name after activation
    """
    if activation == "identity":
        return x
    if activation == "logistic":
        return g.op.Sigmoid(x, name=name)  # pyrefly: ignore[bad-return]
    if activation == "tanh":
        return g.op.Tanh(x, name=name)  # pyrefly: ignore[bad-return]
    if activation == "relu":
        return g.op.Relu(x, name=name)  # pyrefly: ignore[bad-return]
    raise NotImplementedError(f"Activation {activation!r} is not supported.")


@register_sklearn_converter((MLPClassifier,))
def sklearn_mlp_classifier(
    g: GraphBuilder,
    sts: Dict,
    outputs: List[str],
    estimator: MLPClassifier,
    X: str,
    name: str = "mlp_classifier",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.neural_network.MLPClassifier` into ONNX.

    All hidden layers use the activation stored in ``estimator.activation``;
    the output layer uses ``estimator.out_activation_`` which is ``'logistic'``
    for binary classification and ``'softmax'`` for multi-class.

    **Hidden layers** (repeated for each weight matrix except the last):

    .. code-block:: text

        h_prev  ──MatMul(coef_i)──Add(bias_i)──Activation──►  h_i

    **Binary classification** (``out_activation_ == 'logistic'``):

    .. code-block:: text

        h  ──MatMul(coef_out)──Add(bias_out)──Sigmoid──►  proba_pos
                                                               │
                                              Sub(1, ·) ──►  proba_neg
                                                     │
                                              Concat  ──►  probabilities
                                                │
                                           ArgMax──Cast──Gather(classes)──►  label

    **Multi-class classification** (``out_activation_ == 'softmax'``):

    .. code-block:: text

        h  ──MatMul(coef_out)──Add(bias_out)──Softmax──►  probabilities
                                                               │
                                           ArgMax──Cast──Gather(classes)──►  label

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``MLPClassifier``
    :param outputs: desired names (label, probabilities)
    :param X: input tensor name
    :param name: prefix names for the added nodes
    :return: tuple ``(label_result_name, proba_result_name)``
    """
    assert isinstance(
        estimator, MLPClassifier
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = g.onnx_dtype_to_np_dtype(itype)

    coefs = estimator.coefs_
    intercepts = estimator.intercepts_
    hidden_activation = estimator.activation
    out_activation = estimator.out_activation_
    classes = estimator.classes_
    is_binary = len(classes) == 2

    # Forward pass through all hidden layers (all layers except the last).
    h = X
    for i in range(len(coefs) - 1):
        coef = coefs[i].astype(dtype)
        bias = intercepts[i].astype(dtype)
        z = g.op.MatMul(h, coef, name=f"{name}_mm{i}")
        z = g.op.Add(z, bias, name=f"{name}_add{i}")
        h = _apply_activation(g, z, hidden_activation, name=f"{name}_act{i}")  # pyrefly: ignore[bad-argument-type]
    coef_out = coefs[-1].astype(dtype)
    bias_out = intercepts[-1].astype(dtype)
    z_out = g.op.MatMul(h, coef_out, name=f"{name}_mm_out")
    z_out = g.op.Add(z_out, bias_out, name=f"{name}_add_out")

    # Output activation and probability construction.
    if is_binary and out_activation == "logistic":
        proba_pos = g.op.Sigmoid(z_out, name=f"{name}_sigmoid")
        proba_neg = g.op.Sub(np.array([1], dtype=dtype), proba_pos, name=f"{name}_neg")
        proba = g.op.Concat(proba_neg, proba_pos, axis=-1, name=name, outputs=outputs[1:])
    elif out_activation == "softmax":
        proba = g.op.Softmax(z_out, axis=1, name=name, outputs=outputs[1:])
    else:
        raise NotImplementedError(
            f"Output activation {out_activation!r} is not supported for MLPClassifier."
        )

    assert isinstance(proba, str)
    label_idx = g.op.ArgMax(proba, axis=1, keepdims=0, name=name)
    label_idx_cast = g.op.Cast(label_idx, to=onnx.TensorProto.INT64, name=name)

    if np.issubdtype(classes.dtype, np.integer):
        classes_arr = classes.astype(np.int64)
        label = g.op.Gather(
            classes_arr,
            label_idx_cast,
            axis=0,
            name=f"{name}_label",
            outputs=outputs[:1],
        )
        assert isinstance(label, str)
        if not sts:
            g.set_type(label, onnx.TensorProto.INT64)
    else:
        classes_arr = np.array(classes.astype(str))
        label = g.op.Gather(
            classes_arr,
            label_idx_cast,
            axis=0,
            name=f"{name}_label_string",
            outputs=outputs[:1],
        )
        assert isinstance(label, str)
        if not sts:
            g.set_type(label, onnx.TensorProto.STRING)
    return label, proba


@register_sklearn_converter((MLPRegressor,))
def sklearn_mlp_regressor(
    g: GraphBuilder,
    sts: Dict,
    outputs: List[str],
    estimator: MLPRegressor,
    X: str,
    name: str = "mlp_regressor",
) -> str:
    """
    Converts a :class:`sklearn.neural_network.MLPRegressor` into ONNX.

    All hidden layers use the activation stored in ``estimator.activation``
    (``'identity'``, ``'logistic'``, ``'tanh'``, or ``'relu'``); the output
    layer always uses the ``'identity'`` activation (i.e., the linear output
    is returned as-is).

    .. code-block:: text

        X  ──hidden layers──►  h  ──MatMul(coef_out)──Add(bias_out)──►  predictions

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``MLPRegressor``
    :param outputs: desired output names (predictions)
    :param X: input tensor name
    :param name: prefix names for the added nodes
    :return: output tensor name
    """
    assert isinstance(
        estimator, MLPRegressor
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = g.onnx_dtype_to_np_dtype(itype)

    coefs = estimator.coefs_
    intercepts = estimator.intercepts_
    hidden_activation = estimator.activation

    # Forward pass through all hidden layers (all layers except the last).
    h = X
    for i in range(len(coefs) - 1):
        coef = coefs[i].astype(dtype)
        bias = intercepts[i].astype(dtype)
        z = g.op.MatMul(h, coef, name=f"{name}_mm{i}")
        z = g.op.Add(z, bias, name=f"{name}_add{i}")
        assert isinstance(z, str)  # type happiness
        h = _apply_activation(g, z, hidden_activation, name=f"{name}_act{i}")

    # Output layer: linear (identity activation).
    coef_out = coefs[-1].astype(dtype)
    bias_out = intercepts[-1].astype(dtype)
    z_out = g.op.MatMul(h, coef_out, name=f"{name}_mm_out")
    result = g.op.Add(z_out, bias_out, name=f"{name}_add_out", outputs=outputs)
    assert isinstance(result, str)
    return result
