from typing import Tuple, Dict, List, Union
import numpy as np
import onnx
from sklearn.neural_network import MLPClassifier, MLPRegressor
from ..register import register_sklearn_converter
from ...xbuilder import GraphBuilder


def _apply_activation(g: GraphBuilder, x: str, activation: str, name: str) -> str:
    """
    Applies the given activation function to *x* and returns the result name.

    Supported activations:

    * ``'relu'`` → ``Relu``
    * ``'tanh'`` → ``Tanh``
    * ``'logistic'`` → ``Sigmoid``
    * ``'identity'`` → ``Identity`` (pass-through)

    :param g: the graph builder
    :param x: input tensor name
    :param activation: activation name as stored in the sklearn estimator
    :param name: node name prefix
    :return: output tensor name
    """
    if activation == "relu":
        return g.op.Relu(x, name=name)
    if activation == "tanh":
        return g.op.Tanh(x, name=name)
    if activation in ("logistic", "sigmoid"):
        return g.op.Sigmoid(x, name=name)
    if activation == "identity":
        return g.op.Identity(x, name=name)
    raise NotImplementedError(
        f"Activation {activation!r} is not supported. "
        "Supported activations: 'relu', 'tanh', 'logistic', 'identity'."
    )


def _forward_hidden_layers(
    g: GraphBuilder,
    X: str,
    coefs: List[np.ndarray],
    intercepts: List[np.ndarray],
    activation: str,
    dtype,
    name: str,
) -> str:
    """
    Emits ONNX nodes for all hidden layers of an MLP.

    Each hidden layer *i* computes::

        h_i = activation(Gemm(h_{i-1}, coefs[i], intercepts[i]))

    sklearn MLP coefficient matrices are already stored in ``(in_features,
    out_features)`` order, so no ``transB`` transpose is required.

    The input layer (index 0) reads from *X*.  The *last* weight/bias pair
    (``coefs[-1]`` / ``intercepts[-1]``) is **not** processed here; it is
    handled by the caller together with the output activation.

    :param g: graph builder
    :param X: name of the input tensor
    :param coefs: list of all weight matrices (including the output layer)
    :param intercepts: list of all bias vectors (including the output layer)
    :param activation: hidden-layer activation name
    :param dtype: numpy dtype matching the input tensor
    :param name: node name prefix
    :return: name of the last hidden-layer output tensor
    """
    h = X
    # Iterate over hidden layers only (all but the last weight matrix).
    for i, (coef, bias) in enumerate(zip(coefs[:-1], intercepts[:-1])):
        linear = g.op.Gemm(
            h,
            coef.astype(dtype),
            bias.astype(dtype),
            name=f"{name}_layer{i}",
        )
        h = _apply_activation(g, linear, activation, name=f"{name}_act{i}")
    return h


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

    The graph mirrors the forward pass of sklearn's MLP, applying ``Gemm``
    for each fully-connected layer followed by the hidden activation, and
    finally the output activation (``Softmax`` for multi-class,
    ``Sigmoid`` + ``Concat`` for binary):

    **Binary** (``out_activation_ == 'logistic'``):

    .. code-block:: text

        X ──hidden layers──► h
           ──Gemm(coefs[-1], intercepts[-1])──► raw
           ──Sigmoid──► proba_pos
           ──Sub(1, ·) + Concat──► probabilities  ──ArgMax──Cast──Gather──► label

    **Multi-class** (``out_activation_ == 'softmax'``):

    .. code-block:: text

        X ──hidden layers──► h
           ──Gemm(coefs[-1], intercepts[-1])──► raw
           ──Softmax──► probabilities  ──ArgMax──Cast──Gather──► label

    :param g: graph builder
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names ``[label, probabilities]``
    :param estimator: a fitted ``MLPClassifier``
    :param X: input tensor name
    :param name: node name prefix
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

    # ------------------------------------------------------------------ #
    # Hidden layers                                                        #
    # ------------------------------------------------------------------ #
    h = _forward_hidden_layers(g, X, coefs, intercepts, hidden_activation, dtype, name)

    # ------------------------------------------------------------------ #
    # Output layer: linear part                                           #
    # ------------------------------------------------------------------ #
    raw = g.op.Gemm(
        h,
        coefs[-1].astype(dtype),
        intercepts[-1].astype(dtype),
        name=f"{name}_out",
    )

    # ------------------------------------------------------------------ #
    # Output activation + probabilities                                   #
    # ------------------------------------------------------------------ #
    is_binary = out_activation == "logistic"

    if is_binary:
        proba_pos = g.op.Sigmoid(raw, name=f"{name}_sigmoid")
        proba_neg = g.op.Sub(
            np.array([1], dtype=dtype), proba_pos, name=f"{name}_neg"
        )
        proba = g.op.Concat(
            proba_neg, proba_pos, axis=-1, name=f"{name}_concat", outputs=outputs[1:]
        )
    else:
        # Softmax (or any other out_activation treated as softmax for classifiers)
        proba = g.op.Softmax(raw, axis=1, name=f"{name}_softmax", outputs=outputs[1:])

    assert isinstance(proba, str)

    # ------------------------------------------------------------------ #
    # Label: ArgMax + Gather over classes                                 #
    # ------------------------------------------------------------------ #
    label_idx = g.op.ArgMax(proba, axis=1, keepdims=0, name=f"{name}_argmax")
    label_idx_cast = g.op.Cast(label_idx, to=onnx.TensorProto.INT64, name=f"{name}_cast")

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

    The graph mirrors the forward pass of sklearn's MLP, applying ``Gemm``
    for each fully-connected layer followed by the hidden activation,
    and finally the output activation (typically ``identity`` for regression):

    .. code-block:: text

        X ──hidden layers──► h
           ──Gemm(coefs[-1], intercepts[-1])──► raw  ──[out_activation]──► predictions

    :param g: graph builder
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names ``[predictions]``
    :param estimator: a fitted ``MLPRegressor``
    :param X: input tensor name
    :param name: node name prefix
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
    out_activation = estimator.out_activation_

    # ------------------------------------------------------------------ #
    # Hidden layers                                                        #
    # ------------------------------------------------------------------ #
    h = _forward_hidden_layers(g, X, coefs, intercepts, hidden_activation, dtype, name)

    # ------------------------------------------------------------------ #
    # Output layer                                                         #
    # ------------------------------------------------------------------ #
    if out_activation == "identity":
        result = g.op.Gemm(
            h,
            coefs[-1].astype(dtype),
            intercepts[-1].astype(dtype),
            name=f"{name}_out",
            outputs=outputs,
        )
    else:
        raw = g.op.Gemm(
            h,
            coefs[-1].astype(dtype),
            intercepts[-1].astype(dtype),
            name=f"{name}_out",
        )
        result = _apply_activation(g, raw, out_activation, name=f"{name}_out_act")
        # Rename to the desired output name.
        result = g.op.Identity(result, name=f"{name}_identity", outputs=outputs)

    assert isinstance(result, str)
    return result
