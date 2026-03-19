from typing import Dict, List
from sklearn.neural_network import BernoulliRBM
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(BernoulliRBM)
def sklearn_bernoulli_rbm(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: BernoulliRBM,
    X: str,
    name: str = "bernoulli_rbm",
) -> str:
    """
    Converts a :class:`sklearn.neural_network.BernoulliRBM` into ONNX.

    The transform computes the probability that each hidden unit is activated
    given the visible input:

    .. code-block:: text

        P(h=1|v) = sigmoid(v @ W.T + h_bias)

    where ``W`` is ``components_`` of shape ``(n_components, n_features)``
    and ``h_bias`` is ``intercept_hidden_`` of shape ``(n_components,)``.

    **Graph structure**:

    .. code-block:: text

        X  ──MatMul(components_.T)──Add(intercept_hidden_)──Sigmoid──►  hidden

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names (hidden probabilities)
    :param estimator: a fitted ``BernoulliRBM``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name for the hidden unit probabilities
    """
    assert isinstance(
        estimator, BernoulliRBM
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # components_ has shape (n_components, n_features); we need X @ components_.T
    weights = estimator.components_.astype(dtype)  # (n_components, n_features)
    bias = estimator.intercept_hidden_.astype(dtype)  # (n_components,)

    z = g.op.MatMul(X, weights.T, name=f"{name}_mm")
    z = g.op.Add(z, bias, name=f"{name}_add")
    result = g.op.Sigmoid(z, name=f"{name}_sigmoid", outputs=outputs)
    return result
