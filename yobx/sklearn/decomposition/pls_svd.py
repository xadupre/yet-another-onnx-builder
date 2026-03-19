from typing import Dict, List
from sklearn.cross_decomposition import PLSSVD
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(PLSSVD)
def sklearn_pls_svd(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: PLSSVD,
    X: str,
    name: str = "pls_svd",
) -> str:
    """
    Converts a :class:`sklearn.cross_decomposition.PLSSVD` into ONNX.

    The transform formula mirrors :meth:`PLSSVD.transform`:

    .. code-block:: text

        X ──Sub(_x_mean)──► centered ── Div(_x_std) ──► scaled ── MatMul(x_weights_)──► x_scores

    The input is centred and scaled, then projected onto the left singular
    vectors ``x_weights_``, giving ``x_scores`` of shape ``(N, n_components)``.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``PLSSVD``
    :param outputs: desired output names
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output tensor name
    """
    assert isinstance(estimator, PLSSVD), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # Center: Xr = (X - _x_mean) / _x_std
    x_mean = estimator._x_mean.astype(dtype)
    x_std = estimator._x_std.astype(dtype)
    centered = g.op.Sub(X, x_mean, name=name)
    scaled = g.op.Div(centered, x_std, name=name)

    # Project: x_scores = scaled @ x_weights_
    x_weights = estimator.x_weights_.astype(dtype)
    res = g.op.MatMul(scaled, x_weights, name=name, outputs=outputs)

    assert isinstance(res, str)
    g.set_type(res, itype)
    if g.has_shape(X):
        batch_dim = g.get_shape(X)[0]
        n_components = estimator.x_weights_.shape[1]
        g.set_shape(res, (batch_dim, n_components))
    elif g.has_rank(X):
        g.set_rank(res, 2)
    return res
