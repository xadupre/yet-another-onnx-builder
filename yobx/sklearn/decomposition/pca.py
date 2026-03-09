from typing import Dict, List
from sklearn.decomposition import PCA
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(PCA)
def sklearn_pca(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: PCA,
    X: str,
    name: str = "pca",
) -> str:
    """
    Converts a :class:`sklearn.decomposition.PCA` into ONNX.

    The transformation centres the data when the model was fitted with
    centering (i.e. ``mean_`` is not ``None``) and then projects it onto
    the principal components:

    .. code-block:: text

        X  ──Sub(mean_)──►  centered  ──MatMul(components_.T)──►  output
             (if mean_ is not None)

    When ``mean_`` is ``None`` the ``Sub`` node is skipped.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``PCA``
    :param outputs: desired output names (projected inputs)
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output tensor name
    """
    assert isinstance(estimator, PCA), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # Center the data if the model stores a mean vector.
    if estimator.mean_ is not None:
        mean = estimator.mean_.astype(dtype)
        centered = g.op.Sub(X, mean, name=name)
    else:
        centered = X

    # Project onto the principal components.
    # components_ has shape (n_components, n_features); we need (n_features, n_components).
    components_T = estimator.components_.T.astype(dtype)
    res = g.op.MatMul(centered, components_T, name=name, outputs=outputs)
    assert isinstance(res, str)  # type happiness
    if not sts:
        g.set_type(res, itype)
    return res
