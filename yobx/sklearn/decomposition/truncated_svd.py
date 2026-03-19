from typing import Dict, List
from sklearn.decomposition import TruncatedSVD
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(TruncatedSVD)
def sklearn_truncated_svd(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: TruncatedSVD,
    X: str,
    name: str = "truncated_svd",
) -> str:
    """
    Converts a :class:`sklearn.decomposition.TruncatedSVD` into ONNX.

    Unlike PCA, TruncatedSVD does **not** center the data before projecting.
    The transformation is simply a matrix multiplication:

    .. code-block:: text

        X  ──MatMul(components_.T)──►  output

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``TruncatedSVD``
    :param outputs: desired output names (projected inputs)
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output tensor name
    """
    assert isinstance(
        estimator, TruncatedSVD
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # Project onto the singular vectors.
    # components_ has shape (n_components, n_features); we need (n_features, n_components).
    components_T = estimator.components_.T.astype(dtype)
    res = g.op.MatMul(X, components_T, name=name, outputs=outputs)
    assert isinstance(res, str)  # type happiness
    g.set_type(res, itype)
    return res
