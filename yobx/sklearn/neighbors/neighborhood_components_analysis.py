from typing import Dict, List

from sklearn.neighbors import NeighborhoodComponentsAnalysis

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(NeighborhoodComponentsAnalysis)
def sklearn_neighborhood_components_analysis(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: NeighborhoodComponentsAnalysis,
    X: str,
    name: str = "nca",
) -> str:
    """
    Converts a :class:`sklearn.neighbors.NeighborhoodComponentsAnalysis` into ONNX.

    The transformation is a linear projection of the input onto the learned
    components (no centering is applied):

    .. code-block:: text

        X  ──MatMul(components_.T)──►  output

    The learned ``components_`` matrix has shape ``(n_components, n_features)``.
    At inference time this produces an output of shape ``(N, n_components)``.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names (projected inputs)
    :param estimator: a fitted ``NeighborhoodComponentsAnalysis``
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output tensor name
    """
    assert isinstance(
        estimator, NeighborhoodComponentsAnalysis
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # components_ has shape (n_components, n_features); we need (n_features, n_components).
    components_T = estimator.components_.T.astype(dtype)
    res = g.op.MatMul(X, components_T, name=name, outputs=outputs)
    assert isinstance(res, str)
    g.set_type(res, itype)
    return res
