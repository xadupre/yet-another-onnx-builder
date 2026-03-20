from typing import Dict, List
from sklearn.random_projection import GaussianRandomProjection
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(GaussianRandomProjection)
def sklearn_gaussian_random_projection(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: GaussianRandomProjection,
    X: str,
    name: str = "gaussian_random_projection",
) -> str:
    """
    Converts a :class:`sklearn.random_projection.GaussianRandomProjection`
    into ONNX.

    The projection simply multiplies the input by the transposed random
    projection matrix stored in ``components_``:

    .. code-block:: text

        X  ──MatMul(components_.T)──►  output

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names (projected inputs)
    :param estimator: a fitted ``GaussianRandomProjection``
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output tensor name
    """
    assert isinstance(
        estimator, GaussianRandomProjection
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # components_ has shape (n_components, n_features); we need (n_features, n_components).
    components_T = estimator.components_.T.astype(dtype)
    res = g.op.MatMul(X, components_T, name=name, outputs=outputs)
    g.set_type(res, itype)
    if g.has_shape(X):
        batch_dim = g.get_shape(X)[0]
        n_components = estimator.components_.shape[0]
        g.set_shape(res, (batch_dim, n_components))
    return res
