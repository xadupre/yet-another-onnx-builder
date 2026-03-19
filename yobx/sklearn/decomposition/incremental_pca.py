from typing import Dict, List
import numpy as np
from sklearn.decomposition import IncrementalPCA
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(IncrementalPCA)
def sklearn_incremental_pca(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: IncrementalPCA,
    X: str,
    name: str = "incremental_pca",
) -> str:
    """
    Converts a :class:`sklearn.decomposition.IncrementalPCA` into ONNX.

    The transformation centres the data by subtracting the per-feature mean
    stored in ``mean_``, projects onto the principal components, and
    optionally whitens the result when the estimator was fitted with
    ``whiten=True``:

    .. code-block:: text

        X  ──Sub(mean_)──►  centered  ──MatMul(components_.T)──►  projected
                                                                        │
                                              whiten=True ──Div(√var_)──┘
                                                                        │
                                                                     output

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``IncrementalPCA``
    :param outputs: desired output names (projected inputs)
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output tensor name
    """
    assert isinstance(
        estimator, IncrementalPCA
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # Center the data.
    mean = estimator.mean_.astype(dtype)
    centered = g.op.Sub(X, mean, name=name)

    # Project onto the principal components.
    # components_ has shape (n_components, n_features); we need (n_features, n_components).
    components_T = estimator.components_.T.astype(dtype)

    if estimator.whiten:
        # Whiten: divide projected data by sqrt(explained_variance_).
        projected = g.op.MatMul(centered, components_T, name=name)
        scale = np.sqrt(estimator.explained_variance_).astype(dtype)
        res = g.op.Div(projected, scale, name=name, outputs=outputs)
    else:
        res = g.op.MatMul(centered, components_T, name=name, outputs=outputs)

    g.set_type(res, itype)
    return res
