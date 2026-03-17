from typing import Dict, List, Union

import numpy as np
from sklearn.decomposition import MiniBatchSparsePCA, SparsePCA

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter((SparsePCA, MiniBatchSparsePCA))
def sklearn_sparse_pca(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: Union[SparsePCA, MiniBatchSparsePCA],
    X: str,
    name: str = "sparse_pca",
) -> str:
    """
    Converts a :class:`sklearn.decomposition.SparsePCA` or
    :class:`sklearn.decomposition.MiniBatchSparsePCA` into ONNX.

    The transformation replicates :meth:`sklearn.decomposition.SparsePCA.transform`:

    1. Centre the data by subtracting the per-feature mean stored in ``mean_``.
    2. Apply the pre-computed projection matrix *W* via a single matrix
       multiplication, where *W* is derived from the fitted dictionary
       ``components_`` and the regularisation parameter ``ridge_alpha``:

    .. math::

        W = {\\mathbf{C}}^{\\top}
            \\bigl(\\mathbf{C}\\,{\\mathbf{C}}^{\\top} + \\alpha\\,\\mathbf{I}\\bigr)^{-1}

    with :math:`\\mathbf{C}` = ``components_`` and :math:`\\alpha` = ``ridge_alpha``.

    The full graph is:

    .. code-block:: text

        X  ──Sub(mean_)──►  centered  ──MatMul(W)──►  output

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``SparsePCA`` or ``MiniBatchSparsePCA``
    :param outputs: desired output names (sparse codes)
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output tensor name
    """
    assert isinstance(
        estimator, (SparsePCA, MiniBatchSparsePCA)
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # Centre the data.
    mean = estimator.mean_.astype(dtype)
    centered = g.op.Sub(X, mean, name=name)

    # Pre-compute the projection matrix W = components_.T @ inv(components_ @ components_.T + ridge_alpha * I).
    # At inference this reduces the transform to a single MatMul.
    C = estimator.components_  # (n_components, n_features)
    n_components = C.shape[0]
    M = C @ C.T + estimator.ridge_alpha * np.eye(n_components, dtype=C.dtype)  # (n_components, n_components)
    W = C.T @ np.linalg.inv(M)  # (n_features, n_components)
    W = W.astype(dtype)

    res = g.op.MatMul(centered, W, name=name, outputs=outputs)
    assert isinstance(res, str)  # type happiness
    if not sts:
        g.set_type(res, itype)
    return res
