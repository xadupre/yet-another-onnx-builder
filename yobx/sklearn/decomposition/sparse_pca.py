import numpy as np
from numpy import linalg
from typing import Dict, List, Union
from sklearn.decomposition import SparsePCA, MiniBatchSparsePCA
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

    The transformation replicates :meth:`sklearn.decomposition.SparsePCA.transform`,
    which solves a ridge regression problem to project the centred data onto
    the (non-orthogonal) sparse components.  All constant matrices are
    pre-computed at conversion time so the resulting ONNX graph reduces to
    a single centring step followed by one matrix multiplication:

    .. code-block:: text

        ridge_alpha   scalar regularisation (default 0.01)
        cov           = components_ @ components_.T + ridge_alpha * I   # (n_components, n_components)
        W_eff         = components_.T @ inv(cov)                        # (n_features,  n_components)

        X  ──Sub(mean_)──►  centered  ──MatMul(W_eff)──►  output

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``SparsePCA`` or ``MiniBatchSparsePCA``
    :param outputs: desired output names (projected inputs)
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output tensor name
    """
    assert isinstance(estimator, (SparsePCA, MiniBatchSparsePCA)), (
        f"Unexpected type {type(estimator)} for estimator."
    )
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # Center the data.
    mean = estimator.mean_.astype(dtype)
    centered = g.op.Sub(X, mean, name=name)

    # Pre-compute the effective weight matrix W_eff = components_.T @ inv(cov)
    # where cov = components_ @ components_.T + ridge_alpha * I.
    # This collapses the ridge regression into a single MatMul at runtime.
    components = estimator.components_.astype(np.float64)
    n_comp = components.shape[0]
    cov = components @ components.T + estimator.ridge_alpha * np.eye(n_comp)
    W_eff = (components.T @ linalg.inv(cov)).astype(dtype)

    res = g.op.MatMul(centered, W_eff, name=name, outputs=outputs)
    assert isinstance(res, str)  # type happiness
    if not sts:
        g.set_type(res, itype)
    return res
