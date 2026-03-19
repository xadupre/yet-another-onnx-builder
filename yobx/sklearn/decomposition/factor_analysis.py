import numpy as np
from numpy import linalg
from typing import Dict, List
from sklearn.decomposition import FactorAnalysis
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(FactorAnalysis)
def sklearn_factor_analysis(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: FactorAnalysis,
    X: str,
    name: str = "factor_analysis",
) -> str:
    """
    Converts a :class:`sklearn.decomposition.FactorAnalysis` into ONNX.

    The transformation computes the expected mean of the latent variables
    (see Barber, 21.2.33 or Bishop, 12.66).  All constant matrices are
    pre-computed at conversion time so the resulting ONNX graph is as
    simple as the PCA converter:

    .. code-block:: text

        Wpsi   = components_ / noise_variance_          # (n_components, n_features)
        cov_z  = inv(I + Wpsi @ components_.T)          # (n_components, n_components)
        W_eff  = Wpsi.T @ cov_z                         # (n_features,  n_components)

        X  ──Sub(mean_)──►  centered  ──MatMul(W_eff)──►  output

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``FactorAnalysis``
    :param outputs: desired output names (latent variables)
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output tensor name
    """
    assert isinstance(
        estimator, FactorAnalysis
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # Center the data.
    mean = estimator.mean_.astype(dtype)
    centered = g.op.Sub(X, mean, name=name)

    # Pre-compute the effective weight matrix W_eff = Wpsi.T @ cov_z.
    # This collapses the three runtime matrix operations into a single MatMul.
    Wpsi = (estimator.components_ / estimator.noise_variance_).astype(np.float64)
    Ih = np.eye(len(estimator.components_))
    cov_z = linalg.inv(Ih + Wpsi @ estimator.components_.T.astype(np.float64))
    W_eff = (Wpsi.T @ cov_z).astype(dtype)

    res = g.op.MatMul(centered, W_eff, name=name, outputs=outputs)
    assert isinstance(res, str)  # type happiness
    g.set_type(res, itype)
    return res
