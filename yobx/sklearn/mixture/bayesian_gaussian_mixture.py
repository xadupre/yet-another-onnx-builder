from typing import Dict, List, Tuple

import numpy as np
from scipy.special import digamma
from sklearn.mixture import BayesianGaussianMixture

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from .gaussian_mixture import _sklearn_mixture_core


@register_sklearn_converter(BayesianGaussianMixture)
def sklearn_bayesian_gaussian_mixture(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: BayesianGaussianMixture,
    X: str,
    name: str = "bayesian_gaussian_mixture",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.mixture.BayesianGaussianMixture` into ONNX.

    The converter supports all four covariance types supported by
    :class:`~sklearn.mixture.BayesianGaussianMixture`:
    ``'full'``, ``'tied'``, ``'diag'``, and ``'spherical'``.

    At inference time a :class:`~sklearn.mixture.BayesianGaussianMixture`
    uses a variational Bayes approximation.  The weighted log-probability
    for sample *n* under component *k* is:

    .. code-block:: text

        log_p[n, k] = log_weight_k + log_det_k
                      - 0.5 * n_features * log(2π)
                      - 0.5 * quad[n, k]
                      - 0.5 * n_features * log(degrees_of_freedom_[k])
                      + 0.5 * (log_lambda_k - n_features / mean_precision_[k])

    where

    .. code-block:: text

        log_weight_k = _estimate_log_weights()[k]   (digamma-based)
        log_lambda_k = n_features * log(2)
                       + Σ_f digamma(0.5 * (dof_k - f))  for f in [0, n_features)

    The last two lines are constant per component and are folded into the
    ``c_k`` offset together with ``log_weight_k``, so at run-time only the
    same MatMul / ReduceSum operations as for
    :class:`~sklearn.mixture.GaussianMixture` are required.

    **'full'** — per-component Cholesky of the precision matrix
    ``L_k`` (shape ``(K, F, F)``):

    .. code-block:: text

        L_2d  = L.transpose(1,0,2).reshape(F, K*F)          # (F, K*F) constant
        b     = einsum('ki,kij->kj', means_, L)              # (K, F)   constant
        XL    = MatMul(X, L_2d)                              # (N, K*F)
        Y     = Reshape(XL - b, [-1, K, F])                  # (N, K, F)
        quad  = ReduceSum(Y * Y, axis=2)                     # (N, K)

    **'tied'** — single shared Cholesky ``L`` (shape ``(F, F)``):

    .. code-block:: text

        means_L = means_ @ L                                 # (K, F)  constant
        XL      = MatMul(X, L)                               # (N, F)
        Y       = Reshape(XL, [-1, 1, F]) - means_L          # (N, K, F)
        quad    = ReduceSum(Y * Y, axis=2)                   # (N, K)

    **'diag'** — per-component diagonal precision ``A = prec_chol**2``
    (shape ``(K, F)``):

    .. code-block:: text

        B     = means_ * A                                   # (K, F)  constant
        log_p = -0.5 * MatMul(X², Aᵀ) + MatMul(X, Bᵀ) + c  # (N, K)

    **'spherical'** — scalar precision ``prec = prec_chol**2`` per component
    (shape ``(K,)``):

    .. code-block:: text

        x_sq  = ReduceSum(X * X, axis=1, keepdims=1)        # (N, 1)
        cross = MatMul(X, means_ᵀ)                          # (N, K)
        log_p = prec * cross - 0.5 * prec * x_sq + c        # (N, K)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names; ``outputs[0]`` receives the predicted
        component labels and ``outputs[1]`` receives the posterior probabilities
    :param estimator: a fitted ``BayesianGaussianMixture``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: tuple ``(label_result_name, proba_result_name)``
    :raises NotImplementedError: for unsupported ``covariance_type`` values
    """
    assert isinstance(
        estimator, BayesianGaussianMixture
    ), f"Unexpected type {type(estimator)} for estimator."

    n_features = estimator.means_.shape[1]
    dof = estimator.degrees_of_freedom_  # (K,)

    # Variational log-lambda: n_features * log(2) + sum_f digamma(0.5*(dof-f))
    log_lambda = n_features * np.log(2.0) + np.sum(
        digamma(0.5 * (dof - np.arange(0, n_features)[:, np.newaxis])),
        axis=0,
    )  # (K,)

    # Extra constant correction per component (Wishart and mean-precision terms)
    extra_c = -0.5 * n_features * np.log(dof) + 0.5 * (
        log_lambda - n_features / estimator.mean_precision_
    )  # (K,)

    # Effective log-weights combine the variational Dirichlet log-weights with
    # the extra Wishart / mean-precision correction.
    log_weights_effective = estimator._estimate_log_weights() + extra_c  # (K,)

    return _sklearn_mixture_core(
        g, sts, outputs, estimator, X, name, log_weights=log_weights_effective,
    )
