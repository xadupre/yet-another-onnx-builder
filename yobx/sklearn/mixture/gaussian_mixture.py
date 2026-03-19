from typing import Dict, List, Tuple, Union

import numpy as np
import onnx
from sklearn.mixture import GaussianMixture
from sklearn.mixture._base import BaseMixture

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


def _sklearn_mixture_core(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: BaseMixture,
    X: str,
    name: str,
    log_weights: Union[None, "np.ndarray"] = None,
) -> Tuple[str, str]:
    """
    Shared implementation for Gaussian mixture model converters.

    Computes the weighted log-probability for each component and returns
    the predicted label (``ArgMax``) and posterior probabilities
    (``Softmax``).  The computation is identical for
    :class:`~sklearn.mixture.GaussianMixture` and
    :class:`~sklearn.mixture.BayesianGaussianMixture` because both expose
    the same fitted attributes used during inference (``means_``,
    ``weights_``, ``precisions_cholesky_``, ``covariance_type``).

    The weighted log-probability for sample *n* under component *k* is:

    .. code-block:: text

        log_p[n, k] = log(weight_k) + log_det_k
                      - 0.5 * n_features * log(2π)
                      - 0.5 * quad[n, k]

    where ``quad[n, k]`` is the Mahalanobis distance squared, computed
    differently depending on ``covariance_type``.

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
    :param outputs: desired output names
    :param estimator: a fitted mixture model (``GaussianMixture`` or
        ``BayesianGaussianMixture``)
    :param X: input tensor name
    :param name: prefix for added node names
    :param log_weights: precomputed log-weights of shape ``(K,)``; when
        *None* the default ``np.log(estimator.weights_)`` is used.
        Pass ``estimator._estimate_log_weights()`` for
        ``BayesianGaussianMixture`` which uses digamma-based variational
        log-weights instead.
    :return: tuple ``(label_result_name, proba_result_name)``
    :raises NotImplementedError: for unsupported ``covariance_type`` values
    """
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    K = estimator.n_components
    F = estimator.means_.shape[1]
    means = estimator.means_.astype(dtype)  # (K, F)
    if log_weights is None:
        log_weights = np.log(estimator.weights_).astype(dtype)  # (K,)
    else:
        log_weights = np.asarray(log_weights).astype(dtype)  # (K,)
    cov_type = estimator.covariance_type
    prec_chol = estimator.precisions_cholesky_
    half = np.array([0.5], dtype=dtype)

    if cov_type == "full":
        L = prec_chol.astype(dtype)  # (K, F, F)
        # log |L_k| = sum of log(diag(L_k))
        log_det = np.array([np.sum(np.log(np.diag(L[k]))) for k in range(K)], dtype=dtype)  # (K,)
        c = (log_weights + log_det - 0.5 * F * np.log(2.0 * np.pi)).astype(dtype)  # (K,)

        # Precompute b[k] = mu[k] @ L[k] for all k → (K, F)
        b = np.einsum("ki,kij->kj", means, L).astype(dtype)  # (K, F)

        # Reshape L: (K, F, F) → transpose(1,0,2): (F, K, F) → (F, K*F)
        L_2d = L.transpose(1, 0, 2).reshape(F, K * F).astype(dtype)  # (F, K*F)
        b_flat = b.reshape(1, K * F).astype(dtype)  # (1, K*F)

        XL = g.op.MatMul(X, L_2d, name=f"{name}_XL")  # (N, K*F)
        Y_flat = g.op.Sub(XL, b_flat, name=f"{name}_Y_flat")  # (N, K*F)
        Y_sq = g.op.Mul(Y_flat, Y_flat, name=f"{name}_Y_sq")  # (N, K*F)
        Y_sq_3d = g.op.Reshape(
            Y_sq, np.array([-1, K, F], dtype=np.int64), name=f"{name}_Y_sq_3d"
        )  # (N, K, F)
        quad = g.op.ReduceSum(
            Y_sq_3d, np.array([2], dtype=np.int64), keepdims=0, name=f"{name}_quad"
        )  # (N, K)
        log_p = g.op.Add(
            g.op.Mul(g.op.Neg(half, name=f"{name}_neg_half"), quad, name=f"{name}_nq"),
            c,
            name=f"{name}_log_p",
        )  # (N, K)

    elif cov_type == "tied":
        L = prec_chol.astype(dtype)  # (F, F)
        log_det = np.sum(np.log(np.diag(L))).astype(dtype)  # scalar
        c = (log_weights + log_det - 0.5 * F * np.log(2.0 * np.pi)).astype(dtype)  # (K,)

        # Precompute means_L = means @ L → (K, F)
        means_L = (means @ L).astype(dtype)  # (K, F)

        XL = g.op.MatMul(X, L, name=f"{name}_XL")  # (N, F)
        # Reshape XL to (N, 1, F) then subtract means_L (K, F) → (N, K, F)
        XL_3d = g.op.Reshape(
            XL, np.array([-1, 1, F], dtype=np.int64), name=f"{name}_XL_3d"
        )  # (N, 1, F)
        Y = g.op.Sub(XL_3d, means_L, name=f"{name}_Y")  # (N, K, F)
        Y_sq = g.op.Mul(Y, Y, name=f"{name}_Y_sq")  # (N, K, F)
        quad = g.op.ReduceSum(
            Y_sq, np.array([2], dtype=np.int64), keepdims=0, name=f"{name}_quad"
        )  # (N, K)
        log_p = g.op.Add(
            g.op.Mul(g.op.Neg(half, name=f"{name}_neg_half"), quad, name=f"{name}_nq"),
            c,
            name=f"{name}_log_p",
        )  # (N, K)

    elif cov_type == "diag":
        L = prec_chol.astype(dtype)  # (K, F)
        A = (L**2).astype(dtype)  # (K, F) - precisions
        log_det = np.sum(np.log(L), axis=1).astype(dtype)  # (K,)
        mu_sq_prec = np.sum(means**2 * A, axis=1).astype(dtype)  # (K,)
        c = (log_weights + log_det - 0.5 * F * np.log(2.0 * np.pi) - 0.5 * mu_sq_prec).astype(
            dtype
        )  # (K,)

        B = (means * A).astype(dtype)  # (K, F)

        X2 = g.op.Mul(X, X, name=f"{name}_X2")  # (N, F)
        term1 = g.op.Mul(
            half, g.op.MatMul(X2, A.T, name=f"{name}_X2_dot_AT"), name=f"{name}_half_X2_dot_AT"
        )
        term2 = g.op.MatMul(X, B.T, name=f"{name}_X_dot_BT")  # (N, K)
        log_p = g.op.Add(
            g.op.Sub(term2, term1, name=f"{name}_sub"), c, name=f"{name}_log_p"
        )  # (N, K)

    elif cov_type == "spherical":
        L = prec_chol.astype(dtype)  # (K,)
        prec = (L**2).astype(dtype)  # (K,)
        log_det = (F * np.log(L)).astype(dtype)  # (K,)
        mu_sq = np.sum(means**2, axis=1).astype(dtype)  # (K,)
        c = (log_weights + log_det - 0.5 * F * np.log(2.0 * np.pi) - 0.5 * prec * mu_sq).astype(
            dtype
        )  # (K,)

        x_sq = g.op.ReduceSum(
            g.op.Mul(X, X, name=f"{name}_X2"),
            np.array([1], dtype=np.int64),
            keepdims=1,
            name=f"{name}_x_sq",
        )  # (N, 1)
        cross = g.op.MatMul(X, means.T, name=f"{name}_cross")  # (N, K)
        xsq_term = g.op.Mul(
            g.op.Mul(half, prec, name=f"{name}_half_prec"), x_sq, name=f"{name}_xsq_term"
        )  # (N, K)
        cross_term = g.op.Mul(prec, cross, name=f"{name}_cross_term")  # (N, K)
        log_p = g.op.Add(
            g.op.Sub(cross_term, xsq_term, name=f"{name}_before_c"), c, name=f"{name}_log_p"
        )  # (N, K)

    else:
        raise NotImplementedError(
            f"{type(estimator).__name__} converter: unsupported covariance_type={cov_type!r}."
        )

    proba = g.op.Softmax(log_p, axis=1, name=f"{name}_proba", outputs=outputs[1:])
    label_idx = g.op.ArgMax(log_p, axis=1, keepdims=0, name=f"{name}_argmax")
    label = g.op.Cast(
        label_idx, to=onnx.TensorProto.INT64, name=f"{name}_label", outputs=outputs[:1]
    )
    g.set_type(label, onnx.TensorProto.INT64)
    g.set_type(proba, itype)
    return label, proba


@register_sklearn_converter(GaussianMixture)
def sklearn_gaussian_mixture(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: GaussianMixture,
    X: str,
    name: str = "gaussian_mixture",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.mixture.GaussianMixture` into ONNX.

    The converter supports all four covariance types supported by
    :class:`~sklearn.mixture.GaussianMixture`:
    ``'full'``, ``'tied'``, ``'diag'``, and ``'spherical'``.

    In each case the weighted log-probability for sample *n* under component *k*
    is computed as:

    .. code-block:: text

        log_p[n, k] = log(weight_k) + log_det_k
                      - 0.5 * n_features * log(2π)
                      - 0.5 * quad[n, k]

    where ``quad[n, k]`` is the Mahalanobis distance squared, computed
    differently depending on ``covariance_type``.  ``label`` is the
    ``ArgMax`` of ``log_p`` and ``proba`` is its ``Softmax``.

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
    :param estimator: a fitted ``GaussianMixture``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: tuple ``(label_result_name, proba_result_name)``
    :raises NotImplementedError: for unsupported ``covariance_type`` values
    """
    assert isinstance(
        estimator, GaussianMixture
    ), f"Unexpected type {type(estimator)} for estimator."
    return _sklearn_mixture_core(g, sts, outputs, estimator, X, name)
