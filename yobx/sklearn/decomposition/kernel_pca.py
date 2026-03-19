import numpy as np
from typing import Dict, List

from sklearn.decomposition import KernelPCA

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype

# Kernels whose computation requires the dot product X @ X_fit_.T.
_DOT_KERNELS = frozenset({"linear", "poly", "sigmoid"})

# Kernels whose computation requires pairwise squared-Euclidean distances.
_DIST_KERNELS = frozenset({"rbf"})


def _compute_dot_product(
    g: GraphBuilderExtendedProtocol, X: str, X_fit: np.ndarray, name: str
) -> str:
    """Return the ONNX node computing ``X @ X_fit_.T`` (shape ``(N, M)``)."""
    return g.op.MatMul(X, X_fit.T, name=name)


def _compute_sq_euclidean(
    g: GraphBuilderExtendedProtocol, X: str, X_fit: np.ndarray, dtype, itype: int, name: str
) -> str:
    """Return the ONNX node computing pairwise squared-Euclidean distances
    ``||X[i] - X_fit[j]||^2`` (shape ``(N, M)``).

    Uses the identity:
    ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y``
    """
    # ||X[i]||^2  → (N, 1)
    X_sq = g.op.ReduceSum(
        g.op.Mul(X, X, name=f"{name}_xsq_inner"),
        np.array([1], dtype=np.int64),
        keepdims=1,
        name=f"{name}_xsq",
    )
    # ||X_fit[j]||^2  → (1, M)  (static, folded as constant)
    Xfit_sq = (X_fit**2).sum(axis=1, keepdims=True).T.astype(dtype)  # (1, M)

    # Cross term −2 X @ X_fit.T  → (N, M)
    cross = g.op.MatMul(X, X_fit.T, name=f"{name}_cross")
    two = np.array([2.0], dtype=dtype)
    neg_two_cross = g.op.Mul(cross, -two, name=f"{name}_neg2cross")

    # ||X[i]||^2 + ||X_fit[j]||^2 - 2 x·y
    sq_dists = g.op.Add(
        g.op.Add(X_sq, Xfit_sq, name=f"{name}_sq_sum"), neg_two_cross, name=f"{name}_sqdist"
    )
    # Clamp to 0 to avoid tiny negative values due to floating-point rounding
    zero = np.array([0.0], dtype=dtype)
    return g.op.Max(sq_dists, zero, name=f"{name}_sqdist_clamp")


@register_sklearn_converter(KernelPCA)
def sklearn_kernel_pca(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: KernelPCA,
    X: str,
    name: str = "kernel_pca",
) -> str:
    """
    Converts a :class:`sklearn.decomposition.KernelPCA` into ONNX.

    The out-of-sample transform replicates :meth:`sklearn.decomposition.KernelPCA.transform`:

    1. Compute the pairwise kernel matrix between *X* (shape ``(N, F)``) and the
       training data ``X_fit_`` (shape ``(M, F)``):

       * **linear** — ``K = X @ X_fit_.T``
       * **rbf** — ``K[i,j] = exp(−γ · ||X[i] − X_fit_[j]||²)``
       * **laplacian** — ``K[i,j] = exp(−γ · ||X[i] − X_fit_[j]||₁)``  *(not
         supported; raises* ``NotImplementedError`` *)*
       * **poly** — ``K = (γ · X @ X_fit_.T + coef0) ^ degree``
       * **sigmoid** — ``K = tanh(γ · X @ X_fit_.T + coef0)``
       * **cosine** — ``K[i,j] = (X[i]/‖X[i]‖) · (X_fit_[j]/‖X_fit_[j]‖)``
       * **precomputed** / callable — not supported; raises ``NotImplementedError``

    2. Centre the kernel using the statistics stored in ``_centerer``:

       .. code-block:: text

           K_pred_cols = K.sum(axis=1, keepdims=True) / M
           K_centered  = K − K_fit_rows_ − K_pred_cols + K_fit_all_

    3. Compute scaled eigenvectors (zero-eigenvalue columns set to 0):

       .. code-block:: text

           scaled_alphas[:, non_zeros] = eigenvectors_[:, non_zeros]
                                         / sqrt(eigenvalues_[non_zeros])

    4. Project:

       .. code-block:: text

           result = K_centered @ scaled_alphas     # (N, n_components)

    The full ONNX graph (rbf example) is:

    .. code-block:: text

        X (N, F)
          │
          └── ||X[i] − X_fit_[j]||² ────────────────────────────► sq_dists (N, M)
                                                                          │
               sq_dists × (−γ) ──► neg_scaled ──Exp──► K (N, M)
               │
           K − K_fit_rows_ − K.sum(axis=1)/M + K_fit_all_ ──► K_c (N, M)
               │
           MatMul(scaled_alphas) ──► output (N, n_components)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``KernelPCA``
    :param outputs: desired output names
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output tensor name
    :raises NotImplementedError: for ``kernel='precomputed'``, callable kernels,
        or ``kernel='laplacian'``
    """
    assert isinstance(estimator, KernelPCA), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    kernel = estimator.kernel
    if callable(kernel):
        raise NotImplementedError(
            "KernelPCA with a callable kernel is not supported by the ONNX converter."
        )
    if kernel == "precomputed":
        raise NotImplementedError(
            "KernelPCA with kernel='precomputed' is not supported by the ONNX converter."
        )
    if kernel == "laplacian":
        raise NotImplementedError(
            "KernelPCA with kernel='laplacian' is not currently supported by the ONNX converter."
        )

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # ── Extract fitted attributes ────────────────────────────────────────────
    X_fit = estimator.X_fit_.astype(dtype)  # (M, F)
    n_train = int(X_fit.shape[0])

    gamma = np.array(estimator.gamma_, dtype=dtype)
    coef0 = np.array(estimator.coef0, dtype=dtype)
    degree = int(estimator.degree)

    # KernelCenterer statistics
    ctr = estimator._centerer
    K_fit_rows = ctr.K_fit_rows_.astype(dtype).reshape(1, -1)  # (1, M) for broadcast
    K_fit_all = np.array([ctr.K_fit_all_], dtype=dtype)  # scalar

    # Scaled eigenvectors: shape (M, n_components)
    eigenvalues = estimator.eigenvalues_
    eigenvectors = estimator.eigenvectors_
    non_zeros = np.flatnonzero(eigenvalues)
    scaled_alphas = np.zeros_like(eigenvectors, dtype=dtype)
    scaled_alphas[:, non_zeros] = (
        eigenvectors[:, non_zeros] / np.sqrt(eigenvalues[non_zeros])
    ).astype(dtype)

    # ── Step 1: Compute kernel matrix K (N, M) ──────────────────────────────
    if kernel == "linear":
        K = _compute_dot_product(g, X, X_fit, f"{name}_dot")

    elif kernel == "rbf":
        sq_dists = _compute_sq_euclidean(g, X, X_fit, dtype, itype, f"{name}_rbf")
        neg_gamma_arr = np.array([-gamma], dtype=dtype)
        K = g.op.Exp(
            g.op.Mul(sq_dists, neg_gamma_arr, name=f"{name}_rbf_neg_scaled"), name=f"{name}_rbf_k"
        )

    elif kernel == "poly":
        dot = _compute_dot_product(g, X, X_fit, f"{name}_dot")
        gamma_arr = np.array([gamma], dtype=dtype)
        coef0_arr = np.array([coef0], dtype=dtype)
        inner = g.op.Add(
            g.op.Mul(dot, gamma_arr, name=f"{name}_poly_scaled"),
            coef0_arr,
            name=f"{name}_poly_inner",
        )
        degree_arr = np.array([float(degree)], dtype=dtype)
        K = g.op.Pow(inner, degree_arr, name=f"{name}_poly_k")

    elif kernel == "sigmoid":
        dot = _compute_dot_product(g, X, X_fit, f"{name}_dot")
        gamma_arr = np.array([gamma], dtype=dtype)
        coef0_arr = np.array([coef0], dtype=dtype)
        inner = g.op.Add(
            g.op.Mul(dot, gamma_arr, name=f"{name}_sig_scaled"),
            coef0_arr,
            name=f"{name}_sig_inner",
        )
        K = g.op.Tanh(inner, name=f"{name}_sig_k")

    elif kernel == "cosine":
        # Normalize X rows: X_norm[i] = X[i] / ||X[i]||  (handle zero rows)
        X_sq_sum = g.op.ReduceSum(
            g.op.Mul(X, X, name=f"{name}_cos_xsq_inner"),
            np.array([1], dtype=np.int64),
            keepdims=1,
            name=f"{name}_cos_xsq",
        )  # (N, 1)
        eps_arr = np.array([np.finfo(dtype).tiny], dtype=dtype)
        X_norm_denom = g.op.Sqrt(
            g.op.Max(X_sq_sum, eps_arr, name=f"{name}_cos_xsq_clamp"), name=f"{name}_cos_xnorm"
        )
        X_normed = g.op.Div(X, X_norm_denom, name=f"{name}_cos_xnormed")

        # Normalize X_fit rows at conversion time (static)
        X_fit_norms = np.linalg.norm(X_fit, axis=1, keepdims=True)
        X_fit_norms = np.where(X_fit_norms == 0, 1.0, X_fit_norms).astype(dtype)
        X_fit_normed = (X_fit / X_fit_norms).astype(dtype)

        K = g.op.MatMul(X_normed, X_fit_normed.T, name=f"{name}_cos_k")

    else:
        raise NotImplementedError(
            f"KernelPCA kernel {kernel!r} is not supported by the ONNX converter. "
            f"Supported kernels: 'linear', 'rbf', 'poly', 'sigmoid', 'cosine'."
        )

    # ── Step 2: KernelCenterer.transform ────────────────────────────────────
    # K_pred_cols = K.sum(axis=1, keepdims=True) / n_train   → (N, 1)
    n_train_arr = np.array([float(n_train)], dtype=dtype)
    K_row_sum = g.op.ReduceSum(
        K, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_rowsum"
    )  # (N, 1)
    K_pred_cols = g.op.Div(K_row_sum, n_train_arr, name=f"{name}_pred_cols")

    # K_centered = K - K_fit_rows_ - K_pred_cols + K_fit_all_
    K_sub_rows = g.op.Sub(K, K_fit_rows, name=f"{name}_sub_rows")
    K_sub_cols = g.op.Sub(K_sub_rows, K_pred_cols, name=f"{name}_sub_cols")
    K_centered = g.op.Add(K_sub_cols, K_fit_all, name=f"{name}_centered")

    # ── Step 3: Project onto scaled eigenvectors ─────────────────────────────
    res = g.op.MatMul(K_centered, scaled_alphas, name=name, outputs=outputs)

    assert isinstance(res, str)  # type happiness
    g.set_type(res, itype)
    if g.has_shape(X):
        batch_dim = g.get_shape(X)[0]
        g.set_shape(res, (batch_dim, int(estimator.n_components)))
    return res
