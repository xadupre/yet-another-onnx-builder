import numpy as np
from typing import Dict, List

from sklearn.manifold import Isomap

from ..register import register_sklearn_converter
from ..neighbors.kneighbors import _compute_pairwise_distances
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(Isomap)
def sklearn_isomap(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: Isomap,
    X: str,
    name: str = "isomap",
) -> str:
    """
    Converts a :class:`sklearn.manifold.Isomap` into ONNX.

    The out-of-sample embedding follows the algorithm in
    :meth:`sklearn.manifold.Isomap.transform`:

    1. Find the *k* nearest training neighbours for each query point using
       the same distance metric as during fitting (default: Euclidean).
    2. Approximate geodesic distances from every query point to all training
       points by combining the exact neighbour distances with the pre-computed
       training-graph shortest-path matrix (``dist_matrix_``):

       .. code-block:: text

           G_X[i, j] = min over neighbor nb of
                         (eucl_dist(X[i], X_train[nb]) + dist_matrix_[nb, j])

    3. Convert geodesic distances to kernel values: ``K = -0.5 * G_X ** 2``.

    4. Centre the kernel matrix using the statistics stored in the fitted
       :class:`~sklearn.preprocessing.KernelCenterer`:

       .. code-block:: text

           K_centered = K - K_fit_rows_ - K.mean(axis=1, keepdims=True) + K_fit_all_

    5. Project onto the scaled eigenvectors of the training kernel:

       .. code-block:: text

           result = K_centered @ (eigenvectors_ / sqrt(eigenvalues_))

    The full ONNX graph (standard-ONNX path) is:

    .. code-block:: text

        X (N, F)
          │
          └── pairwise Euclidean distances ──────────────────► dists (N, M)
                                                                      │
                                          TopK(k, largest=0) ─────────┤
                                                                  │   │
                                              nb_dists (N, k)    │   │
                                              nb_idx   (N, k)    │   │
                                                       │         │   │
         dist_matrix_ (M, M) ──Gather(nb_idx_flat)────┘         │
                  │                 │                            │
                  │   Reshape(-1,k,M) + Unsqueeze(nb_dists,2) ──┘
                  │                 │
                  │             Add → ReduceMin(axis=1) ──► G_X (N, M)
                  │
                  G_X²  × (−0.5) ──► K (N, M)
                  │
              K − K_fit_rows_ − K.mean(axis=1) + K_fit_all_ ──► K_c (N, M)
                  │
              MatMul(scaled_alphas) ──► output (N, n_components)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``Isomap``
    :param outputs: desired output names
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output tensor name
    """
    assert isinstance(estimator, Isomap), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # ── Extract fitted attributes ───────────────────────────────────────────
    training_data = estimator.nbrs_._fit_X.astype(dtype)  # (M, F)
    dist_matrix = estimator.dist_matrix_.astype(dtype)  # (M, M)
    n_neighbors = int(estimator.n_neighbors)
    n_train = int(training_data.shape[0])

    metric = estimator.nbrs_.effective_metric_
    metric_params = dict(estimator.nbrs_.effective_metric_params_)

    # KernelCenterer statistics
    ctr = estimator.kernel_pca_._centerer
    K_fit_rows = ctr.K_fit_rows_.astype(dtype)  # (M,)
    K_fit_all = np.array([ctr.K_fit_all_], dtype=dtype)  # scalar

    # Scaled eigenvectors: shape (M, n_components)
    eigenvalues = estimator.kernel_pca_.eigenvalues_
    eigenvectors = estimator.kernel_pca_.eigenvectors_
    non_zeros = np.flatnonzero(eigenvalues)
    scaled_alphas = np.zeros_like(eigenvectors, dtype=dtype)
    scaled_alphas[:, non_zeros] = (
        eigenvectors[:, non_zeros] / np.sqrt(eigenvalues[non_zeros])
    ).astype(dtype)

    # ── Step 1: Pairwise distances X → training data (N, M) ────────────────
    dists = _compute_pairwise_distances(
        g, X, training_data, itype, metric, f"{name}_dist", **metric_params
    )

    # ── Step 2: k nearest neighbours ───────────────────────────────────────
    nb_dists, nb_idx = g.op.TopK(
        dists,
        np.array([n_neighbors], dtype=np.int64),
        axis=1,
        largest=0,
        sorted=1,
        name=f"{name}_topk",
    )  # nb_dists (N, k), nb_idx (N, k)

    # ── Step 3: Approximate geodesic distances G_X ─────────────────────────
    # Flatten indices → (N*k,), gather dist_matrix rows → (N*k, M)
    nb_idx_flat = g.op.Reshape(nb_idx, np.array([-1], dtype=np.int64), name=f"{name}_idx_flat")
    dist_rows = g.op.Gather(
        dist_matrix, nb_idx_flat, axis=0, name=f"{name}_dist_rows"
    )  # (N*k, M)

    # Reshape to (N, k, M) using -1 so N is inferred dynamically
    dist_rows_3d = g.op.Reshape(
        dist_rows,
        np.array([-1, n_neighbors, n_train], dtype=np.int64),
        name=f"{name}_dist_rows_3d",
    )  # (N, k, M)

    # Expand neighbour distances: (N, k) → (N, k, 1) for broadcasting
    nb_dists_3d = g.op.Unsqueeze(
        nb_dists, np.array([2], dtype=np.int64), name=f"{name}_nb_dists_3d"
    )  # (N, k, 1)

    # For each (i, nb, j): dist(X[i], X_train[nb]) + dist_matrix_[nb, j]
    extended = g.op.Add(dist_rows_3d, nb_dists_3d, name=f"{name}_extended")  # (N, k, M)

    # Geodesic distance = minimum over neighbours
    G_X = g.op.ReduceMin(
        extended, np.array([1], dtype=np.int64), keepdims=0, name=f"{name}_G_X"
    )  # (N, M)

    # ── Step 4: Kernel values K = -0.5 * G_X ** 2 ──────────────────────────
    G_X_sq = g.op.Mul(G_X, G_X, name=f"{name}_sq")
    half_neg = np.array([-0.5], dtype=dtype)
    K = g.op.Mul(G_X_sq, half_neg, name=f"{name}_kernel")  # (N, M)

    # ── Step 5: KernelCenterer.transform ───────────────────────────────────
    # K_pred_cols = K.sum(axis=1, keepdims=True) / n_train
    n_train_arr = np.array([float(n_train)], dtype=dtype)
    K_row_sum = g.op.ReduceSum(
        K, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_row_sum"
    )  # (N, 1)
    K_pred_cols = g.op.Div(K_row_sum, n_train_arr, name=f"{name}_pred_cols")  # (N, 1)

    # K_centered = K - K_fit_rows_ - K_pred_cols + K_fit_all_
    K_fit_rows_2d = K_fit_rows.reshape(1, -1)  # (1, M) for broadcasting
    K_sub_rows = g.op.Sub(K, K_fit_rows_2d, name=f"{name}_sub_rows")
    K_sub_cols = g.op.Sub(K_sub_rows, K_pred_cols, name=f"{name}_sub_cols")
    K_centered = g.op.Add(K_sub_cols, K_fit_all, name=f"{name}_centered")  # (N, M)

    # ── Step 6: Project onto scaled eigenvectors ────────────────────────────
    res = g.op.MatMul(K_centered, scaled_alphas, name=name, outputs=outputs)

    g.set_type(res, itype)
    if g.has_shape(X):
        batch_dim = g.get_shape(X)[0]
        g.set_shape(res, (batch_dim, int(estimator.n_components)))
    return res
