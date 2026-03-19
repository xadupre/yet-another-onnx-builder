import numpy as np
from typing import Dict, List

from sklearn.manifold import LocallyLinearEmbedding

from ..register import register_sklearn_converter
from ..neighbors.kneighbors import _compute_pairwise_distances
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype

# Small constant added to denominators in the CG solver to avoid division by
# zero when the residual or the search-direction curvature is numerically zero.
_CG_EPS: float = 1e-10


def _batched_cg_solve(g: GraphBuilderExtendedProtocol, C: str, k: int, dtype, name: str) -> str:
    """
    Solve ``C @ w = ones`` for a batch of symmetric positive-definite matrices
    ``C`` of shape ``(N, k, k)`` using the Conjugate Gradient method, unrolled
    for exactly ``k`` iterations (enough for exact convergence on a ``k × k``
    system in exact arithmetic).

    :param g: graph builder
    :param C: name of the ``(N, k, k)`` SPD tensor
    :param k: system size (number of neighbours)
    :param dtype: numpy floating-point dtype (float32 or float64)
    :param name: node-name prefix
    :return: name of the ``(N, k)`` solution tensor
    """
    eps = np.array([_CG_EPS], dtype=dtype)

    # ── Initialise b = ones(N, k) by zeroing a row-sum of C and adding 1. ──
    # This avoids any Shape/Expand ops while giving the correct dynamic shape.
    b_zeros = g.op.Mul(
        g.op.ReduceSum(
            C, np.array([2], dtype=np.int64), keepdims=0, name=f"{name}_cg_rowsum"
        ),  # (N, k)
        np.array([0.0], dtype=dtype),
        name=f"{name}_cg_zeros",
    )  # (N, k) — all zeros
    b = g.op.Add(b_zeros, np.array([1.0], dtype=dtype), name=f"{name}_cg_b")  # (N, k)

    # x0 = 0,  r0 = b,  p0 = b
    x = b_zeros  # (N, k) — all zeros (reuse)
    r = b  # r0 = b - C @ 0 = b
    p = b  # p0 = r0

    for it in range(k):
        pfx = f"{name}_cg{it}"

        # Cp = C @ p  →  (N, k, k) @ (N, k, 1) = (N, k, 1)  →  (N, k)
        p_col = g.op.Unsqueeze(p, np.array([2], dtype=np.int64), name=f"{pfx}_pcol")
        Cp_col = g.op.MatMul(C, p_col, name=f"{pfx}_Cp_col")
        Cp = g.op.Squeeze(Cp_col, np.array([2], dtype=np.int64), name=f"{pfx}_Cp")

        # rr  = <r,  r>  per sample  (N,)
        rr = g.op.ReduceSum(
            g.op.Mul(r, r, name=f"{pfx}_rr_sq"),
            np.array([1], dtype=np.int64),
            keepdims=0,
            name=f"{pfx}_rr",
        )
        # pCp = <p, Cp>  per sample  (N,)
        pCp = g.op.ReduceSum(
            g.op.Mul(p, Cp, name=f"{pfx}_pCp_sq"),
            np.array([1], dtype=np.int64),
            keepdims=0,
            name=f"{pfx}_pCp",
        )

        # alpha = rr / (pCp + eps)  (N,)
        alpha = g.op.Div(rr, g.op.Add(pCp, eps, name=f"{pfx}_pCp_safe"), name=f"{pfx}_alpha")
        alpha_2d = g.op.Unsqueeze(
            alpha, np.array([1], dtype=np.int64), name=f"{pfx}_alpha2d"
        )  # (N, 1)

        # x = x + alpha * p
        x = g.op.Add(x, g.op.Mul(alpha_2d, p, name=f"{pfx}_ap"), name=f"{pfx}_x")

        # r_new = r - alpha * Cp
        r_new = g.op.Sub(r, g.op.Mul(alpha_2d, Cp, name=f"{pfx}_aCp"), name=f"{pfx}_r")

        # rr_new = <r_new, r_new>  per sample  (N,)
        rr_new = g.op.ReduceSum(
            g.op.Mul(r_new, r_new, name=f"{pfx}_rrnew_sq"),
            np.array([1], dtype=np.int64),
            keepdims=0,
            name=f"{pfx}_rr_new",
        )

        # beta = rr_new / (rr + eps)  (N,)
        beta = g.op.Div(rr_new, g.op.Add(rr, eps, name=f"{pfx}_rr_safe"), name=f"{pfx}_beta")
        beta_2d = g.op.Unsqueeze(
            beta, np.array([1], dtype=np.int64), name=f"{pfx}_beta2d"
        )  # (N, 1)

        # p = r_new + beta * p
        p = g.op.Add(r_new, g.op.Mul(beta_2d, p, name=f"{pfx}_bp"), name=f"{pfx}_p")
        r = r_new

    return x


@register_sklearn_converter(LocallyLinearEmbedding)
def sklearn_locally_linear_embedding(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: LocallyLinearEmbedding,
    X: str,
    name: str = "lle",
) -> str:
    """
    Converts a :class:`sklearn.manifold.LocallyLinearEmbedding` into ONNX.

    The out-of-sample embedding follows the algorithm in
    :meth:`sklearn.manifold.LocallyLinearEmbedding.transform`:

    1. Find the *k* nearest training neighbours for each query point using
       the same distance metric as during fitting (default: Euclidean).

    2. Compute *barycentric reconstruction weights* — the coefficients
       ``w`` that minimise the local reconstruction error::

           minimize  ||x - w @ X_neighbours||²   s.t.  sum(w) = 1

       The (regularised) closed-form solution is:

       .. code-block:: text

           C      = v @ v.T + R * I_k        (Gram matrix, regularised)
           v      = x - X_neighbours         (local displacement vectors)
           R      = reg * trace(C)  if trace(C) > 0,  else  reg
           w_raw  = C⁻¹ @ ones_k             (solved by Conjugate Gradient)
           w      = w_raw / sum(w_raw)        (normalised to sum to 1)

    3. Apply weights to the training embedding::

           result = w @ embedding_            (N, n_components)

    The full ONNX graph is:

    .. code-block:: text

        X (N, F)
          │
          ├── pairwise Euclidean distances ──────────────────────► dists (N, M)
          │                                                               │
          │                                         TopK(k, largest=0) ──┘
          │                                                   │
          │                               nb_idx (N, k)       │
          │                                   │               │
          │   X_train (M, F) ──Gather──────────┘             │
          │                                                   │
          │   nb_feats (N, k, F)                              │
          │
          v = X[:, None, :] - nb_feats  → (N, k, F)
          │
          C = v @ v.T  → (N, k, k)
          │
          C_reg = C + R * I_k  → (N, k, k)
          │
          w_raw = CG(C_reg, ones)  → (N, k)
          │
          w = w_raw / sum(w_raw)  → (N, k)
          │
          embedding_ (M, n_comp) ──Gather(nb_idx)──► emb_nb (N, k, n_comp)
          │
          result = w[:, None, :] @ emb_nb  → (N, n_components)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names (embedded inputs)
    :param estimator: a fitted ``LocallyLinearEmbedding``
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output tensor name
    """
    assert isinstance(
        estimator, LocallyLinearEmbedding
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # ── Extract fitted attributes ───────────────────────────────────────────
    training_data = estimator.nbrs_._fit_X.astype(dtype)  # (M, F)
    embedding = estimator.embedding_.astype(dtype)  # (M, n_components)
    k = int(estimator.n_neighbors)
    n_components = int(estimator.n_components)
    reg = float(estimator.reg)

    metric = estimator.nbrs_.effective_metric_
    metric_params = dict(estimator.nbrs_.effective_metric_params_)

    _n_train = int(training_data.shape[0])
    n_features = int(training_data.shape[1])

    # ── Step 1: Pairwise distances X → training data (N, M) ────────────────
    dists = _compute_pairwise_distances(
        g, X, training_data, itype, metric, f"{name}_dist", **metric_params
    )

    # ── Step 2: k nearest neighbours (indices, no distances needed) ─────────
    _nb_dists, nb_idx = g.op.TopK(
        dists, np.array([k], dtype=np.int64), axis=1, largest=0, sorted=1, name=f"{name}_topk"
    )  # nb_idx: (N, k)

    # ── Step 3: Gather neighbour feature vectors ────────────────────────────
    nb_idx_flat = g.op.Reshape(
        nb_idx, np.array([-1], dtype=np.int64), name=f"{name}_idx_flat"
    )  # (N*k,)
    nb_feats_flat = g.op.Gather(
        training_data, nb_idx_flat, axis=0, name=f"{name}_nb_flat"
    )  # (N*k, F)
    nb_feats = g.op.Reshape(
        nb_feats_flat, np.array([-1, k, n_features], dtype=np.int64), name=f"{name}_nb"
    )  # (N, k, F)

    # ── Step 4: Local displacement vectors v = X[:, None, :] - neighbours ──
    X_unsq = g.op.Unsqueeze(X, np.array([1], dtype=np.int64), name=f"{name}_X_unsq")  # (N, 1, F)
    v = g.op.Sub(X_unsq, nb_feats, name=f"{name}_v")  # (N, k, F)

    # ── Step 5: Gram matrix C = v @ v.T ─────────────────────────────────────
    # v: (N, k, F)  →  v.T: (N, F, k)  →  C = v @ v.T: (N, k, k)
    v_T = g.op.Transpose(v, perm=[0, 2, 1], name=f"{name}_vT")  # (N, F, k)
    C = g.op.MatMul(v, v_T, name=f"{name}_C")  # (N, k, k)

    # ── Step 6: Regularisation (mirrors sklearn's barycenter_weights) ───────
    # trace(C) = ||v||_F² per sample  =  ReduceSum(v², axes=(1,2))
    trace = g.op.ReduceSum(
        g.op.Mul(v, v, name=f"{name}_vsq"),
        np.array([1, 2], dtype=np.int64),
        keepdims=0,
        name=f"{name}_trace",
    )  # (N,)

    reg_arr = np.array([reg], dtype=dtype)
    zero_f = np.array([0.0], dtype=dtype)
    # R = reg * trace  if trace > 0  else  reg
    R = g.op.Where(
        g.op.Greater(trace, zero_f, name=f"{name}_trace_pos"),
        g.op.Mul(trace, reg_arr, name=f"{name}_reg_trace"),
        reg_arr,
        name=f"{name}_R",
    )  # (N,)

    # C_reg = C + R * I_k  (add scaled identity to each batch element)
    eye_k = np.eye(k, dtype=dtype)  # (k, k)
    R_3d = g.op.Reshape(R, np.array([-1, 1, 1], dtype=np.int64), name=f"{name}_R_3d")  # (N,1,1)
    C_reg = g.op.Add(
        C, g.op.Mul(R_3d, eye_k, name=f"{name}_R_eye"), name=f"{name}_C_reg"
    )  # (N, k, k)

    # ── Step 7: Solve C_reg @ w = ones via Conjugate Gradient ───────────────
    w_raw = _batched_cg_solve(g, C_reg, k, dtype, name)  # (N, k)

    # ── Step 8: Normalise weights to sum to 1 (as in barycenter_weights) ────
    w_sum = g.op.ReduceSum(
        w_raw, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_wsum"
    )  # (N, 1)
    w_norm = g.op.Div(w_raw, w_sum, name=f"{name}_wnorm")  # (N, k)

    # ── Step 9: Apply normalised weights to the training embedding ───────────
    # Gather embedding rows for all flattened neighbour indices (N*k,)
    emb_flat = g.op.Gather(
        embedding, nb_idx_flat, axis=0, name=f"{name}_emb_flat"
    )  # (N*k, n_components)
    emb_nb = g.op.Reshape(
        emb_flat, np.array([-1, k, n_components], dtype=np.int64), name=f"{name}_emb_nb"
    )  # (N, k, n_components)

    # result[n] = w_norm[n] @ emb_nb[n]
    #   w_norm (N, k)  →  unsqueeze  →  (N, 1, k)
    #   (N, 1, k) @ (N, k, n_comp)  =  (N, 1, n_comp)  →  squeeze  →  (N, n_comp)
    w_2d = g.op.Unsqueeze(w_norm, np.array([1], dtype=np.int64), name=f"{name}_w_2d")  # (N, 1, k)
    result_3d = g.op.MatMul(w_2d, emb_nb, name=f"{name}_mm")  # (N, 1, n_components)
    res = g.op.Squeeze(
        result_3d, np.array([1], dtype=np.int64), name=name, outputs=outputs
    )  # (N, n_components)

    assert isinstance(res, str)
    g.set_type(res, itype)
    if g.has_shape(X):
        batch_dim = g.get_shape(X)[0]
        g.set_shape(res, (batch_dim, n_components))
    return res
