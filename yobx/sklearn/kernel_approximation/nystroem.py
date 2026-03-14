import numpy as np
from typing import Dict, List

from sklearn.kernel_approximation import Nystroem

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(Nystroem)
def sklearn_nystroem(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: Nystroem,
    X: str,
    name: str = "nystroem",
) -> str:
    """
    Converts a :class:`sklearn.kernel_approximation.Nystroem` into ONNX.

    The out-of-sample transform replicates
    :meth:`sklearn.kernel_approximation.Nystroem.transform`:

    1. Compute the pairwise kernel matrix between *X* (shape ``(N, F)``) and
       the stored components ``components_`` (shape ``(C, F)``):

       * **linear** — ``K = X @ components_.T``
       * **rbf** — ``K[i,j] = exp(-gamma * ||X[i] - components_[j]||^2)``
       * **poly** — ``K = (gamma * X @ components_.T + coef0) ^ degree``
       * **sigmoid** — ``K = tanh(gamma * X @ components_.T + coef0)``
       * **cosine** — ``K[i,j] = (X[i]/||X[i]||) * (components_[j]/||components_[j]||)``

       Callable and ``precomputed`` kernels are not supported.

    2. Project using the normalization matrix stored in ``normalization_``:

       .. code-block:: text

           result = K @ normalization_.T     # (N, C)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``Nystroem``
    :param outputs: desired output names (transformed inputs)
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output tensor name
    :raises NotImplementedError: for callable kernels, ``kernel='precomputed'``,
        or unsupported kernel names
    """
    assert isinstance(estimator, Nystroem), (
        f"Unexpected type {type(estimator)} for estimator."
    )
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    kernel = estimator.kernel
    if callable(kernel):
        raise NotImplementedError(
            "Nystroem with a callable kernel is not supported by the ONNX converter."
        )
    if kernel == "precomputed":
        raise NotImplementedError(
            "Nystroem with kernel='precomputed' is not supported by the ONNX converter."
        )

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # ── Extract fitted attributes ────────────────────────────────────────────
    components = estimator.components_.astype(dtype)  # (C, F)
    n_components = int(components.shape[0])
    n_features = int(components.shape[1])
    normalization_T = estimator.normalization_.T.astype(dtype)  # (C, C)

    # Resolve kernel hyper-parameters (falling back to sklearn defaults
    # when the user left them as None).
    gamma_val = estimator.gamma
    if gamma_val is None:
        gamma_val = 1.0 / n_features
    gamma = np.array([gamma_val], dtype=dtype)

    coef0_val = estimator.coef0
    if coef0_val is None:
        coef0_val = 1.0
    coef0 = np.array([coef0_val], dtype=dtype)

    degree_val = estimator.degree
    if degree_val is None:
        degree_val = 3
    degree = int(degree_val)

    # ── Step 1: Compute kernel matrix K (N, C) ──────────────────────────────
    if kernel == "linear":
        K = g.op.MatMul(X, components.T, name=f"{name}_dot")

    elif kernel == "rbf":
        # Pairwise squared-Euclidean distances: ||X[i] - components[j]||^2
        # Using the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
        X_sq = g.op.ReduceSum(
            g.op.Mul(X, X, name=f"{name}_xsq_inner"),
            np.array([1], dtype=np.int64),
            keepdims=1,
            name=f"{name}_xsq",
        )  # (N, 1)
        comp_sq = (components**2).sum(axis=1, keepdims=True).T.astype(dtype)  # (1, C)
        cross = g.op.MatMul(X, components.T, name=f"{name}_cross")
        neg_two_cross = g.op.Mul(cross, np.array([-2.0], dtype=dtype), name=f"{name}_neg2cross")
        sq_dists = g.op.Max(
            g.op.Add(
                g.op.Add(X_sq, comp_sq, name=f"{name}_sq_sum"),
                neg_two_cross,
                name=f"{name}_sqdist",
            ),
            np.array([0.0], dtype=dtype),
            name=f"{name}_sqdist_clamp",
        )
        neg_gamma = np.array([-gamma_val], dtype=dtype)
        K = g.op.Exp(
            g.op.Mul(sq_dists, neg_gamma, name=f"{name}_neg_scaled"),
            name=f"{name}_rbf_k",
        )

    elif kernel == "poly":
        dot = g.op.MatMul(X, components.T, name=f"{name}_dot")
        inner = g.op.Add(
            g.op.Mul(dot, gamma, name=f"{name}_poly_scaled"),
            coef0,
            name=f"{name}_poly_inner",
        )
        K = g.op.Pow(inner, np.array([float(degree)], dtype=dtype), name=f"{name}_poly_k")

    elif kernel == "sigmoid":
        dot = g.op.MatMul(X, components.T, name=f"{name}_dot")
        inner = g.op.Add(
            g.op.Mul(dot, gamma, name=f"{name}_sig_scaled"),
            coef0,
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
        X_normed = g.op.Div(
            X,
            g.op.Sqrt(
                g.op.Max(X_sq_sum, eps_arr, name=f"{name}_cos_xsq_clamp"),
                name=f"{name}_cos_xnorm",
            ),
            name=f"{name}_cos_xnormed",
        )
        # Normalize components at conversion time (static)
        comp_norms = np.linalg.norm(components, axis=1, keepdims=True)
        comp_norms = np.where(comp_norms == 0, 1.0, comp_norms).astype(dtype)
        comp_normed = (components / comp_norms).astype(dtype)
        K = g.op.MatMul(X_normed, comp_normed.T, name=f"{name}_cos_k")

    else:
        raise NotImplementedError(
            f"Nystroem kernel {kernel!r} is not supported by the ONNX converter. "
            f"Supported kernels: 'linear', 'rbf', 'poly', 'sigmoid', 'cosine'."
        )

    # ── Step 2: Project using normalization_.T ───────────────────────────────
    res = g.op.MatMul(K, normalization_T, name=name, outputs=outputs)

    assert isinstance(res, str)  # type happiness
    if not sts:
        g.set_type(res, itype)
        if g.has_shape(X):
            batch_dim = g.get_shape(X)[0]
            g.set_shape(res, (batch_dim, n_components))
    return res
