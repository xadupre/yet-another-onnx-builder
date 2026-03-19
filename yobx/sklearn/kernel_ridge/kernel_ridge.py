from typing import Dict, List

import numpy as np
from sklearn.kernel_ridge import KernelRidge

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


def _emit_kernel_ridge_matrix(
    g: GraphBuilderExtendedProtocol,
    kernel: str,
    gamma: float,
    degree: int,
    coef0: float,
    X: str,
    X_fit: np.ndarray,
    name: str,
    dtype,
) -> str:
    """
    Emit ONNX operations computing the kernel matrix ``K(X, X_fit)``.

    *X* is a graph node of shape ``(N, F)``; *X_fit* is a fixed numpy array
    of shape ``(M, F)`` stored as graph constants.  Returns a tensor name
    of shape ``(N, M)``.

    Supported kernels
    -----------------
    * ``'linear'``: ``K(x, y) = x · y``
    * ``'rbf'``: ``K(x, y) = exp(-γ ‖x - y‖²)``
    * ``'poly'`` / ``'polynomial'``: ``K(x, y) = (γ x · y + c₀)^d``
    * ``'sigmoid'``: ``K(x, y) = tanh(γ x · y + c₀)``
    * ``'cosine'``: ``K(x, y) = x · y / (‖x‖ ‖y‖)``
    * ``'laplacian'``: ``K(x, y) = exp(-γ ‖x - y‖₁)``
    * ``'chi2'``: ``K(x, y) = exp(-γ Σ (xᵢ - yᵢ)² / (xᵢ + yᵢ))``

    :param g: graph builder
    :param kernel: kernel name string
    :param gamma: effective gamma value (already resolved from ``None``)
    :param degree: polynomial degree (only used for ``'poly'`` / ``'polynomial'``)
    :param coef0: free term (only used for ``'poly'``, ``'polynomial'``, ``'sigmoid'``)
    :param X: input tensor name, shape ``(N, F)``
    :param X_fit: fixed training matrix, shape ``(M, F)``
    :param name: node name prefix
    :param dtype: numpy dtype
    :return: ONNX tensor name of shape ``(N, M)``
    :raises NotImplementedError: for callable kernels or ``'precomputed'``
    """
    if kernel == "linear":
        # K(X, Y) = X @ Y.T
        return g.op.MatMul(X, X_fit.T.astype(dtype), name=f"{name}_linear_K")

    elif kernel == "rbf":
        # K(X, Y) = exp(-gamma * ||X-Y||^2)
        if g.has_opset("com.microsoft"):
            X_fit_name = g.make_initializer(f"{name}_Xfit", X_fit.astype(dtype))
            D = g.make_node(
                "CDist",
                [X, X_fit_name],
                domain="com.microsoft",
                metric="sqeuclidean",
                name=f"{name}_cdist",
            )
            D = g.op.Max(D, np.array(0.0, dtype=dtype), name=f"{name}_Dclamp")
        else:
            # expand-trick:  ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x·y^T
            c_sq = np.sum(X_fit**2, axis=1)[np.newaxis, :].astype(dtype)  # (1, M)
            x_sq = g.op.Mul(X, X, name=f"{name}_xsq")
            x_sq_sum = g.op.ReduceSum(
                x_sq, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_xsqsum"
            )  # (N, 1)
            cross = g.op.MatMul(X, X_fit.T.astype(dtype), name=f"{name}_cross")
            two_cross = g.op.Mul(np.array(2.0, dtype=dtype), cross, name=f"{name}_2cross")
            sq_plus = g.op.Add(x_sq_sum, c_sq, name=f"{name}_sqplus")
            D = g.op.Sub(sq_plus, two_cross, name=f"{name}_D")
            D = g.op.Max(D, np.array(0.0, dtype=dtype), name=f"{name}_Dclamp")
        neg_gamma_D = g.op.Mul(np.array(-gamma, dtype=dtype), D, name=f"{name}_ngD")
        return g.op.Exp(neg_gamma_D, name=f"{name}_rbf_K")

    elif kernel in ("poly", "polynomial"):
        # K(X, Y) = (gamma * X @ Y.T + coef0)^degree
        cross = g.op.MatMul(X, X_fit.T.astype(dtype), name=f"{name}_cross")
        gamma_cross = g.op.Mul(np.array(gamma, dtype=dtype), cross, name=f"{name}_gC")
        biased = g.op.Add(gamma_cross, np.array(coef0, dtype=dtype), name=f"{name}_bias")
        return g.op.Pow(biased, np.array(float(degree), dtype=dtype), name=f"{name}_poly_K")

    elif kernel == "sigmoid":
        # K(X, Y) = tanh(gamma * X @ Y.T + coef0)
        cross = g.op.MatMul(X, X_fit.T.astype(dtype), name=f"{name}_cross")
        gamma_cross = g.op.Mul(np.array(gamma, dtype=dtype), cross, name=f"{name}_gC")
        biased = g.op.Add(gamma_cross, np.array(coef0, dtype=dtype), name=f"{name}_bias")
        return g.op.Tanh(biased, name=f"{name}_sigmoid_K")

    elif kernel == "cosine":
        # K(X, Y) = X @ Y.T / (||X||_2 * ||Y||_2)
        cross = g.op.MatMul(X, X_fit.T.astype(dtype), name=f"{name}_cross")
        # row norms of X at runtime: (N, 1)
        x_sq = g.op.Mul(X, X, name=f"{name}_xsq")
        x_sq_sum = g.op.ReduceSum(
            x_sq, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_xsqsum"
        )
        x_norm = g.op.Sqrt(
            g.op.Max(x_sq_sum, np.array(1e-12, dtype=dtype), name=f"{name}_xsqmax"),
            name=f"{name}_xnorm",
        )  # (N, 1)
        # row norms of X_fit at conversion time: (1, M)
        y_norms = np.sqrt(np.maximum(np.sum(X_fit**2, axis=1), 1e-12)).astype(dtype)[
            np.newaxis, :
        ]  # (1, M)
        norm_outer = g.op.Mul(x_norm, y_norms, name=f"{name}_norm_outer")  # (N, M)
        return g.op.Div(cross, norm_outer, name=f"{name}_cosine_K")

    elif kernel == "laplacian":
        # K(X, Y) = exp(-gamma * ||X-Y||_1)
        # Broadcast: (N, 1, F) - (1, M, F) → (N, M, F), then ReduceSum abs over F
        X_exp = g.op.Unsqueeze(X, np.array([1], dtype=np.int64), name=f"{name}_Xexp")  # (N,1,F)
        X_fit_3d = X_fit.astype(dtype)[np.newaxis, :, :]  # (1, M, F)
        diff = g.op.Sub(X_exp, X_fit_3d, name=f"{name}_diff")  # (N, M, F)
        abs_diff = g.op.Abs(diff, name=f"{name}_absdiff")  # (N, M, F)
        l1_dist = g.op.ReduceSum(
            abs_diff, np.array([2], dtype=np.int64), keepdims=0, name=f"{name}_l1dist"
        )  # (N, M)
        neg_gamma_D = g.op.Mul(np.array(-gamma, dtype=dtype), l1_dist, name=f"{name}_ngD")
        return g.op.Exp(neg_gamma_D, name=f"{name}_laplacian_K")

    elif kernel == "chi2":
        # K(x, y) = exp(-gamma * sum_i (x_i - y_i)^2 / (x_i + y_i))
        X_exp = g.op.Unsqueeze(X, np.array([1], dtype=np.int64), name=f"{name}_Xexp")  # (N,1,F)
        X_fit_3d = X_fit.astype(dtype)[np.newaxis, :, :]  # (1, M, F)
        diff = g.op.Sub(X_exp, X_fit_3d, name=f"{name}_diff")  # (N, M, F)
        diff_sq = g.op.Mul(diff, diff, name=f"{name}_diffsq")  # (N, M, F)
        denom = g.op.Add(X_exp, X_fit_3d, name=f"{name}_denom")  # (N, M, F)
        denom_safe = g.op.Max(denom, np.array(1e-30, dtype=dtype), name=f"{name}_denom_safe")
        ratio = g.op.Div(diff_sq, denom_safe, name=f"{name}_ratio")  # (N, M, F)
        chi2_sum = g.op.ReduceSum(
            ratio, np.array([2], dtype=np.int64), keepdims=0, name=f"{name}_chi2sum"
        )  # (N, M)
        neg_gamma_D = g.op.Mul(np.array(-gamma, dtype=dtype), chi2_sum, name=f"{name}_ngD")
        return g.op.Exp(neg_gamma_D, name=f"{name}_chi2_K")

    else:
        raise NotImplementedError(
            f"Kernel '{kernel}' is not supported for ONNX conversion. "
            "Supported kernels: 'linear', 'rbf', 'poly', 'polynomial', "
            "'sigmoid', 'cosine', 'laplacian', 'chi2'."
        )


@register_sklearn_converter(KernelRidge)
def sklearn_kernel_ridge(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: KernelRidge,
    X: str,
    name: str = "kernel_ridge",
) -> str:
    """
    Converts a :class:`sklearn.kernel_ridge.KernelRidge` into ONNX.

    The prediction formula is:

    .. code-block:: text

        K = kernel(X, X_fit_)          # (N, M)
        y_pred = K @ dual_coef_        # (N,) or (N, n_targets)

    where ``X_fit_`` (the training data) and ``dual_coef_`` (the dual solution)
    are stored as ONNX constants at conversion time.

    **Supported kernels**: ``'linear'``, ``'rbf'``, ``'poly'``/``'polynomial'``,
    ``'sigmoid'``, ``'cosine'``, ``'laplacian'``, ``'chi2'``.
    Callable kernels and ``'precomputed'`` are not supported.

    Graph structure (single-target)

    .. code-block:: text

        X  ──kernel(X, X_fit_)──►  K (N, M)
                                       │
                                   MatMul(dual_coef_)  ──►  y_pred (N,)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a fitted ``KernelRidge``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name (predictions)
    """
    assert isinstance(estimator, KernelRidge), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    kernel = estimator.kernel
    if callable(kernel):
        raise NotImplementedError(
            "Callable kernels are not supported for ONNX conversion of KernelRidge. "
            "Use a string kernel name instead."
        )
    if kernel == "precomputed":
        raise NotImplementedError(
            "The 'precomputed' kernel is not supported for ONNX conversion of KernelRidge."
        )

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # Resolve effective gamma (sklearn default: 1 / n_features when gamma=None)
    gamma = (
        float(estimator.gamma) if estimator.gamma is not None else 1.0 / estimator.X_fit_.shape[1]
    )
    degree = int(estimator.degree)
    coef0 = float(estimator.coef0)

    X_fit = estimator.X_fit_.astype(dtype)  # (M, F)
    dual_coef = estimator.dual_coef_.astype(dtype)  # (M,) or (M, n_targets)

    # K = K(X, X_fit) → (N, M)
    K = _emit_kernel_ridge_matrix(g, kernel, gamma, degree, coef0, X, X_fit, name, dtype)

    # y_pred = K @ dual_coef
    if dual_coef.ndim == 1:
        # Single-target: (N, M) @ (M, 1) → (N, 1) → reshape to (N,)
        y_raw = g.op.MatMul(K, dual_coef.reshape(-1, 1), name=f"{name}_mm")  # (N, 1)
        result = g.op.Reshape(
            y_raw, np.array([-1], dtype=np.int64), name=name, outputs=outputs[:1]
        )  # (N,)
    else:
        # Multi-target: (N, M) @ (M, n_targets) → (N, n_targets)
        result = g.op.MatMul(K, dual_coef, name=name, outputs=outputs[:1])

    g.set_type(result, itype)
    return result
