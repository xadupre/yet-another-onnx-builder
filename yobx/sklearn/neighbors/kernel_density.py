from typing import Dict, List

import numpy as np
from scipy.special import gamma
from sklearn.neighbors import KernelDensity

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


def _log_norm_constant(kernel: str, h: float, D: int, N: int) -> float:
    """
    Return the log of the normalization constant for a kernel density estimator.

    The density estimate is:

    .. code-block:: text

        density(x) = sum_i k(d(x, x_i) / h) / norm_constant

    where ``k`` is the unnormalized kernel function and ``norm_constant``
    depends on the kernel type, bandwidth *h*, number of features *D*, and
    number of training samples *N*.

    Normalization constants (consistent with :class:`sklearn.neighbors.KernelDensity`):

    * ``'gaussian'``: ``N · (2π)^(D/2) · h^D``
    * ``'tophat'``: ``N · vol_D · h^D``
    * ``'epanechnikov'``: ``N · vol_D · h^D · 2 / (D + 2)``
    * ``'exponential'``: ``N · Γ(D + 1) · vol_D · h^D``
    * ``'linear'``: ``N · vol_D · h^D / (D + 1)``
    * ``'cosine'``: ``N · D · vol_D · (π/4) · h^D · ∫₀¹ cos(πr/2) r^(D-1) dr``

    where ``vol_D = π^(D/2) / Γ(D/2 + 1)`` is the volume of the unit ball in
    ``D`` dimensions.

    :param kernel: kernel name
    :param h: bandwidth (positive scalar)
    :param D: number of features
    :param N: number of training samples
    :return: ``log(norm_constant)``
    :raises NotImplementedError: for unsupported kernel names
    """
    vol_D = np.pi ** (D / 2) / gamma(D / 2 + 1)

    if kernel == "gaussian":
        C = N * (2 * np.pi) ** (D / 2) * h**D
    elif kernel == "tophat":
        C = N * vol_D * h**D
    elif kernel == "epanechnikov":
        C = N * vol_D * h**D * 2.0 / (D + 2)
    elif kernel == "exponential":
        C = N * gamma(D + 1) * vol_D * h**D
    elif kernel == "linear":
        C = N * vol_D * h**D / (D + 1)
    elif kernel == "cosine":
        # Surface area of D-dim unit sphere times the radial integral
        # C = N · D · vol_D · (π/4) · h^D · ∫₀¹ cos(π·r/2) · r^(D-1) dr
        from scipy.integrate import quad

        radial_int, _ = quad(lambda r: np.cos(np.pi / 2 * r) * r ** (D - 1), 0.0, 1.0)
        S_D = D * vol_D  # surface area of D-dim unit sphere
        C = N * S_D * (np.pi / 4) * radial_int * h**D
    else:
        raise NotImplementedError(
            f"Kernel {kernel!r} is not supported for ONNX conversion of KernelDensity. "
            "Supported kernels: 'gaussian', 'tophat', 'epanechnikov', "
            "'exponential', 'linear', 'cosine'."
        )

    return float(np.log(C))


def _emit_sq_euclidean_distances(
    g: GraphBuilderExtendedProtocol,
    X: str,
    X_train: np.ndarray,
    dtype,
    name: str,
) -> str:
    """
    Emit ONNX ops for the squared Euclidean distance matrix ``(N_test, N_train)``.

    Uses ``com.microsoft.CDist`` when the opset is available, otherwise falls
    back to the expand-and-dot identity:

    .. code-block:: text

        ||x - c||² = ||x||² - 2·x·cᵀ + ||c||²

    :param g: graph builder
    :param X: input tensor name, shape ``(N, F)``
    :param X_train: training data, shape ``(M, F)``
    :param dtype: numpy floating-point dtype
    :param name: node-name prefix
    :return: ONNX tensor name of shape ``(N, M)``
    """
    if g.has_opset("com.microsoft"):
        X_train_name = g.make_initializer(f"{name}_Xtrain", X_train.astype(dtype))
        sq_dists = g.make_node(
            "CDist",
            [X, X_train_name],
            domain="com.microsoft",
            metric="sqeuclidean",
            name=f"{name}_cdist",
        )
        return g.op.Max(sq_dists, np.array(0.0, dtype=dtype), name=f"{name}_sq_clip")

    # Expand-trick: ||x-c||² = ||x||² + ||c||² - 2·x·cᵀ
    c_sq = np.sum(X_train**2, axis=1)[np.newaxis, :].astype(dtype)  # (1, M)
    x_sq = g.op.Mul(X, X, name=f"{name}_xsq")
    x_sq_sum = g.op.ReduceSum(
        x_sq,
        np.array([1], dtype=np.int64),
        keepdims=1,
        name=f"{name}_xsqsum",
    )  # (N, 1)
    cross = g.op.MatMul(X, X_train.T.astype(dtype), name=f"{name}_cross")  # (N, M)
    two_cross = g.op.Mul(np.array(2.0, dtype=dtype), cross, name=f"{name}_2cross")
    sq_plus = g.op.Add(x_sq_sum, c_sq, name=f"{name}_sqplus")
    sq_dists = g.op.Sub(sq_plus, two_cross, name=f"{name}_sq")
    return g.op.Max(sq_dists, np.array(0.0, dtype=dtype), name=f"{name}_sq_clip")


@register_sklearn_converter(KernelDensity)
def sklearn_kernel_density(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: KernelDensity,
    X: str,
    name: str = "kde",
) -> str:
    """
    Converts a :class:`sklearn.neighbors.KernelDensity` into ONNX.

    The converter implements :meth:`~sklearn.neighbors.KernelDensity.score_samples`,
    which returns the **log-density** at each query point:

    .. code-block:: text

        log_density(x) = log( (1/N) · Σᵢ k( ‖x − xᵢ‖ / h ) ) − log(norm_h)

    where ``k`` is the unnormalized kernel, ``h`` the bandwidth, and
    ``norm_h`` the kernel-specific normalization factor that does not depend on
    the query *x*.

    Rearranging, the output equals:

    .. code-block:: text

        output(x) = log(Σᵢ k( ‖x − xᵢ‖ / h )) − log(N · norm_h · h^D)

    All normalization constants are **precomputed at conversion time** and
    stored as ONNX scalar initializers, so the resulting graph contains only
    arithmetic and reduction operations.

    **Supported kernels**

    +------------------+------------------------------------------------------+
    | ``'gaussian'``   | ``k(t) = exp(−t²/2)``                                |
    +------------------+------------------------------------------------------+
    | ``'exponential'``| ``k(t) = exp(−t)``                                   |
    +------------------+------------------------------------------------------+
    | ``'tophat'``     | ``k(t) = 1``  for ``t ≤ 1``, else ``0``             |
    +------------------+------------------------------------------------------+
    | ``'epanechnikov'``| ``k(t) = 1 − t²`` for ``t ≤ 1``, else ``0``       |
    +------------------+------------------------------------------------------+
    | ``'linear'``     | ``k(t) = 1 − t``  for ``t ≤ 1``, else ``0``        |
    +------------------+------------------------------------------------------+
    | ``'cosine'``     | ``k(t) = (π/4)·cos(πt/2)`` for ``t ≤ 1``, else 0  |
    +------------------+------------------------------------------------------+

    where ``t = ‖x − xᵢ‖ / h``.

    **Graph structure (gaussian kernel, standard-ONNX path)**

    .. code-block:: text

        X (N, F)       X_train (M, F)
          │                 │
          └──sq_euclidean───┘  →  sq_dists (N, M)
                                       │
                               Mul(−0.5/h²)
                                       │
                               ReduceLogSumExp(axis=1)  →  log_sum (N,)
                                       │
                               Sub(log_norm_const)  →  log_density (N,)

    For **compact kernels** (``tophat``, ``epanechnikov``, ``linear``,
    ``cosine``) the same squared-distance matrix is used but kernel values
    are summed directly, then the log is taken.  When no training sample
    falls within the bandwidth the score is ``−∞`` (matching sklearn
    behaviour for degenerate cases).

    **Computation paths**

    *With* ``com.microsoft`` *opset* (CDist path):
    Squared distances are delegated to ``com.microsoft.CDist``
    (``metric="sqeuclidean"``), which is hardware-accelerated by ONNX Runtime.

    *Without* ``com.microsoft`` *opset* (standard ONNX path):
    Squared distances are computed using the expansion identity
    ``||x-c||² = ||x||² − 2·x·cᵀ + ||c||²``, which requires only
    ``MatMul`` and element-wise ops available since opset 13.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names; ``outputs[0]`` receives the
        log-density vector of shape ``(N,)``
    :param estimator: a fitted :class:`~sklearn.neighbors.KernelDensity`
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name for the log-density (shape ``(N,)``)
    """
    assert isinstance(
        estimator, KernelDensity
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    kernel = estimator.kernel
    if callable(kernel):
        raise NotImplementedError(
            "Callable kernels are not supported for ONNX conversion of KernelDensity."
        )

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # bandwidth_ was added in sklearn 1.2; fall back to bandwidth for older versions
    h = float(getattr(estimator, "bandwidth_", estimator.bandwidth))
    X_train = np.asarray(estimator.tree_.data).astype(dtype)  # (M, F)
    N, D = X_train.shape

    # Precompute the log-normalization constant (scalar, computed at conversion time)
    log_norm = _log_norm_constant(kernel, h, D, N)
    log_norm_arr = np.array(log_norm, dtype=dtype)

    # ── Step 1: Squared Euclidean distances (N_test, M) ──────────────────────
    sq_dists = _emit_sq_euclidean_distances(g, X, X_train, dtype, f"{name}_sqdist")

    # ── Step 2: Compute log(Σᵢ kernel_i) ─────────────────────────────────────
    if kernel == "gaussian":
        # log k(t) = −t²/2 with t = d/h → log k = −0.5 · sq_dist / h²
        neg_half_h2 = np.array(-0.5 / h**2, dtype=dtype)
        log_kernel = g.op.Mul(sq_dists, neg_half_h2, name=f"{name}_lk")
        log_sum = g.op.ReduceLogSumExp(
            log_kernel,
            np.array([1], dtype=np.int64),
            keepdims=0,
            name=f"{name}_logsumexp",
        )

    elif kernel == "exponential":
        # k(t) = exp(−t) with t = d/h → log k = −d/h = −sqrt(sq_dist)/h
        dists = g.op.Sqrt(sq_dists, name=f"{name}_dists")
        neg_inv_h = np.array(-1.0 / h, dtype=dtype)
        log_kernel = g.op.Mul(dists, neg_inv_h, name=f"{name}_lk")
        log_sum = g.op.ReduceLogSumExp(
            log_kernel,
            np.array([1], dtype=np.int64),
            keepdims=0,
            name=f"{name}_logsumexp",
        )

    else:
        # Compact-support kernels: compute kernel values directly and sum.
        dists = g.op.Sqrt(sq_dists, name=f"{name}_dists")
        inv_h = np.array(1.0 / h, dtype=dtype)
        normed = g.op.Mul(dists, inv_h, name=f"{name}_normed")  # t = d/h (N, M)

        # Indicator for t ≤ 1 (i.e. d ≤ h), cast to float
        one = np.array(1.0, dtype=dtype)
        in_support = g.op.Cast(
            g.op.LessOrEqual(normed, one, name=f"{name}_mask"),
            to=itype,
            name=f"{name}_in_support",
        )  # (N, M) float

        if kernel == "tophat":
            kernel_vals = in_support

        elif kernel == "epanechnikov":
            # k(t) = 1 − t² (for t ≤ 1, else 0)
            t_sq = g.op.Mul(normed, normed, name=f"{name}_tsq")
            k_raw = g.op.Sub(one, t_sq, name=f"{name}_k_raw")  # 1 − t² (unclamped)
            kernel_vals = g.op.Mul(k_raw, in_support, name=f"{name}_kv")

        elif kernel == "linear":
            # k(t) = 1 − t (for t ≤ 1, else 0)
            k_raw = g.op.Sub(one, normed, name=f"{name}_k_raw")  # 1 − t (unclamped)
            kernel_vals = g.op.Mul(k_raw, in_support, name=f"{name}_kv")

        elif kernel == "cosine":
            # k(t) = (π/4) · cos(π·t/2) (for t ≤ 1, else 0)
            half_pi = np.array(np.pi / 2, dtype=dtype)
            pi_over_4 = np.array(np.pi / 4, dtype=dtype)
            arg = g.op.Mul(normed, half_pi, name=f"{name}_arg")  # π·t/2
            cos_val = g.op.Cos(arg, name=f"{name}_cos")
            k_raw = g.op.Mul(pi_over_4, cos_val, name=f"{name}_k_raw")
            kernel_vals = g.op.Mul(k_raw, in_support, name=f"{name}_kv")

        else:
            raise NotImplementedError(
                f"Kernel {kernel!r} is not supported for ONNX conversion of "
                "KernelDensity. Supported kernels: 'gaussian', 'tophat', "
                "'epanechnikov', 'exponential', 'linear', 'cosine'."
            )

        # Sum kernel values over training points (axis=1) → (N,)
        kernel_sum = g.op.ReduceSum(
            kernel_vals,
            np.array([1], dtype=np.int64),
            keepdims=0,
            name=f"{name}_ksum",
        )  # (N,)

        # log(kernel_sum): produces −∞ when sum=0 (empty support), matching sklearn
        log_sum = g.op.Log(kernel_sum, name=f"{name}_log_ksum")

    # ── Step 3: Subtract log normalization constant ───────────────────────────
    result = g.op.Sub(
        log_sum,
        log_norm_arr,
        name=name,
        outputs=outputs[:1],
    )

    assert isinstance(result, str)
    if not sts:
        g.set_type(result, itype)
    return result
