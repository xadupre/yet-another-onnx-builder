from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from scipy.linalg import solve_triangular
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    Matern,
    Product,
    RBF,
    Sum,
    WhiteKernel,
)

from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter

# ── constants from sklearn._gpc used in the probability approximation ────────
# Williams & Barber (1998) 5-point approximation of ∫ σ(z) N(z|f, v) dz.
from sklearn.gaussian_process._gpc import COEFS as _GPC_COEFS  # (5, 1)
from sklearn.gaussian_process._gpc import LAMBDAS as _GPC_LAMBDAS  # (5, 1)

_LAMBDAS_1D = _GPC_LAMBDAS.ravel()  # shape (5,)
_COEFS_1D = _GPC_COEFS.ravel()  # shape (5,)
_COEFS_HALF_SUM = float(0.5 * _COEFS_1D.sum())


# ── kernel helpers ────────────────────────────────────────────────────────────


def _emit_pairwise_sq_dist(
    g: GraphBuilderExtendedProtocol,
    X: str,
    X_ref: np.ndarray,
    length_scale: np.ndarray,
    name: str,
    dtype,
) -> str:
    """
    Compute pairwise squared Euclidean distances (after length-scale
    normalisation) between rows of *X* (shape ``(N, F)``, graph node) and
    rows of *X_ref* (shape ``(M, F)``, fixed constant).

    The formula used is::

        D[i,j] = ||X[i]/l - X_ref[j]/l||²

    When the ``com.microsoft`` domain is registered in the graph builder the
    computation is delegated to ``com.microsoft.CDist`` with
    ``metric="sqeuclidean"``, which is hardware-accelerated by ONNX Runtime.
    The length-scale normalisation is absorbed into the inputs:
    ``X`` is divided at runtime and ``X_ref`` is pre-divided at conversion time.

    For the standard-ONNX path the numerically stable expand-expand trick is
    used::

        D[i,j] = ||X[i]/l||² + ||X_ref[j]/l||² - 2 · X[i]/l · X_ref[j]/l

    :param g: graph builder
    :param X: input tensor name, shape ``(N, F)``
    :param X_ref: fixed reference matrix, shape ``(M, F)``
    :param length_scale: per-feature or scalar length-scale array
    :param name: node name prefix
    :param dtype: numpy dtype
    :return: ONNX tensor name of shape ``(N, M)``
    """
    # Pre-normalise training data at conversion time (constant)
    X_ref_norm = (X_ref / length_scale).astype(dtype)  # (M, F)

    # Normalise query data at run time
    ls = np.asarray(length_scale, dtype=dtype).reshape(1, -1)  # (1, F)
    X_norm = g.op.Div(X, ls, name=f"{name}_xnorm")  # (N, F)

    # ── CDist path (com.microsoft) ─────────────────────────────────────────
    if g.has_opset("com.microsoft"):
        X_ref_norm_name = g.make_initializer(f"{name}_Xref_norm", X_ref_norm)
        D = g.make_node(
            "CDist",
            [X_norm, X_ref_norm_name],
            domain="com.microsoft",
            metric="sqeuclidean",
            name=f"{name}_cdist",
        )
        return g.op.Max(D, np.array(0.0, dtype=dtype), name=f"{name}_Dclamp")

    # ── Standard ONNX path ─────────────────────────────────────────────────
    c_sq = np.sum(X_ref_norm**2, axis=1)[np.newaxis, :].astype(dtype)  # (1, M)

    # ||X_norm[i]||² → (N, 1)
    x_sq = g.op.Mul(X_norm, X_norm, name=f"{name}_xsq")
    x_sq_sum = g.op.ReduceSum(
        x_sq, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_xsqsum"
    )  # (N, 1)

    # 2 · X_norm @ X_ref_norm.T → (N, M)
    cross = g.op.MatMul(X_norm, X_ref_norm.T.astype(dtype), name=f"{name}_cross")
    two_cross = g.op.Mul(np.array(2.0, dtype=dtype), cross, name=f"{name}_2cross")

    # D = x_sq_sum + c_sq - 2*cross
    sq_plus = g.op.Add(x_sq_sum, c_sq, name=f"{name}_sqplus")
    D = g.op.Sub(sq_plus, two_cross, name=f"{name}_D")

    # Clip to non-negative to guard against tiny floating-point negatives
    return g.op.Max(D, np.array(0.0, dtype=dtype), name=f"{name}_Dclamp")


def _emit_kernel_matrix(
    g: GraphBuilderExtendedProtocol, kernel, X: str, X_ref: np.ndarray, name: str, dtype
) -> str:
    """
    Emit ONNX operations computing the kernel matrix ``K(X, X_ref)``.

    *X* is a graph node of shape ``(N, F)``; *X_ref* is a fixed numpy array
    of shape ``(M, F)`` stored as graph constants.  Returns a tensor name
    of shape ``(N, M)``.

    Supported kernels
    -----------------
    * :class:`~sklearn.gaussian_process.kernels.RBF` (scalar or ARD length-scale)
    * :class:`~sklearn.gaussian_process.kernels.ConstantKernel`
    * :class:`~sklearn.gaussian_process.kernels.WhiteKernel`
      (returns zeros for the off-diagonal test-vs-train block)
    * :class:`~sklearn.gaussian_process.kernels.Matern`
      (``nu`` ∈ {0.5, 1.5, 2.5, ∞})
    * :class:`~sklearn.gaussian_process.kernels.DotProduct`
    * :class:`~sklearn.gaussian_process.kernels.Product` (recursive)
    * :class:`~sklearn.gaussian_process.kernels.Sum` (recursive)

    :param g: graph builder
    :param kernel: a fitted sklearn kernel object
    :param X: input tensor name, shape ``(N, F)``
    :param X_ref: fixed reference matrix, shape ``(M, F)``
    :param name: node name prefix
    :param dtype: numpy dtype
    :return: ONNX tensor name of shape ``(N, M)``
    :raises NotImplementedError: for unsupported kernel types or Matern ``nu`` values
    """
    X_ref = np.asarray(X_ref, dtype=dtype)
    M = X_ref.shape[0]

    # NOTE: Matern inherits from RBF in sklearn, so the Matern check MUST come first.
    if isinstance(kernel, Matern):
        ls = np.atleast_1d(np.asarray(kernel.length_scale, dtype=dtype))
        nu = kernel.nu
        D = _emit_pairwise_sq_dist(g, X, X_ref, ls, f"{name}_mat_D", dtype)

        if nu == np.inf:
            return g.op.Exp(
                g.op.Mul(np.array(-0.5, dtype=dtype), D, name=f"{name}_mat_inf_scale"),
                name=f"{name}_mat_inf_K",
            )
        elif nu == 0.5:
            # k = exp(-r),  r = sqrt(D)
            r = g.op.Sqrt(D, name=f"{name}_mat05_r")
            return g.op.Exp(
                g.op.Mul(np.array(-1.0, dtype=dtype), r, name=f"{name}_mat05_nr"),
                name=f"{name}_mat05_K",
            )
        elif nu == 1.5:
            # k = (1 + √3·r) · exp(-√3·r)
            sqrt3 = np.array(np.sqrt(3.0), dtype=dtype)
            r = g.op.Mul(sqrt3, g.op.Sqrt(D, name=f"{name}_mat15_sqD"), name=f"{name}_mat15_r")
            one_plus_r = g.op.Add(np.array(1.0, dtype=dtype), r, name=f"{name}_mat15_1pr")
            neg_r = g.op.Mul(np.array(-1.0, dtype=dtype), r, name=f"{name}_mat15_nr")
            return g.op.Mul(
                one_plus_r, g.op.Exp(neg_r, name=f"{name}_mat15_exp"), name=f"{name}_mat15_K"
            )
        elif nu == 2.5:
            # k = (1 + √5·r + 5·r²/3) · exp(-√5·r)
            sqrt5 = np.array(np.sqrt(5.0), dtype=dtype)
            r = g.op.Mul(sqrt5, g.op.Sqrt(D, name=f"{name}_mat25_sqD"), name=f"{name}_mat25_r")
            r2_3 = g.op.Div(
                g.op.Mul(r, r, name=f"{name}_mat25_r2"),
                np.array(3.0, dtype=dtype),
                name=f"{name}_mat25_r2_3",
            )
            poly = g.op.Add(
                g.op.Add(np.array(1.0, dtype=dtype), r, name=f"{name}_mat25_1pr"),
                r2_3,
                name=f"{name}_mat25_poly",
            )
            neg_r = g.op.Mul(np.array(-1.0, dtype=dtype), r, name=f"{name}_mat25_nr")
            return g.op.Mul(
                poly, g.op.Exp(neg_r, name=f"{name}_mat25_exp"), name=f"{name}_mat25_K"
            )
        else:
            raise NotImplementedError(
                f"Matern kernel with nu={nu} is not supported. "
                "Supported values: 0.5, 1.5, 2.5, inf."
            )

    elif isinstance(kernel, RBF):
        ls = np.atleast_1d(np.asarray(kernel.length_scale, dtype=dtype))
        D = _emit_pairwise_sq_dist(g, X, X_ref, ls, f"{name}_rbf", dtype)
        return g.op.Exp(
            g.op.Mul(np.array(-0.5, dtype=dtype), D, name=f"{name}_rbf_scale"),
            name=f"{name}_rbf_K",
        )

    elif isinstance(kernel, ConstantKernel):
        c = np.array(kernel.constant_value, dtype=dtype)
        # Build (N, M) filled with c:
        #   MatMul(X, zero_col) → (N, 1) of zeros, then add c * ones(1, M)
        zero_col = np.zeros((X_ref.shape[1], 1), dtype=dtype)
        x_zero = g.op.MatMul(X, zero_col, name=f"{name}_ck_zero")  # (N, 1)
        c_row = np.full((1, M), float(c), dtype=dtype)
        return g.op.Add(x_zero, c_row, name=f"{name}_ck_K")

    elif isinstance(kernel, WhiteKernel):
        # k(x_test, x_train) = 0 for all test/train pairs
        zero_col = np.zeros((X_ref.shape[1], 1), dtype=dtype)
        x_zero = g.op.MatMul(X, zero_col, name=f"{name}_wk_zero")  # (N, 1)
        zero_row = np.zeros((1, M), dtype=dtype)
        return g.op.Add(x_zero, zero_row, name=f"{name}_wk_K")

    elif isinstance(kernel, DotProduct):
        sigma_sq = np.array(kernel.sigma_0**2, dtype=dtype).reshape(1, 1)  # (1, 1)
        cross = g.op.MatMul(X, X_ref.T, name=f"{name}_dp_cross")  # (N, M)
        return g.op.Add(cross, sigma_sq, name=f"{name}_dp_K")

    elif isinstance(kernel, Product):
        K1 = _emit_kernel_matrix(g, kernel.k1, X, X_ref, f"{name}_pk1", dtype)
        K2 = _emit_kernel_matrix(g, kernel.k2, X, X_ref, f"{name}_pk2", dtype)
        return g.op.Mul(K1, K2, name=f"{name}_prod_K")

    elif isinstance(kernel, Sum):
        K1 = _emit_kernel_matrix(g, kernel.k1, X, X_ref, f"{name}_sk1", dtype)
        K2 = _emit_kernel_matrix(g, kernel.k2, X, X_ref, f"{name}_sk2", dtype)
        return g.op.Add(K1, K2, name=f"{name}_sum_K")

    else:
        raise NotImplementedError(
            f"Kernel {type(kernel).__name__} is not supported in ONNX conversion. "
            "Supported kernels: RBF, ConstantKernel, WhiteKernel, "
            "Matern (nu ∈ {0.5, 1.5, 2.5, ∞}), "
            "DotProduct, Product, Sum."
        )


def _kernel_diag_const(kernel) -> Optional[float]:
    """
    Return ``k(x, x)`` as a constant float if the kernel is stationary
    (i.e. its self-covariance does not depend on *x*), otherwise return
    ``None``.

    :param kernel: a fitted sklearn kernel object
    :return: the constant diagonal value, or ``None``
    """
    # NOTE: Matern inherits from RBF in sklearn, so the Matern check MUST come first.
    if isinstance(kernel, Matern):
        return 1.0
    if isinstance(kernel, RBF):
        return 1.0
    if isinstance(kernel, ConstantKernel):
        return float(kernel.constant_value)
    if isinstance(kernel, WhiteKernel):
        return float(kernel.noise_level)
    if isinstance(kernel, Product):
        d1 = _kernel_diag_const(kernel.k1)
        d2 = _kernel_diag_const(kernel.k2)
        return d1 * d2 if (d1 is not None and d2 is not None) else None
    if isinstance(kernel, Sum):
        d1 = _kernel_diag_const(kernel.k1)
        d2 = _kernel_diag_const(kernel.k2)
        return d1 + d2 if (d1 is not None and d2 is not None) else None
    return None  # DotProduct or unknown kernel


def _emit_kernel_diag(g: GraphBuilderExtendedProtocol, kernel, X: str, name: str, dtype) -> str:
    """
    Emit ONNX operations computing the kernel diagonal ``k(X[i], X[i])``
    for each row *i* of *X*.  Returns a tensor of shape ``(N,)``.

    For stationary kernels (RBF, Matern, etc.) the diagonal is a constant
    scalar; for :class:`~sklearn.gaussian_process.kernels.DotProduct` it
    depends on the input norm.

    :param g: graph builder
    :param kernel: a fitted sklearn kernel object
    :param X: input tensor name, shape ``(N, F)``
    :param name: node name prefix
    :param dtype: numpy dtype
    :return: ONNX tensor name of shape ``(N,)``
    """
    c = _kernel_diag_const(kernel)
    if c is not None:
        # Stationary: diagonal is c everywhere.
        # Trick: (X * 0) row-summed → (N,) of zeros, then add c.
        x_zeros = g.op.Mul(X, np.array(0.0, dtype=dtype), name=f"{name}_diag_zero")
        row_zero = g.op.ReduceSum(
            x_zeros, np.array([1], dtype=np.int64), keepdims=0, name=f"{name}_diag_rsum"
        )  # (N,)
        return g.op.Add(row_zero, np.array(c, dtype=dtype), name=f"{name}_diag_c")

    if isinstance(kernel, DotProduct):
        # k(x, x) = sigma_0² + ||x||²
        X_sq = g.op.Mul(X, X, name=f"{name}_dp_diag_xsq")
        X_sq_sum = g.op.ReduceSum(
            X_sq, np.array([1], dtype=np.int64), keepdims=0, name=f"{name}_dp_diag_sum"
        )  # (N,)
        sigma_sq = np.array(kernel.sigma_0**2, dtype=dtype)
        return g.op.Add(X_sq_sum, sigma_sq, name=f"{name}_dp_diag")

    if isinstance(kernel, Product):
        d1 = _emit_kernel_diag(g, kernel.k1, X, f"{name}_pk1", dtype)
        d2 = _emit_kernel_diag(g, kernel.k2, X, f"{name}_pk2", dtype)
        return g.op.Mul(d1, d2, name=f"{name}_prod_diag")

    if isinstance(kernel, Sum):
        d1 = _emit_kernel_diag(g, kernel.k1, X, f"{name}_sk1", dtype)
        d2 = _emit_kernel_diag(g, kernel.k2, X, f"{name}_sk2", dtype)
        return g.op.Add(d1, d2, name=f"{name}_sum_diag")

    raise NotImplementedError(f"Kernel diagonal for {type(kernel).__name__} is not supported.")


# ── binary GPC helper ─────────────────────────────────────────────────────────


def _emit_gpc_binary_pi_star(
    g: GraphBuilderExtendedProtocol, kernel, be, name: str, X: str, dtype
) -> str:
    """
    Emit ONNX operations for the binary GPC sigmoid probability ``π*``.

    Implements Algorithm 3.2 of *Gaussian Processes for Machine Learning*
    (Rasmussen & Williams 2006) followed by the Williams-Barber (1998)
    five-point Gauss-Hermite approximation of the probit integral.

    Steps
    -----

    .. code-block:: text

        K_trans = kernel(X, X_train)              # (N, M) — kernel matrix
        latent_mean = K_trans @ (y_train - π)      # (N,)   — posterior mean
        v = M_pre @ K_trans.T                     # (M, N) — M_pre precomputed
        latent_var  = diag(K(X,X)) - sum(v², 0)  # (N,)   — posterior variance
        π* = Σ_j COEFS[j] · integral_j + 0.5·ΣCOEFS   # (N,)

    *M_pre = L⁻¹ · diag(W_sr)* is precomputed once at conversion time using
    ``scipy.linalg.solve_triangular``.

    :param g: graph builder
    :param kernel: fitted binary-estimator kernel
    :param be: fitted ``_BinaryGaussianProcessClassifierLaplace`` instance
    :param name: node name prefix
    :param X: input tensor name, shape ``(N, F)``
    :param dtype: numpy dtype
    :return: ONNX tensor name of shape ``(N,)`` containing ``π*``
    """
    X_train = be.X_train_.astype(dtype)  # (M, F)
    coef = (be.y_train_ - be.pi_).astype(dtype)  # (M,)

    # Pre-compute M_pre = L⁻¹ · diag(W_sr) at conversion time.
    # We solve L @ X = I for X = L⁻¹ using back-substitution; this is a one-time
    # cost at conversion, not at inference, so the full inverse is acceptable here.
    L_inv = solve_triangular(be.L_, np.eye(len(be.L_)), lower=True).astype(dtype)
    M_pre = (L_inv * be.W_sr_[np.newaxis, :]).astype(dtype)  # (M, M)

    # K_trans = kernel(X, X_train) → (N, M)
    K_trans = _emit_kernel_matrix(g, kernel, X, X_train, f"{name}_Kt", dtype)

    # latent_mean = K_trans @ coef → (N,)
    # Reshape coef to (M, 1) for MatMul, then squeeze
    lm_mm = g.op.MatMul(K_trans, coef.reshape(-1, 1), name=f"{name}_lm_mm")  # (N, 1)
    latent_mean = g.op.Reshape(lm_mm, np.array([-1], dtype=np.int64), name=f"{name}_lm")  # (N,)

    # v = M_pre @ K_trans.T → (M, N)
    K_trans_T = g.op.Transpose(K_trans, perm=[1, 0], name=f"{name}_KtT")
    v = g.op.MatMul(M_pre, K_trans_T, name=f"{name}_v")  # (M, N)

    # latent_var = kernel_diag(X) - sum(v², axis=0) → (N,)
    kdiag = _emit_kernel_diag(g, kernel, X, f"{name}_kdiag", dtype)  # (N,)
    v_sq = g.op.Mul(v, v, name=f"{name}_vsq")
    v_sq_sum = g.op.ReduceSum(
        v_sq, np.array([0], dtype=np.int64), keepdims=0, name=f"{name}_vss"
    )  # (N,)
    latent_var = g.op.Sub(kdiag, v_sq_sum, name=f"{name}_lv_raw")
    # Clamp to a small positive value to keep latent_var > 0 in the presence
    # of numerical round-off (the sqrt and 1/(2*var) below require positivity).
    latent_var = g.op.Max(latent_var, np.array(1e-10, dtype=dtype), name=f"{name}_lv")  # (N,)

    # ── Williams-Barber approximation ─────────────────────────────────────
    # alpha = 1 / (2 · latent_var)   → (N,)
    alpha = g.op.Div(
        np.array(1.0, dtype=dtype),
        g.op.Mul(np.array(2.0, dtype=dtype), latent_var, name=f"{name}_2lv"),
        name=f"{name}_alpha",
    )

    # γ = ΛΛΛΛΛ · latent_mean  → (5, N):  broadcast (5,1) × (1,N)
    lm_2d = g.op.Unsqueeze(
        latent_mean, np.array([0], dtype=np.int64), name=f"{name}_lm2d"
    )  # (1,N)
    lambdas_col = _LAMBDAS_1D.reshape(-1, 1).astype(dtype)  # (5, 1)
    gamma = g.op.Mul(lambdas_col, lm_2d, name=f"{name}_gamma")  # (5, N)

    # α and λ² in (1,N) and (5,1) shapes for broadcasting
    alpha_2d = g.op.Unsqueeze(alpha, np.array([0], dtype=np.int64), name=f"{name}_a2d")  # (1,N)
    lambdas_sq_col = (_LAMBDAS_1D**2).reshape(-1, 1).astype(dtype)  # (5, 1)

    # ratio = α / (α + λ²)  → (5, N)
    denom = g.op.Add(alpha_2d, lambdas_sq_col, name=f"{name}_denom")  # (5, N)
    ratio = g.op.Div(alpha_2d, denom, name=f"{name}_ratio")  # (5, N)

    # erf(γ · √ratio)  → (5, N)
    erf_arg = g.op.Mul(gamma, g.op.Sqrt(ratio, name=f"{name}_sqratio"), name=f"{name}_erf_arg")
    erf_val = g.op.Erf(erf_arg, name=f"{name}_erf")  # (5, N)

    # √(π / α)   → (1, N)
    sqrt_pi_over_alpha = g.op.Sqrt(
        g.op.Div(np.array(np.pi, dtype=dtype), alpha_2d, name=f"{name}_pia"), name=f"{name}_sqpia"
    )

    # 2 · √(latent_var · 2π)  → (1, N)
    lv_2d = g.op.Unsqueeze(
        latent_var, np.array([0], dtype=np.int64), name=f"{name}_lv2d"
    )  # (1,N)
    two_sqrt_lv2pi = g.op.Mul(
        np.array(2.0, dtype=dtype),
        g.op.Sqrt(
            g.op.Mul(lv_2d, np.array(2.0 * np.pi, dtype=dtype), name=f"{name}_lv2pi"),
            name=f"{name}_sqrtlv2pi",
        ),
        name=f"{name}_2sqrtlv2pi",
    )  # (1, N)

    # integrals = √(π/α) · erf_val / (2·√(latent_var·2π))   → (5, N)
    integrals = g.op.Div(
        g.op.Mul(sqrt_pi_over_alpha, erf_val, name=f"{name}_num"),
        two_sqrt_lv2pi,
        name=f"{name}_integrals",
    )

    # π* = Σ_j COEFS[j] · integrals[j] + 0.5 · ΣCOEFS   → (N,)
    coefs_col = _COEFS_1D.reshape(-1, 1).astype(dtype)  # (5, 1)
    weighted = g.op.Mul(coefs_col, integrals, name=f"{name}_weighted")  # (5, N)
    pi_star = g.op.Add(
        g.op.ReduceSum(weighted, np.array([0], dtype=np.int64), keepdims=0, name=f"{name}_pisum"),
        np.array(_COEFS_HALF_SUM, dtype=dtype),
        name=f"{name}_pi_star",
    )  # (N,)
    return pi_star


def _emit_label_and_proba_from_scores(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    scores: str,
    classes: np.ndarray,
    name: str,
    outputs: List[str],
) -> Tuple[str, str]:
    """
    Shared helper: ArgMax(scores) → label, Softmax(scores) → proba.

    :param g: graph builder
    :param sts: shapes
    :param scores: logit or un-normalised score tensor, shape ``(N, C)``
    :param classes: class labels array
    :param name: node name prefix
    :param outputs: desired output names ``[label_name, proba_name]``
    :return: ``(label, proba)``
    """
    proba = g.op.Softmax(scores, axis=1, name=name, outputs=outputs[1:])
    assert isinstance(proba, str)

    label_idx = g.op.ArgMax(scores, axis=1, keepdims=0, name=name)
    label_idx_cast = g.op.Cast(label_idx, to=onnx.TensorProto.INT64, name=name)

    if np.issubdtype(classes.dtype, np.integer):
        classes_arr = classes.astype(np.int64)
        label = g.op.Gather(
            classes_arr, label_idx_cast, axis=0, name=f"{name}_label", outputs=outputs[:1]
        )
        assert isinstance(label, str)
        if not sts:
            g.set_type(label, onnx.TensorProto.INT64)
    else:
        classes_arr = np.array(classes.astype(str))
        label = g.op.Gather(
            classes_arr, label_idx_cast, axis=0, name=f"{name}_label_str", outputs=outputs[:1]
        )
        assert isinstance(label, str)
        if not sts:
            g.set_type(label, onnx.TensorProto.STRING)
    return label, proba


# ── GaussianProcessRegressor converter ───────────────────────────────────────


@register_sklearn_converter(GaussianProcessRegressor)
def sklearn_gaussian_process_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: GaussianProcessRegressor,
    X: str,
    name: str = "gpr",
) -> Union[str, Tuple[str, ...]]:
    """
    Converts a :class:`sklearn.gaussian_process.GaussianProcessRegressor`
    into ONNX (mean prediction only).

    The predictive mean follows Algorithm 2.1 of *Gaussian Processes for
    Machine Learning* (Rasmussen & Williams 2006, p. 19):

    .. code-block:: text

        K_trans = kernel(X, X_train)          # (N, M)
        y_mean  = K_trans @ α + y_train_mean  # (N,)  or  (N, n_targets)

    where ``α`` (``estimator.alpha_``) and the denormalisation offsets
    (``_y_train_mean``, ``_y_train_std``) are stored as ONNX constants.

    **Supported kernels** — see :func:`_emit_kernel_matrix` for the full list.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a fitted ``GaussianProcessRegressor``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name (predictions)
    """
    assert isinstance(
        estimator, GaussianProcessRegressor
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    X_train = estimator.X_train_.astype(dtype)  # (M, F)
    alpha = estimator.alpha_.astype(dtype)  # (M,) or (M, n_targets)
    # _y_train_mean/_y_train_std may be 0-d scalars (normalize_y=True) or 1-D arrays
    y_mean_train = np.atleast_1d(estimator._y_train_mean).astype(dtype)
    y_std_train = np.atleast_1d(estimator._y_train_std).astype(dtype)

    # K_trans = kernel(X, X_train) → (N, M)
    K_trans = _emit_kernel_matrix(g, estimator.kernel_, X, X_train, f"{name}_K", dtype)

    if alpha.ndim == 1:
        # Single-target: alpha is (M,), reshape to (M, 1) for MatMul
        y_raw = g.op.MatMul(K_trans, alpha.reshape(-1, 1), name=f"{name}_raw")  # (N, 1)
        # Undo normalisation: y = y_std * y_raw + y_mean
        y_unnorm = g.op.Add(
            g.op.Mul(y_raw, np.array(float(y_std_train[0]), dtype=dtype), name=f"{name}_std"),
            np.array(float(y_mean_train[0]), dtype=dtype),
            name=f"{name}_unnorm",
        )
        # Squeeze (N, 1) → (N,)
        result = g.op.Reshape(
            y_unnorm, np.array([-1], dtype=np.int64), name=name, outputs=outputs[:1]
        )
    else:
        # Multi-target: alpha is (M, n_targets)
        y_raw = g.op.MatMul(K_trans, alpha, name=f"{name}_raw")  # (N, n_targets)
        # Undo normalisation: y = y_std * y_raw + y_mean  (broadcast over N)
        result = g.op.Add(
            g.op.Mul(y_raw, y_std_train, name=f"{name}_std"),
            y_mean_train,
            name=name,
            outputs=outputs[:1],
        )

    assert isinstance(result, str)
    if not sts:
        g.set_type(result, itype)
    return result


# ── GaussianProcessClassifier converter ──────────────────────────────────────


@register_sklearn_converter(GaussianProcessClassifier)
def sklearn_gaussian_process_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: GaussianProcessClassifier,
    X: str,
    name: str = "gpc",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.gaussian_process.GaussianProcessClassifier`
    into ONNX.

    **Binary classification** (two classes):

    .. code-block:: text

        K_trans  = kernel(X, X_train)                      # (N, M)
        f_mean   = K_trans @ (y_train - π)                 # (N,)   posterior mean
        v        = M_pre @ K_trans.T                       # (M, N) M_pre precomputed
        f_var    = diag(K(X,X)) - Σ_col(v²)              # (N,)   posterior variance
        π*       ≈ Williams-Barber 5-point approximation   # (N,)
        proba    = [[1-π*, π*]]                             # (N, 2)

    where ``M_pre = L⁻¹ · diag(W_sr)`` is precomputed once at conversion
    time.

    **Multiclass** (``multi_class="one_vs_rest"``):

    Each binary sub-estimator produces ``π*ₖ``; these are stacked and
    row-normalised:

    .. code-block:: text

        Y[:, k]  = π*ₖ  for k in 0…C-1
        proba    = Y / row_sum(Y)

    **Supported kernels** — see :func:`_emit_kernel_matrix` for the full list.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names (label, probabilities)
    :param estimator: a fitted ``GaussianProcessClassifier``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: tuple ``(label_result_name, proba_result_name)``
    :raises NotImplementedError: for ``multi_class="one_vs_one"`` with >2 classes
    """
    assert isinstance(
        estimator, GaussianProcessClassifier
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    if estimator.n_classes_ > 2 and estimator.multi_class == "one_vs_one":
        raise NotImplementedError(
            "GaussianProcessClassifier with multi_class='one_vs_one' and "
            "more than two classes is not supported in ONNX conversion. "
            "Use multi_class='one_vs_rest' instead."
        )

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)
    classes = estimator.classes_

    be = estimator.base_estimator_

    if estimator.n_classes_ == 2:
        # Binary: base_estimator_ is _BinaryGaussianProcessClassifierLaplace
        pi_star = _emit_gpc_binary_pi_star(g, be.kernel_, be, f"{name}_bin", X, dtype)
        # proba = [[1-π*, π*]]  → (N, 2)
        pi_star_2d = g.op.Unsqueeze(pi_star, np.array([1], dtype=np.int64), name=f"{name}_ps2d")
        one_minus = g.op.Sub(np.array(1.0, dtype=dtype), pi_star_2d, name=f"{name}_1mp")
        proba = g.op.Concat(one_minus, pi_star_2d, axis=1, name=name, outputs=outputs[1:])
        assert isinstance(proba, str)

        label_idx = g.op.ArgMax(proba, axis=1, keepdims=0, name=name)
        label_idx_int64 = g.op.Cast(label_idx, to=onnx.TensorProto.INT64, name=name)
        if np.issubdtype(classes.dtype, np.integer):
            label = g.op.Gather(
                classes.astype(np.int64),
                label_idx_int64,
                axis=0,
                name=f"{name}_label",
                outputs=outputs[:1],
            )
            assert isinstance(label, str)
            if not sts:
                g.set_type(label, onnx.TensorProto.INT64)
        else:
            label = g.op.Gather(
                np.array(classes.astype(str)),
                label_idx_int64,
                axis=0,
                name=f"{name}_label_str",
                outputs=outputs[:1],
            )
            assert isinstance(label, str)
            if not sts:
                g.set_type(label, onnx.TensorProto.STRING)
        return label, proba

    # Multiclass: base_estimator_ is OneVsRestClassifier
    pi_star_cols = []
    for i, bin_est in enumerate(be.estimators_):
        pi_i = _emit_gpc_binary_pi_star(
            g, bin_est.kernel_, bin_est, f"{name}_est{i}", X, dtype
        )  # (N,)
        pi_star_cols.append(
            g.op.Unsqueeze(pi_i, np.array([1], dtype=np.int64), name=f"{name}_est{i}_2d")
        )  # (N, 1)

    # Concatenate → (N, n_classes)
    Y = g.op.Concat(*pi_star_cols, axis=1, name=f"{name}_Y")
    # Normalise rows: Y / row_sum(Y)
    row_sums = g.op.ReduceSum(Y, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_rsum")
    proba = g.op.Div(Y, row_sums, name=name, outputs=outputs[1:])
    assert isinstance(proba, str)

    label_idx = g.op.ArgMax(proba, axis=1, keepdims=0, name=name)
    label_idx_int64 = g.op.Cast(label_idx, to=onnx.TensorProto.INT64, name=name)
    if np.issubdtype(classes.dtype, np.integer):
        label = g.op.Gather(
            classes.astype(np.int64),
            label_idx_int64,
            axis=0,
            name=f"{name}_label",
            outputs=outputs[:1],
        )
        assert isinstance(label, str)
        if not sts:
            g.set_type(label, onnx.TensorProto.INT64)
    else:
        label = g.op.Gather(
            np.array(classes.astype(str)),
            label_idx_int64,
            axis=0,
            name=f"{name}_label_str",
            outputs=outputs[:1],
        )
        assert isinstance(label, str)
        if not sts:
            g.set_type(label, onnx.TensorProto.STRING)
    return label, proba
