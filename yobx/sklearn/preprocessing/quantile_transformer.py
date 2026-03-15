import numpy as np
import onnx
from typing import Dict, List
from sklearn.preprocessing import QuantileTransformer
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


def _horner(g, q, coeffs, dtype, name):
    """Horner evaluation of a polynomial in *q*.

    Computes ``coeffs[0]*q^n + coeffs[1]*q^(n-1) + ... + coeffs[-1]``
    where all arithmetic is done via ONNX nodes.

    :param g: graph builder
    :param q: ONNX tensor name (any shape)
    :param coeffs: sequence of scalar coefficients (highest degree first)
    :param dtype: numpy dtype used to cast the constants
    :param name: node name prefix
    :return: ONNX tensor name holding the result
    """
    acc = np.array([coeffs[0]], dtype=dtype)
    for i, c in enumerate(coeffs[1:], start=1):
        acc = g.op.Add(
            g.op.Mul(acc, q, name=f"{name}_h_mul{i}"),
            np.array([c], dtype=dtype),
            name=f"{name}_h_add{i}",
        )
    return acc


def _ndtri_approx(g, p, dtype, name):
    """Rational-function approximation of the inverse normal CDF (ndtri / ppf).

    Uses Acklam's algorithm (max absolute error ≈ 1.15 x 10⁻⁹).

    .. code-block:: text

        Region 1 - low tail  (0 < p < p_low):
            q = sqrt(-2 log p)
            result = poly_c(q) / poly_d(q)

        Region 2 - central   (p_low ≤ p ≤ p_high):
            q = p - 0.5
            r = q²
            result = poly_a(r)·q / poly_b(r)

        Region 3 - high tail (p_high < p < 1):
            result = -ndtri(1 - p)   (by symmetry)

    :param g: graph builder
    :param p: ONNX tensor name holding probabilities (values in (0, 1))
    :param dtype: numpy dtype
    :param name: node name prefix
    :return: ONNX tensor name with the approximated ndtri values
    """
    # ── constants ────────────────────────────────────────────────────────────
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]

    p_low = np.array([0.02425], dtype=dtype)
    p_high = np.array([1.0 - 0.02425], dtype=dtype)
    two = np.array([-2.0], dtype=dtype)
    one = np.array([1.0], dtype=dtype)
    # Clip bounds: use nextafter to ensure the upper bound is strictly < 1.0
    # even in float32 (where 1.0 - 1e-10 rounds to exactly 1.0).
    eps = np.nextafter(np.zeros(1, dtype=dtype), np.ones(1, dtype=dtype))
    eps_hi = np.nextafter(np.ones(1, dtype=dtype), np.zeros(1, dtype=dtype))

    # Clip input to open interval so log / sqrt are defined.
    p_safe = g.op.Clip(p, eps, eps_hi, name=f"{name}_pclip")

    # ── region 1 / 3 (tail): q = sqrt(-2·log(p)) ─────────────────────────
    log_p = g.op.Log(p_safe, name=f"{name}_log_p")
    neg2_log_p = g.op.Mul(two, log_p, name=f"{name}_n2lp")
    q_tail = g.op.Sqrt(neg2_log_p, name=f"{name}_q_tail")
    num_tail = _horner(g, q_tail, c, dtype, f"{name}_nc")
    # Denominator: (((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1
    den_tail = g.op.Add(
        g.op.Mul(_horner(g, q_tail, d, dtype, f"{name}_dc"), q_tail, name=f"{name}_dc_q"),
        one,
        name=f"{name}_den_tail",
    )
    x_low = g.op.Div(num_tail, den_tail, name=f"{name}_x_low")

    # ── region 3 (high tail): symmetry x_high = -ndtri(1-p) ─────────────
    p1m = g.op.Sub(one, p_safe, name=f"{name}_1mp")
    log_1mp = g.op.Log(p1m, name=f"{name}_log_1mp")
    neg2_log_1mp = g.op.Mul(two, log_1mp, name=f"{name}_n2l1mp")
    q_high = g.op.Sqrt(neg2_log_1mp, name=f"{name}_q_high")
    num_high = _horner(g, q_high, c, dtype, f"{name}_nhc")
    # Denominator: (((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1
    den_high = g.op.Add(
        g.op.Mul(_horner(g, q_high, d, dtype, f"{name}_dhc"), q_high, name=f"{name}_dhc_q"),
        one,
        name=f"{name}_den_high",
    )
    x_high = g.op.Neg(
        g.op.Div(num_high, den_high, name=f"{name}_x_high_raw"), name=f"{name}_x_high"
    )

    # ── region 2 (central) ───────────────────────────────────────────────
    q_cen = g.op.Sub(p_safe, np.array([0.5], dtype=dtype), name=f"{name}_q_cen")
    r = g.op.Mul(q_cen, q_cen, name=f"{name}_r")
    num_cen = g.op.Mul(_horner(g, r, a, dtype, f"{name}_na"), q_cen, name=f"{name}_num_cen")
    # Denominator: ((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r + 1
    den_cen = g.op.Add(
        g.op.Mul(_horner(g, r, b, dtype, f"{name}_db"), r, name=f"{name}_db_r"),
        one,
        name=f"{name}_den_cen",
    )
    x_cen = g.op.Div(num_cen, den_cen, name=f"{name}_x_cen")

    # ── select region by two Where ops ───────────────────────────────────
    is_low = g.op.Less(p_safe, p_low, name=f"{name}_is_low")
    is_high = g.op.Greater(p_safe, p_high, name=f"{name}_is_high")
    x_cen_or_high = g.op.Where(is_high, x_high, x_cen, name=f"{name}_ch")
    result = g.op.Where(is_low, x_low, x_cen_or_high, name=f"{name}_ndtri")
    return result


def _interp_1d_batch(g, X, q_T_const, r_const, dtype, name):
    """Vectorised ``np.interp`` for every feature of *X* simultaneously.

    Implements piecewise linear interpolation::

        for each sample i and feature j:
            output[i, j] = np.interp(X[i, j], q_T_const[j, :], r_const)

    The indices are computed by counting how many quantile values each
    input element exceeds (``Greater`` + ``ReduceSum``).  A flat-index
    ``Gather`` is used to look up the per-feature quantile boundaries
    without requiring ``GatherND`` (which complicates type inference when
    the index tensor has a dynamic batch dimension).

    :param g: graph builder
    :param X: input tensor name - shape ``(N, F)``
    :param q_T_const: numpy array ``(F, Q)`` - sorted quantile values per feature
    :param r_const: numpy array ``(Q,)`` - reference values (same for all features)
    :param dtype: numpy dtype of the computation
    :param name: node name prefix
    :return: tensor name - shape ``(N, F)``
    """
    n_features, n_quantiles = q_T_const.shape
    q_flat = q_T_const.astype(dtype).reshape(-1)  # (F*Q,) - row-major
    r = r_const.astype(dtype)  # (Q,)

    zero_i = np.array([0], dtype=np.int64)
    nq_m1 = np.array([n_quantiles - 1], dtype=np.int64)
    one_i = np.array([1], dtype=np.int64)
    zero_f = np.array([0.0], dtype=dtype)
    one_f = np.array([1.0], dtype=dtype)

    # ── Step 1: count how many quantile boundaries X exceeds ─────────────
    # X_exp : (N, F, 1),  q_T_exp : (1, F, Q)
    X_exp = g.op.Unsqueeze(X, np.array([2], dtype=np.int64), name=f"{name}_unsq")
    q_T_exp = q_T_const.astype(dtype).reshape(1, n_features, n_quantiles)  # (1, F, Q)
    mask = g.op.Greater(X_exp, q_T_exp, name=f"{name}_gt")
    # idx[i, j] = sum_k  X[i,j] > q_T[j, k]  →  (N, F)
    idx = g.op.ReduceSum(
        g.op.Cast(mask, to=onnx.TensorProto.INT64, name=f"{name}_cast"),
        np.array([2], dtype=np.int64),
        keepdims=0,
        name=f"{name}_idx",
    )

    # ── Step 2: clamp to valid bucket range ──────────────────────────────
    left = g.op.Clip(
        g.op.Sub(idx, one_i, name=f"{name}_idxm1"), zero_i, nq_m1, name=f"{name}_left"
    )
    right = g.op.Clip(idx, zero_i, nq_m1, name=f"{name}_right")

    # ── Step 3: flat-index Gather for quantile boundary values ───────────
    # q_flat is laid out row-major: q_flat[j*Q + k] = q_T[j, k].
    # Build flat offsets: feature_offset[j] = j * Q  →  shape (1, F), broadcast.
    feature_offset = (np.arange(n_features, dtype=np.int64) * n_quantiles).reshape(
        1, n_features
    )  # (1, F) constant

    flat_left = g.op.Add(feature_offset, left, name=f"{name}_fl")  # (N, F)
    flat_right = g.op.Add(feature_offset, right, name=f"{name}_fr")  # (N, F)

    # Gather quantile x-values from the flat array.
    xp_left = g.op.Gather(q_flat, flat_left, axis=0, name=f"{name}_xpl")  # (N, F)
    xp_right = g.op.Gather(q_flat, flat_right, axis=0, name=f"{name}_xpr")  # (N, F)

    # Gather reference (y) values - r is 1-D, same for all features.
    fp_left = g.op.Gather(r, left, axis=0, name=f"{name}_fpl")  # (N, F)
    fp_right = g.op.Gather(r, right, axis=0, name=f"{name}_fpr")  # (N, F)

    # ── Step 4: linear interpolation ─────────────────────────────────────
    denom = g.op.Sub(xp_right, xp_left, name=f"{name}_denom")
    # Avoid division by zero when left == right (boundary or tie).
    safe_denom = g.op.Where(
        g.op.Equal(denom, zero_f, name=f"{name}_eq0"), one_f, denom, name=f"{name}_sd"
    )
    t = g.op.Clip(
        g.op.Div(g.op.Sub(X, xp_left, name=f"{name}_xdiff"), safe_denom, name=f"{name}_t_raw"),
        zero_f,
        one_f,
        name=f"{name}_t",
    )
    result = g.op.Add(
        fp_left,
        g.op.Mul(t, g.op.Sub(fp_right, fp_left, name=f"{name}_fpdiff"), name=f"{name}_mul"),
        name=f"{name}_res",
    )
    return result


@register_sklearn_converter(QuantileTransformer)
def sklearn_quantile_transformer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: QuantileTransformer,
    X: str,
    name: str = "quantile_transformer",
) -> str:
    """
    Converts a :class:`sklearn.preprocessing.QuantileTransformer` into ONNX.

    The transformation maps each feature to a uniform or normal distribution
    using piecewise linear interpolation through the fitted quantile values.

    .. code-block:: text

        X  ──interp(quantiles_, references_)──►  uniform [0,1]
                                                    │
                              output_distribution='uniform'? ──► output
                                                    │
                              output_distribution='normal'
                                                    │
                                                ▼  ndtri  ▼
                                          ──Clip(clip_min, clip_max)──► output

    **Interpolation** follows sklearn exactly: for each feature *j* the
    forward transform computes the bidirectional average

    .. code-block:: text

        0.5 · (interp(x, q_j, r) - interp(-x, -q_j_rev, -r_rev))

    which correctly handles tied quantile values (repeated feature values
    in the training data).

    **Normal distribution** output uses Acklam's rational-function
    approximation of the inverse normal CDF (max error ≈ 1.15 x 10⁻⁹).

    Minimum opset requirements:

    * opset ≥ 13 - ``ReduceSum`` with axes as input tensor

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``QuantileTransformer``
    :param outputs: desired output names
    :param X: input name (shape ``(N, F)``)
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(
        estimator, QuantileTransformer
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # quantiles_ shape: (n_quantiles, n_features); transpose to (n_features, n_quantiles)
    q_T = estimator.quantiles_.T.astype(dtype)  # (F, Q)
    r = estimator.references_.astype(dtype)  # (Q,)

    # ── forward interpolation ────────────────────────────────────────────
    r1 = _interp_1d_batch(g, X, q_T, r, dtype, f"{name}_fwd")

    # ── reverse interpolation: interp(-X, -q_rev_T, -r_rev) ─────────────
    # sklearn uses the bidirectional average to handle tied quantile values.
    q_T_rev = (-q_T)[:, ::-1]  # (F, Q) — negated and column-reversed
    r_rev = (-r)[::-1]  # (Q,)   — negated and reversed
    neg_X = g.op.Neg(X, name=f"{name}_neg")
    r2 = _interp_1d_batch(g, neg_X, q_T_rev, r_rev, dtype, f"{name}_bwd")

    # result = 0.5 * (r1 - r2)
    half = np.array([0.5], dtype=dtype)
    uniform = g.op.Mul(g.op.Sub(r1, r2, name=f"{name}_bidir_sub"), half, name=f"{name}_uniform")

    if estimator.output_distribution == "uniform":
        res = g.op.Identity(uniform, name=name, outputs=outputs)
    else:
        # normal distribution: apply inverse normal CDF then clip.
        ndtri_out = _ndtri_approx(g, uniform, dtype, f"{name}_ndtri")

        # sklearn clips to ndtri(BOUNDS_THRESHOLD ± spacing) ≈ ±5.199.
        BOUNDS_THRESHOLD = 1e-7
        clip_min = float(np.float64(_acklam_scalar(BOUNDS_THRESHOLD - np.spacing(np.float64(1)))))
        clip_max = float(
            np.float64(_acklam_scalar(1.0 - (BOUNDS_THRESHOLD - np.spacing(np.float64(1)))))
        )
        res = g.op.Clip(
            ndtri_out,
            np.array([clip_min], dtype=dtype),
            np.array([clip_max], dtype=dtype),
            name=name,
            outputs=outputs,
        )

    assert isinstance(res, str)
    if not sts:
        g.set_type_shape_unary_op(res, X)
    return res


# ── scalar helper used only at converter-build time ─────────────────────────


def _acklam_scalar(p: float) -> float:
    """Evaluate Acklam's ndtri approximation for a single scalar *p*.

    Used at graph-construction time to compute clip bounds that match
    sklearn's ``scipy.stats.norm.ppf`` values.

    This is a plain-Python scalar mirror of the ONNX-graph version
    :func:`_ndtri_approx`.  The two implementations are kept separate
    because :func:`_ndtri_approx` builds ONNX nodes using the graph
    builder API, while this function evaluates a *single* Python float
    at converter-build time (when building the ``Clip`` bound constants).
    """
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]

    p_low = 0.02425
    p = max(1e-10, min(1.0 - 1e-10, p))

    def poly(coeffs, x):
        v = coeffs[0]
        for ci in coeffs[1:]:
            v = v * x + ci
        return v

    if p < p_low:
        q = (-2.0 * np.log(p)) ** 0.5
        return poly(c, q) / (poly(d, q) * q + 1.0)
    if p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        return poly(a, r) * q / (poly(b, r) * r + 1.0)
    q = (-2.0 * np.log(1.0 - p)) ** 0.5
    return -(poly(c, q) / (poly(d, q) * q + 1.0))
