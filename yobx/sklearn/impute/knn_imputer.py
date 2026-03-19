import numpy as np
from typing import Dict, List

from sklearn.impute import KNNImputer

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype

# Metric name used with com.microsoft.CDist for the base squared-distance matrix.
# CDist sqeuclidean = ||x||² + ||y||² - 2·x·yᵀ over *all* features including NaN-
# zeroed positions; two MatMul corrections remove the spurious contributions.
_KNN_IMPUTER_CDIST_METRIC = "sqeuclidean"


@register_sklearn_converter(KNNImputer)
def sklearn_knn_imputer(  # noqa: C901
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: KNNImputer,
    X: str,
    name: str = "knn_imputer",
) -> str:
    """
    Converts a :class:`sklearn.impute.KNNImputer` into ONNX.

    Each missing value in feature *f* of a test sample *i* is replaced by
    the (weighted) mean of the ``n_neighbors`` nearest training samples
    that also have a **valid** (non-NaN) value in feature *f*.  Distances
    are computed with the *nan-euclidean* metric (features that are NaN in
    either sample are excluded from the distance calculation).

    Both ``weights='uniform'`` and ``weights='distance'`` are supported.

    **Distance computation**

    When the ``com.microsoft`` ONNX domain is registered,
    ``com.microsoft.CDist`` (with ``metric="sqeuclidean"``) is used as a
    hardware-accelerated starting point.  CDist computes
    ``||x_filled[i] - train_filled[j]||²`` over *all* features including
    NaN-zeroed positions; two MatMul corrections remove the extra terms:

    .. code-block:: text

        nan_sq[i,j] = CDist(x_filled, train_filled)[i,j]      (sqeuclidean)
                    - MatMul(nan_x_float, train_sq.T)[i,j]     (y_f² where test NaN)
                    - MatMul(ax_sq, nan_train.T)[i,j]          (x_f² where train NaN)

    When CDist is not available the same result is obtained via four
    standard MatMul operations:

    .. code-block:: text

        nan_sq[i,j] = MatMul(x_sq, train_valid.T)[i,j]
                    - 2 × MatMul(x_filled, train_filled.T)[i,j]
                    + MatMul(vx_float, train_sq.T)[i,j]

    In both cases *n_valid* (count of features valid in both samples) is
    computed separately and used to scale the result:

    .. code-block:: text

        n_valid[i,j] = MatMul(vx_float, train_valid.T)[i,j]
        dist_sq[i,j] = n_features / max(n_valid[i,j], 1) × nan_sq[i,j]
        dist_sq[i,j] = inf  if n_valid[i,j] == 0
        dists[i,j]   = sqrt(dist_sq[i,j])

    **Imputation (per feature)**

    For each feature *f*, distances to training samples with NaN in feature
    *f* are set to infinity before ``TopK`` so that only valid donors are
    selected:

    .. code-block:: text

        D_f = Where(train_valid[:, f], dists, inf)          [N, M]
        top_k_dists_f, top_k_idx_f = TopK(D_f, k, axis=1, largest=False)
                                                            [N, k]
        neighbor_vals_f = Gather(train_filled[:, f], top_k_idx_f.flat())
                        .reshape(N, k)
        is_inf_f = IsInf(top_k_dists_f)               [N, k] — fewer than k donors?
        valid_float_f = Cast(~is_inf_f, float)         [N, k]
        imputed_f = sum(neighbor_vals_f * valid_float_f) / max(sum(valid_float_f), 1)

    The imputed column vectors are concatenated and ``Where``-selected
    against the original *X* (NaN preserved where no valid donors exist):

    .. code-block:: text

        imputed = Concat(imputed_0, ..., imputed_{F-1}, axis=1)  [N, F]
        result  = Where(IsNaN(X), imputed, X)                    [N, F]

    ``add_indicator=True`` is not supported and raises
    :class:`NotImplementedError`.  Custom callable ``metric`` values are
    not supported and raise :class:`NotImplementedError`.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``KNNImputer``
    :param outputs: desired output names
    :param X: input name
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(estimator, KNNImputer), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    if getattr(estimator, "add_indicator", False):
        raise NotImplementedError("KNNImputer with add_indicator=True is not supported.")

    metric = getattr(estimator, "metric", "nan_euclidean")
    if callable(metric) or metric != "nan_euclidean":
        raise NotImplementedError(
            f"KNNImputer converter only supports metric='nan_euclidean', got {metric!r}."
        )

    opset = g.get_opset("")
    if opset < 13:
        raise NotImplementedError(
            f"KNNImputer converter requires opset >= 13 "
            f"(ReduceSum with axes as input was added in opset 13), "
            f"but the graph builder has opset {opset}."
        )

    # ------------------------------------------------------------------
    # Extract fitted attributes
    # ------------------------------------------------------------------
    fit_X = estimator._fit_X  # [M, F], training data (may have NaN)
    mask_fit_X = estimator._mask_fit_X  # [M, F], True = missing
    n_neighbors = estimator.n_neighbors
    weights = estimator.weights

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    _M, F = fit_X.shape

    # Training matrices with NaN replaced by 0.
    train_filled = np.where(mask_fit_X, 0, fit_X).astype(dtype)  # [M, F]
    train_valid = (~mask_fit_X).astype(dtype)  # [M, F], 1.0 = valid
    train_sq = (train_filled**2).astype(dtype)  # [M, F]

    # Transposes for the MatMul-based distance computation (standard path).
    train_valid_T = train_valid.T.astype(dtype)  # [F, M]
    train_filled_T = train_filled.T.astype(dtype)  # [F, M]
    train_sq_T = train_sq.T.astype(dtype)  # [F, M]

    # NaN mask of training data as float, transposed — for CDist correction path.
    mask_fit_X_float_T = mask_fit_X.astype(dtype).T  # [F, M], 1.0 = NaN in train

    # ------------------------------------------------------------------
    # Step 1: Build NaN mask and zero-filled test data
    # ------------------------------------------------------------------
    mask_X_node = g.op.IsNaN(X, name=f"{name}_isnan")  # [N, F] bool
    valid_X_node = g.op.Not(mask_X_node, name=f"{name}_valid_x")  # [N, F] bool
    vx_float = g.op.Cast(
        valid_X_node, to=itype, name=f"{name}_vx_float"
    )  # [N, F] float: 1.0 = valid

    zeros1 = np.zeros(1, dtype=dtype)
    x_filled = g.op.Where(mask_X_node, zeros1, X, name=f"{name}_x_filled")  # [N, F]
    ax_sq = g.op.Mul(x_filled, x_filled, name=f"{name}_ax_sq")  # [N, F]

    one1 = np.ones(1, dtype=dtype)

    # ------------------------------------------------------------------
    # Step 2: Compute NaN-aware euclidean distances  →  dists [N, M]
    # ------------------------------------------------------------------
    # n_valid[i,j] = count of features valid in both test_i and train_j.
    # Always computed — needed for scaling regardless of path.
    n_valid = g.op.MatMul(vx_float, train_valid_T, name=f"{name}_n_valid")  # [N, M]

    if g.has_opset("com.microsoft"):
        # ----- CDist fast path -------------------------------------------
        # CDist(x_filled, train_filled, sqeuclidean)[i,j]
        #   = ||x_filled[i] - train_filled[j]||²   (all features, NaN→0)
        #   = nan_sq[i,j]
        #     + Σ_f y_f²[j] * x_nan[i,f]   (y terms where test is NaN)
        #     + Σ_f x_f²[i] * y_nan[j,f]   (x terms where train is NaN)
        # => nan_sq = CDist² - y_sq_invalid - x_sq_invalid
        train_filled_init = g.make_initializer(f"{name}_train_filled", train_filled)
        sq_dists_raw = g.make_node(
            "CDist",
            [x_filled, train_filled_init],
            domain="com.microsoft",
            metric=_KNN_IMPUTER_CDIST_METRIC,
            name=f"{name}_cdist",
        )
        # nan_x_float[i,f] = 1.0 where test[i,f] is NaN.
        nan_x_float = g.op.Sub(one1, vx_float, name=f"{name}_nan_x_float")
        # Subtract contributions from NaN-masked positions.
        y_sq_invalid = g.op.MatMul(
            nan_x_float, train_sq_T, name=f"{name}_ysq_inv"
        )  # [N, M]: Σ y_f² where test is NaN
        x_sq_invalid = g.op.MatMul(
            ax_sq, mask_fit_X_float_T, name=f"{name}_xsq_inv"
        )  # [N, M]: Σ x_f² where train is NaN
        sub1 = g.op.Sub(sq_dists_raw, y_sq_invalid, name=f"{name}_sq1")
        sum_sq = g.op.Sub(sub1, x_sq_invalid, name=f"{name}_sum_sq_raw")
    else:
        # ----- Standard four-MatMul path ---------------------------------
        cross = g.op.MatMul(x_filled, train_filled_T, name=f"{name}_cross")  # [N, M]
        x_sq_contrib = g.op.MatMul(ax_sq, train_valid_T, name=f"{name}_x_sq_c")  # [N, M]
        t_sq_contrib = g.op.MatMul(vx_float, train_sq_T, name=f"{name}_t_sq_c")  # [N, M]
        two = np.array([2.0], dtype=dtype)
        two_cross = g.op.Mul(two, cross, name=f"{name}_two_cross")
        sq_plus = g.op.Add(x_sq_contrib, t_sq_contrib, name=f"{name}_sq_plus")
        sum_sq = g.op.Sub(sq_plus, two_cross, name=f"{name}_sum_sq_raw")

    sum_sq = g.op.Max(sum_sq, zeros1, name=f"{name}_sum_sq_clip")  # clip rounding errors
    denom = g.op.Max(n_valid, one1, name=f"{name}_denom")
    n_feat_arr = np.array([float(F)], dtype=dtype)
    scale = g.op.Div(n_feat_arr, denom, name=f"{name}_scale")
    dist_sq = g.op.Mul(scale, sum_sq, name=f"{name}_dist_sq")

    # Distance is inf when no features are valid in both samples.
    inf_arr = np.array([np.inf], dtype=dtype)
    n_valid_zero = g.op.Equal(n_valid, zeros1, name=f"{name}_n_valid_zero")
    dist_sq = g.op.Where(n_valid_zero, inf_arr, dist_sq, name=f"{name}_dist_sq_safe")
    dists = g.op.Sqrt(dist_sq, name=f"{name}_dists")  # [N, M]

    # ------------------------------------------------------------------
    # Step 3: Per-feature imputation
    #
    # For each feature f, distances to training samples that have NaN in
    # feature f are set to inf so that TopK selects only valid donors.
    # This matches sklearn's behaviour of using min(k, |valid_donors_f|)
    # nearest valid donors for feature f.
    # ------------------------------------------------------------------
    k_arr = np.array([n_neighbors], dtype=np.int64)
    axis1 = np.array([1], dtype=np.int64)
    nk_shape = np.array([-1, n_neighbors], dtype=np.int64)

    eps = None
    if weights == "distance":
        eps = np.array([np.finfo(dtype).eps], dtype=dtype)

    imputed_cols: List[str] = []
    for f in range(F):
        # Boolean mask: True where training row j has a valid value for feature f.
        valid_donors_f = (~mask_fit_X[:, f]).astype(bool)  # [M]
        # Column f of the zero-filled training matrix.
        train_col_f = train_filled[:, f].astype(dtype)  # [M]

        # Make non-valid donors unreachable by setting their distance to inf.
        D_f = g.op.Where(valid_donors_f, dists, inf_arr, name=f"{name}_D_{f}")  # [N, M]

        # k nearest valid donors.
        top_k_dists_f, top_k_idx_f = g.op.TopK(
            D_f, k_arr, axis=1, largest=0, sorted=1, name=f"{name}_topk_{f}"
        )  # [N, k] each

        # Gather the feature-f values of the k selected donors.
        flat_idx_f = g.op.Reshape(
            top_k_idx_f, np.array([-1], dtype=np.int64), name=f"{name}_fidx_{f}"
        )  # [N*k]
        neighbor_vals_f = g.op.Gather(
            train_col_f, flat_idx_f, axis=0, name=f"{name}_nv_{f}"
        )  # [N*k]
        neighbor_vals_f = g.op.Reshape(
            neighbor_vals_f, nk_shape, name=f"{name}_nv2d_{f}"
        )  # [N, k]

        # Validity flag: True when distance is finite (i.e. there were enough
        # valid donors to fill all k slots).
        is_inf_f = g.op.IsInf(top_k_dists_f, name=f"{name}_isinf_{f}")
        valid_mask_f = g.op.Not(is_inf_f, name=f"{name}_vmask_{f}")
        valid_float_f = g.op.Cast(
            valid_mask_f, to=itype, name=f"{name}_vfloat_{f}"
        )  # [N, k]: 1.0 for finite-dist donors, 0.0 for inf-dist slots

        if weights == "uniform":
            # Weighted sum / count of valid donors.
            wvals_f = g.op.Mul(neighbor_vals_f, valid_float_f, name=f"{name}_wv_{f}")
            sum_f = g.op.ReduceSum(wvals_f, axis1, keepdims=0, name=f"{name}_sum_{f}")  # [N]
            cnt_f = g.op.ReduceSum(
                valid_float_f, axis1, keepdims=0, name=f"{name}_cnt_{f}"
            )  # [N]
            safe_cnt = g.op.Max(cnt_f, one1, name=f"{name}_scnt_{f}")
            imputed_fv = g.op.Div(sum_f, safe_cnt, name=f"{name}_imp_{f}")

        else:  # weights == "distance"
            # Inverse-distance weights; IEEE 754 guarantees 1/inf = 0, so
            # inf-distance (invalid) donors automatically get zero weight.
            assert eps is not None, f"{eps=}, unexpected value"
            inv_dists_f = g.op.Div(
                one1,
                g.op.Max(top_k_dists_f, eps, name=f"{name}_sdists_{f}"),
                name=f"{name}_invd_{f}",
            )  # [N, k]: 0 for inf-dist donors
            # Explicitly zero out invalid donors for numerical safety.
            combined_w_f = g.op.Mul(inv_dists_f, valid_float_f, name=f"{name}_cw_{f}")  # [N, k]
            wvals_f = g.op.Mul(neighbor_vals_f, combined_w_f, name=f"{name}_wv_{f}")
            sum_f = g.op.ReduceSum(wvals_f, axis1, keepdims=0, name=f"{name}_sum_{f}")  # [N]
            sumw_f = g.op.ReduceSum(
                combined_w_f, axis1, keepdims=0, name=f"{name}_sumw_{f}"
            )  # [N]
            safe_sumw = g.op.Max(sumw_f, eps, name=f"{name}_ssumw_{f}")
            imputed_fv = g.op.Div(sum_f, safe_sumw, name=f"{name}_imp_{f}")

        # Unsqueeze to [N, 1] for later Concat along axis=1.
        imputed_col = g.op.Unsqueeze(
            imputed_fv, np.array([1], dtype=np.int64), name=f"{name}_col_{f}"
        )  # [N, 1]
        imputed_cols.append(imputed_col)

    # Combine per-feature columns: [N, F]
    imputed = g.op.Concat(*imputed_cols, axis=1, name=f"{name}_concat")

    # Apply imputation only where the original value is NaN.
    # If all valid donors for a feature are exhausted (inf distances),
    # the imputed value is 0 / max(0, 1) = 0, but the result keeps NaN
    # only when X is NaN (Where keeps original X for non-NaN positions).
    res = g.op.Where(mask_X_node, imputed, X, name=name, outputs=outputs)

    g.set_type_shape_unary_op(res, X)
    return res
