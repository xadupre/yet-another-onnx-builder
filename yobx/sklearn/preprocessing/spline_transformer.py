import numpy as np
from typing import Dict, List

from sklearn.preprocessing import SplineTransformer

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


def _bspline_poly_coefficients(t: np.ndarray, degree: int):
    """
    Precompute polynomial piece coefficients for the B-spline design matrix.

    For a B-spline with knot vector *t* and *degree* this returns the
    piecewise-polynomial representation of every basis function: for each
    interval ``[bp[k], bp[k+1])`` and each basis function ``j``,
    ``coeff[k, j, p]`` is the coefficient of ``(x - bp[k])^p``.

    The conversion is exact (up to floating-point precision) because B-splines
    are polynomials within each knot interval.

    :param t: full knot vector (sorted, may have repeated values)
    :param degree: degree of the B-spline
    :return: tuple ``(bp, coeff)`` where

        * ``bp`` — unique breakpoints, shape ``(n_bp,)``
        * ``coeff`` — coefficient array, shape ``(n_intervals, n_splines, degree+1)``
    """
    from scipy.interpolate import PPoly

    n_splines = len(t) - degree - 1
    bp = np.unique(t)
    n_intervals = len(bp) - 1

    coeff = np.zeros((n_intervals, n_splines, degree + 1))
    for j in range(n_splines):
        c_j = np.zeros(n_splines)
        c_j[j] = 1.0
        pp = PPoly.from_spline((t, c_j, degree))
        # pp.c shape: (degree+1, n_intervals), high-degree first → flip to low-degree first.
        coeff[:, j, :] = pp.c[::-1, :].T

    return bp, coeff


@register_sklearn_converter(SplineTransformer)
def sklearn_spline_transformer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: SplineTransformer,
    X: str,
    name: str = "spline_transformer",
) -> str:
    """
    Converts a :class:`sklearn.preprocessing.SplineTransformer` into ONNX.

    The implementation evaluates the B-spline design matrix feature-by-feature
    using precomputed piecewise-polynomial coefficients (one polynomial piece
    per knot interval per basis function).  For each input feature the
    following graph is produced:

    .. code-block:: text

        x_col (N,)
          │
          ├─Clip(xmin, xmax)──► x_eff (N,)          [only for extrapolation='constant']
          │
          ├─Unsqueeze──LessOrEqual(bp)──Cast──ReduceSum──Sub(1)──Clip ──► seg (N,)
          │                                                                  │
          ├─Gather(bp, seg)──────────────────────────────────────────► bp_seg (N,)
          │
          ├─Sub(bp_seg)──► x_rel (N,)
          │
          ├─Unsqueeze──Pow(powers)──► poly_basis (N, d+1)
          │
          Gather(coeff, seg)──► gathered (N, n_splines, d+1)
          │
          MatMul(Unsqueeze(poly_basis), Transpose(gathered))──Squeeze ──► dm_i (N, n_splines)

    All per-feature design matrices are concatenated along axis 1.  When
    ``include_bias=False``, the last spline column of each feature block is
    dropped via a ``Gather`` on axis 1.

    Supported ``extrapolation`` modes:

    * ``'constant'`` (default) — clamp the input to the training range; values
      outside the range receive the boundary basis-function values.
    * ``'continue'`` — continue the polynomial of the boundary interval; no
      clamping is applied.

    Other modes (``'error'``, ``'linear'``, ``'periodic'``) are not yet
    supported.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``SplineTransformer``
    :param outputs: desired output names
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output tensor name
    :raises NotImplementedError: for unsupported ``extrapolation`` modes or
        ``interaction_only=True``
    """
    assert isinstance(
        estimator, SplineTransformer
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    if estimator.extrapolation not in ("constant", "continue"):
        raise NotImplementedError(
            f"SplineTransformer converter does not yet support "
            f"extrapolation={estimator.extrapolation!r}. "
            f"Supported modes: 'constant', 'continue'."
        )

    if getattr(estimator, "interaction_only", False):
        raise NotImplementedError(
            "SplineTransformer converter does not support interaction_only=True."
        )

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    n_features = estimator.n_features_in_
    degree = estimator.degree
    n_splines = estimator.bsplines_[0].c.shape[0]  # basis functions per feature

    powers_const = np.arange(degree + 1, dtype=dtype)  # [0, 1, ..., degree]

    feature_parts: List[str] = []

    for feat_idx in range(n_features):
        spl = estimator.bsplines_[feat_idx]
        t = spl.t.astype(dtype)
        feat_degree = spl.k

        # Precompute polynomial coefficient matrix for this feature's B-splines.
        bp, coeff = _bspline_poly_coefficients(t, feat_degree)
        bp = bp.astype(dtype)
        coeff = coeff.astype(dtype)
        n_intervals = len(bp) - 1

        feat_name = f"{name}_f{feat_idx}"

        # ── Step 1: extract feature column x_col (N,) ─────────────────────────
        # Using a rank-0 index causes Gather to drop the indexed axis from the
        # output shape, giving (N,) rather than (N, 1).
        col_idx = np.array(feat_idx, dtype=np.int64)
        x_col = g.op.Gather(X, col_idx, axis=1, name=f"{feat_name}_col")

        # ── Step 2: handle extrapolation ──────────────────────────────────────
        xmin = float(t[feat_degree])  # left boundary of B-spline support
        xmax = float(t[-feat_degree - 1])  # right boundary of B-spline support

        if estimator.extrapolation == "constant":
            x_eff = g.op.Clip(
                x_col,
                np.array(xmin, dtype=dtype),
                np.array(xmax, dtype=dtype),
                name=f"{feat_name}_clip",
            )
        else:
            x_eff = x_col

        # ── Step 3: find knot-interval segment for each sample ─────────────────
        # seg[n] = largest k s.t. bp[k] <= x_eff[n], clamped to [0, n_intervals-1].
        x_unsq = g.op.Unsqueeze(
            x_eff, np.array([1], dtype=np.int64), name=f"{feat_name}_unsq"
        )  # (N, 1)
        # le[n, k] = 1 if bp[k] <= x_eff[n] else 0
        le = g.op.LessOrEqual(bp, x_unsq, name=f"{feat_name}_le")  # (N, n_bp)
        le_int = g.op.Cast(le, to=7, name=f"{feat_name}_le_int")  # INT64=7
        count = g.op.ReduceSum(
            le_int,
            np.array([1], dtype=np.int64),
            keepdims=0,
            name=f"{feat_name}_count",
        )  # (N,)
        one = np.array(1, dtype=np.int64)
        seg_raw = g.op.Sub(count, one, name=f"{feat_name}_seg_raw")  # (N,)
        seg = g.op.Clip(
            seg_raw,
            np.array(0, dtype=np.int64),
            np.array(n_intervals - 1, dtype=np.int64),
            name=f"{feat_name}_seg",
        )  # (N,)

        # ── Step 4: compute x_rel = x_eff - bp[seg] ───────────────────────────
        bp_for_seg = g.op.Gather(bp, seg, axis=0, name=f"{feat_name}_bp_seg")  # (N,)
        x_rel = g.op.Sub(x_eff, bp_for_seg, name=f"{feat_name}_xrel")  # (N,)

        # ── Step 5: build polynomial basis [1, x_rel, x_rel², ..., x_rel^d] ──
        x_rel_unsq = g.op.Unsqueeze(
            x_rel, np.array([1], dtype=np.int64), name=f"{feat_name}_xrel_unsq"
        )  # (N, 1)
        poly_basis = g.op.Pow(
            x_rel_unsq, powers_const, name=f"{feat_name}_poly"
        )  # (N, d+1) via broadcasting

        # ── Step 6: gather coefficient matrix rows for each segment ─────────
        coeff_const = coeff  # shape (n_intervals, n_splines, d+1)
        gathered = g.op.Gather(
            coeff_const, seg, axis=0, name=f"{feat_name}_gather"
        )  # (N, n_splines, d+1)

        # ── Step 7: evaluate polynomial via batched MatMul ────────────────────
        # poly_basis: (N, d+1) → unsqueeze to (N, 1, d+1)
        # gathered transposed: (N, d+1, n_splines)
        # result = MatMul((N,1,d+1), (N,d+1,n_splines)) = (N, 1, n_splines)
        # → squeeze axis 1 → (N, n_splines)
        poly_3d = g.op.Unsqueeze(
            poly_basis, np.array([1], dtype=np.int64), name=f"{feat_name}_poly3d"
        )  # (N, 1, d+1)
        gathered_T = g.op.Transpose(
            gathered, perm=[0, 2, 1], name=f"{feat_name}_gT"
        )  # (N, d+1, n_splines)
        dm_3d = g.op.MatMul(poly_3d, gathered_T, name=f"{feat_name}_mm")  # (N,1,n_splines)
        dm_i = g.op.Squeeze(
            dm_3d, np.array([1], dtype=np.int64), name=f"{feat_name}_dm"
        )  # (N, n_splines)

        feature_parts.append(dm_i)

    # ── Concatenate all per-feature design matrices ────────────────────────────
    if n_features == 1:
        full_dm = feature_parts[0]
    else:
        full_dm = g.op.Concat(*feature_parts, axis=1, name=f"{name}_concat")
    # full_dm shape: (N, n_features * n_splines)

    # ── Drop last-spline column per feature when include_bias=False ───────────
    if not estimator.include_bias:
        # The last column of each feature's n_splines block is removed.
        keep_cols = np.array(
            [i for i in range(n_features * n_splines) if (i + 1) % n_splines != 0],
            dtype=np.int64,
        )
        res = g.op.Gather(full_dm, keep_cols, axis=1, name=name, outputs=outputs)
    else:
        res = g.op.Identity(full_dm, name=name, outputs=outputs)

    assert isinstance(res, str)  # type happiness
    if not sts:
        g.set_type(res, itype)
        if g.has_shape(X):
            batch_dim = g.get_shape(X)[0]
            g.set_shape(res, (batch_dim, estimator.n_features_out_))
    return res
