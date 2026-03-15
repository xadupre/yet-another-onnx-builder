import numpy as np
import onnx
from typing import Dict, List

from sklearn.isotonic import IsotonicRegression

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(IsotonicRegression)
def sklearn_isotonic_regression(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: IsotonicRegression,
    X: str,
    name: str = "isotonic_regression",
) -> str:
    """
    Converts a :class:`sklearn.isotonic.IsotonicRegression` into ONNX.

    The prediction follows :meth:`sklearn.isotonic.IsotonicRegression.predict`,
    which uses piecewise-linear interpolation via :func:`numpy.interp`:

    .. code-block:: text

        predict(X) = np.interp(X.ravel(), X_thresholds_, y_thresholds_)

    Values below ``X_min_`` are clamped to ``y_thresholds_[0]`` and values
    above ``X_max_`` are clamped to ``y_thresholds_[-1]``.

    **ONNX graph structure** (K breakpoints, K ≥ 2):

    .. code-block:: text

        X (N, 1) or (N,)
          │
          Reshape(-1) ──► x_flat (N,)
          │
          Clip(X_min_, X_max_) ──► x_clipped (N,)
          │
          Unsqueeze(-1) ──► x_exp (N, 1)
          │
        xp (K,) ──── GreaterOrEqual ──► cmp (N, K)  [bool]
          │               │
          │          Cast(INT64) ──► cmp_int (N, K)
          │               │
          │          ReduceSum(axis=1) ──► seg_count (N,)
          │               │
          │         Sub(1) ──► seg_lo_raw (N,)
          │               │
          │         Clip(0, K-2) ──► seg_lo (N,)  int64
          │               │
          │         Add(1) ──► seg_hi (N,)  int64
          │
        Gather(xp, seg_lo) ──► xp_lo (N,)
        Gather(xp, seg_hi) ──► xp_hi (N,)
        Gather(fp, seg_lo) ──► fp_lo (N,)
        Gather(fp, seg_hi) ──► fp_hi (N,)
          │
        dx  = xp_hi - xp_lo
        t   = (x_clipped - xp_lo) / dx
        out = fp_lo + t * (fp_hi - fp_lo)  ──► predictions (N,)

    When all training samples collapse to a single breakpoint (K = 1), the
    graph simply broadcasts the constant ``y_thresholds_[0]`` to all rows.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a fitted ``IsotonicRegression``
    :param X: input tensor name (shape ``(N,)`` or ``(N, 1)``)
    :param name: prefix for added node names
    :return: output tensor name (shape ``(N,)``)
    """
    assert isinstance(
        estimator, IsotonicRegression
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    xp = estimator.X_thresholds_.astype(dtype)  # (K,)
    fp = estimator.y_thresholds_.astype(dtype)  # (K,)
    K = len(xp)

    # Flatten input to 1D: handles both (N,) and (N, 1) shapes.
    x_flat = g.op.Reshape(X, np.array([-1], dtype=np.int64), name=f"{name}_flatten")  # (N,)

    # --- Constant model (K == 1) ------------------------------------------
    if K == 1:
        const = fp.reshape(1)  # (1,)
        x_shape = g.op.Shape(x_flat, name=f"{name}_shape")
        return g.op.Expand(const, x_shape, name=name, outputs=outputs)

    # --- Piecewise-linear interpolation (K >= 2) --------------------------

    # Clip input to [X_min_, X_max_] — matches np.interp boundary behaviour.
    x_min = np.array(float(estimator.X_min_), dtype=dtype)
    x_max = np.array(float(estimator.X_max_), dtype=dtype)
    x_clipped = g.op.Clip(x_flat, x_min, x_max, name=f"{name}_clip")  # (N,)

    # Determine segment index: count how many breakpoints are <= x_clipped.
    x_exp = g.op.Unsqueeze(
        x_clipped, np.array([-1], dtype=np.int64), name=f"{name}_unsqueeze"
    )  # (N, 1)
    # Compare against all breakpoints: (N, 1) >= (K,) → (N, K)
    cmp = g.op.GreaterOrEqual(x_exp, xp, name=f"{name}_cmp")
    cmp_int = g.op.Cast(cmp, to=onnx.TensorProto.INT64, name=f"{name}_cast_cmp")
    # Sum across breakpoint axis to get the count of breakpoints <= x.
    seg_count = g.op.ReduceSum(
        cmp_int, np.array([1], dtype=np.int64), keepdims=0, name=f"{name}_seg_count"
    )  # (N,) int64, values in [0, K]

    # seg_lo = clamp(seg_count - 1, 0, K-2)
    one_i64 = np.array(1, dtype=np.int64)
    seg_lo_raw = g.op.Sub(seg_count, one_i64, name=f"{name}_seg_lo_raw")  # (N,)
    seg_lo = g.op.Clip(
        seg_lo_raw,
        np.array(0, dtype=np.int64),
        np.array(K - 2, dtype=np.int64),
        name=f"{name}_seg_lo",
    )  # (N,) in [0, K-2]
    seg_hi = g.op.Add(seg_lo, one_i64, name=f"{name}_seg_hi")  # (N,) in [1, K-1]

    # Gather breakpoint and target values for the lower and upper segment ends.
    xp_lo = g.op.Gather(xp, seg_lo, axis=0, name=f"{name}_xp_lo")  # (N,)
    xp_hi = g.op.Gather(xp, seg_hi, axis=0, name=f"{name}_xp_hi")  # (N,)
    fp_lo = g.op.Gather(fp, seg_lo, axis=0, name=f"{name}_fp_lo")  # (N,)
    fp_hi = g.op.Gather(fp, seg_hi, axis=0, name=f"{name}_fp_hi")  # (N,)

    # Linear interpolation: t = (x - xp_lo) / (xp_hi - xp_lo)
    dx = g.op.Sub(xp_hi, xp_lo, name=f"{name}_dx")  # (N,)
    x_sub_lo = g.op.Sub(x_clipped, xp_lo, name=f"{name}_x_sub_lo")  # (N,)
    t = g.op.Div(x_sub_lo, dx, name=f"{name}_t")  # (N,)

    # result = fp_lo + t * (fp_hi - fp_lo)
    fp_diff = g.op.Sub(fp_hi, fp_lo, name=f"{name}_fp_diff")  # (N,)
    interp = g.op.Mul(t, fp_diff, name=f"{name}_interp")  # (N,)
    return g.op.Add(fp_lo, interp, name=name, outputs=outputs)  # (N,)
