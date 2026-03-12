import numpy as np
import onnx
from typing import Dict, List, Union

from sklearn.preprocessing import KBinsDiscretizer

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(KBinsDiscretizer)
def sklearn_kbins_discretizer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: KBinsDiscretizer,
    X: str,
    name: str = "kbins",
) -> str:
    """
    Converts a :class:`sklearn.preprocessing.KBinsDiscretizer` into ONNX.

    Supported values of the *encode* hyperparameter:

    * ``'ordinal'`` — each feature is replaced by its 0-based integer bin
      index, cast to the input floating-point dtype.  Output shape: ``(N, F)``.
    * ``'onehot-dense'`` and ``'onehot'`` — each feature is one-hot encoded
      into ``n_bins_[j]`` columns.  The one-hot blocks are concatenated along
      axis 1.  Output shape: ``(N, sum(n_bins_))``.

    The bin index for feature *j* is computed by counting how many interior
    thresholds (``bin_edges_[j][1:-1]``) are *less than or equal to* the
    sample value:

    .. code-block:: text

        X (N, F)
          │
          └─Unsqueeze(axis=2)──► X_exp (N, F, 1)
                                      │
        thresholds (1, F, T) ─────────┤
                                      ▼
                              GreaterOrEqual ──► (N, F, T)  bool
                                      │
                                  Cast(int64)
                                      │
                               ReduceSum(axis=2) ──► bin_indices (N, F)  int64
                                      │
                                 Min / Max clip
                                      │
                             [ordinal]  Cast(float) ──► output (N, F)
                       [onehot(-dense)] OneHot + Concat ──► output (N, sum(n_bins_))

    Interior thresholds for features that have fewer bins than the maximum are
    padded with ``+inf`` so that the excess comparisons always yield ``False``
    and contribute 0 to the sum.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``KBinsDiscretizer``
    :param outputs: desired output names
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(
        estimator, KBinsDiscretizer
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    encode = estimator.encode
    n_bins = (
        estimator.n_bins_
        if estimator.n_bins_.dtype == np.int64
        else estimator.n_bins_.astype(np.int64)
    )  # (F,)
    n_features = len(n_bins)

    # Build padded thresholds matrix (F, T) where T = max(n_bins - 1).
    # Features with fewer bins are padded with +inf so comparisons contribute 0.
    max_thresholds = int((n_bins - 1).max())
    thresholds = np.full((n_features, max_thresholds), np.inf, dtype=dtype)
    for j, edges in enumerate(estimator.bin_edges_):
        inner = edges[1:-1].astype(dtype)
        thresholds[j, : len(inner)] = inner

    # X_exp: (N, F, 1) — needed for broadcasting against thresholds (1, F, T).
    X_exp = g.op.Unsqueeze(
        X, np.array([2], dtype=np.int64), name=f"{name}_unsqueeze"
    )  # (N, F, 1)

    # Broadcast comparison: (N, F, T)
    thresholds_3d = thresholds[np.newaxis]  # (1, F, T)
    cmp = g.op.GreaterOrEqual(X_exp, thresholds_3d, name=f"{name}_cmp")

    # Sum True values along the threshold axis → bin_indices (N, F) int64
    cmp_int = g.op.Cast(cmp, to=onnx.TensorProto.INT64, name=f"{name}_cast_cmp")
    bin_indices = g.op.ReduceSum(
        cmp_int,
        np.array([2], dtype=np.int64),
        keepdims=0,
        name=f"{name}_sum",
    )  # (N, F)

    # Clip each feature to [0, n_bins_[j] - 1].
    zeros = np.zeros(n_features, dtype=np.int64)
    max_idx = n_bins - 1  # (F,) int64
    bin_indices = g.op.Max(bin_indices, zeros, name=f"{name}_clip_lo")
    bin_indices = g.op.Min(bin_indices, max_idx, name=f"{name}_clip_hi")

    if encode == "ordinal":
        res = g.op.Cast(
            bin_indices, to=itype, name=f"{name}_cast_out", outputs=outputs
        )
        assert isinstance(res, str)
        if not sts:
            g.set_type_shape_unary_op(res, X)
        return res

    # onehot / onehot-dense — one-hot encode each feature then concatenate.
    #
    # ONNX OneHot(indices, depth, values) expects:
    #   indices : shape (N,)  or any shape
    #   depth   : scalar int64
    #   values  : [off_value, on_value]
    one_hot_values = np.array([0.0, 1.0], dtype=dtype)

    one_hot_cols: List[str] = []
    for j in range(n_features):
        # Extract column j: (N, F) → (N,) then (N, 1) for later concat
        col_j = g.op.Gather(
            bin_indices,
            np.array(j, dtype=np.int64),
            axis=1,
            name=f"{name}_gather_{j}",
        )  # (N,)
        depth_j = np.array(int(n_bins[j]), dtype=np.int64)
        oh_j = g.op.OneHot(
            col_j,
            depth_j,
            one_hot_values,
            axis=-1,
            name=f"{name}_onehot_{j}",
        )  # (N, n_bins[j])
        one_hot_cols.append(oh_j)

    if len(one_hot_cols) == 1:
        res = g.op.Identity(one_hot_cols[0], name=f"{name}_identity", outputs=outputs)
    else:
        res = g.op.Concat(
            *one_hot_cols, axis=1, name=f"{name}_concat", outputs=outputs
        )
    assert isinstance(res, str)
    if not sts:
        g.set_type(res, itype)
    return res
