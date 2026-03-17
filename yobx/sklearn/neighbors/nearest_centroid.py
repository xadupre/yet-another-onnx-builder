from typing import Dict, List, Tuple, Union

import numpy as np
import onnx
from sklearn.neighbors import NearestCentroid

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from .kneighbors import _compute_pairwise_distances


def _compute_discriminant_scores(
    g: GraphBuilderExtendedProtocol,
    X: str,
    centroids: np.ndarray,
    within_std: np.ndarray,
    class_prior: np.ndarray,
    itype: int,
    metric: str,
    name: str,
) -> str:
    """
    Computes the discriminant scores used by
    :meth:`~sklearn.neighbors.NearestCentroid._decision_function`.

    For class ``c`` the score is:

    .. code-block:: text

        score_c = -dist(X_norm, centroid_norm_c)² + 2 * log(prior_c)

    where ``X_norm`` and ``centroid_norm_c`` are divided element-wise by
    ``within_class_std_dev_`` for features where the standard deviation is
    non-zero.

    :param g: graph builder
    :param X: input tensor name – shape ``(N, F)``
    :param centroids: fitted centroid matrix – shape ``(C, F)``
    :param within_std: pooled within-class standard deviation – shape ``(F,)``
    :param class_prior: class prior probabilities – shape ``(C,)``
    :param itype: ONNX element type of *X*
    :param metric: distance metric (``"euclidean"`` or ``"manhattan"``)
    :param name: node name prefix
    :return: discriminant scores tensor – shape ``(N, C)``
    """
    dtype = tensor_dtype_to_np_dtype(itype)
    mask = within_std != 0

    if mask.any():
        # Divide by std where std != 0; leave other features unchanged.
        norm_arr = np.where(mask, within_std, np.ones_like(within_std)).astype(dtype)
        X_norm = g.op.Div(X, norm_arr, name=f"{name}_x_norm")
        centroids_norm = (centroids / norm_arr[np.newaxis, :]).astype(dtype)
    else:
        X_norm = X
        centroids_norm = centroids.astype(dtype)

    # Pairwise distances on normalised data, then square them: (N, C)
    dists = _compute_pairwise_distances(
        g, X_norm, centroids_norm, itype, metric, f"{name}_dist"
    )
    sq_dists = g.op.Mul(dists, dists, name=f"{name}_sq_dists")

    # Discriminant score: -sq_dist + 2 * log(prior)  → (N, C)
    log_prior = (2.0 * np.log(class_prior)).astype(dtype)  # (C,)
    neg_sq = g.op.Neg(sq_dists, name=f"{name}_neg_sq")
    return g.op.Add(neg_sq, log_prior, name=f"{name}_scores")


@register_sklearn_converter(NearestCentroid)
def sklearn_nearest_centroid(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: NearestCentroid,
    X: str,
    name: str = "nearest_centroid",
) -> Union[str, Tuple[str, str]]:
    """
    Converts a :class:`sklearn.neighbors.NearestCentroid` into ONNX.

    Reproduces both :meth:`~sklearn.neighbors.NearestCentroid.predict` and
    :meth:`~sklearn.neighbors.NearestCentroid.predict_proba`.

    **Uniform-prior labels path** (all ``class_prior_`` values are equal):

    sklearn assigns each sample to the nearest centroid using raw pairwise
    distances (no feature normalisation):

    .. code-block:: text

        X (N, F)
          │
          └─── pairwise distances ──────────────────────────────────► dists (N, C)
                                                                               │
                                              ArgMin(axis=1) ──────────────► idx (N,)
                                                                               │
                               Gather(classes_) ──────────────────────────► label (N,)

    **Non-uniform-prior labels path**:

    The discriminant score for class ``c`` (eq. 18.2, ESL 2nd ed.) is:

    .. code-block:: text

        score_c = -dist(X_norm, centroid_norm_c)² + 2 * log(prior_c)

    where ``X_norm`` and ``centroid_norm_c`` are divided element-wise by
    ``within_class_std_dev_`` (features with zero std are left unchanged):

    .. code-block:: text

        X (N, F)
          │
          ├── Div(within_class_std_dev_) ──────────────────────────► X_norm (N, F)
          │                                                                   │
          │                               pairwise distances ──────────────► dists (N, C)
          │                                                                   │
          │                                       Mul(dists, dists) ────────► sq_dists (N, C)
          │                                                                   │
          │                 -sq_dists + 2*log(class_prior_) ───────────────► scores (N, C)
          │                                                                   │
          └───────────────────────────────────── ArgMax(axis=1) ────────────► idx (N,)
                                                                               │
                               Gather(classes_) ──────────────────────────► label (N,)

    **Probabilities path** (always uses discriminant scores):

    sklearn's :meth:`~sklearn.neighbors.NearestCentroid.predict_proba`
    always goes through ``_decision_function``, which applies the discriminant
    score formulation above regardless of prior uniformity.  The probabilities
    are the softmax of those scores:

    .. code-block:: text

        scores (N, C)
          │
          ├── ReduceMax(axis=1, keepdims=1) ────────────────────────► max_s (N, 1)
          │                                                                   │
          ├── Sub(max_s) ──────────────────────────────────────────► shifted (N, C)
          │                                                                   │
          ├── Exp ────────────────────────────────────────────────► exp_s (N, C)
          │                                                                   │
          └── Div(ReduceSum(exp_s, axis=1, keepdims=1)) ──────────► proba (N, C)

    Supported metrics: ``"euclidean"`` and ``"manhattan"``.

    :param g: graph builder
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names; ``outputs[0]`` receives the predicted
        labels and ``outputs[1]`` (if present) receives the class probabilities
    :param estimator: a fitted ``NearestCentroid``
    :param X: input tensor name
    :param name: prefix names for the added nodes
    :return: predicted label tensor (and optionally probability tensor as second output)
    """
    assert isinstance(estimator, NearestCentroid)
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    centroids = estimator.centroids_.astype(dtype)  # (C, F)
    classes_arr = estimator.classes_
    class_prior = estimator.class_prior_  # (C,)
    within_std = estimator.within_class_std_dev_.astype(dtype)  # (F,)

    if np.issubdtype(classes_arr.dtype, np.integer):
        classes_init = classes_arr.astype(np.int64)
    else:
        classes_init = classes_arr

    uniform_prior = np.allclose(class_prior, class_prior[0])
    metric = estimator.metric  # "euclidean" or "manhattan"
    n_out = len(outputs)

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    if uniform_prior:
        # sklearn uses pairwise_distances_argmin(X, centroids_, metric=metric)
        # – no normalisation.
        raw_dists = _compute_pairwise_distances(
            g, X, centroids, itype, metric, f"{name}_raw_dist"
        )  # (N, C)
        class_idx = g.op.ArgMin(raw_dists, axis=1, keepdims=0, name=f"{name}_argmin")
    else:
        # sklearn uses _decision_function → argmax of discriminant scores.
        scores_label = _compute_discriminant_scores(
            g, X, centroids, within_std, class_prior, itype, metric, f"{name}_lbl"
        )
        class_idx = g.op.ArgMax(scores_label, axis=1, keepdims=0, name=f"{name}_argmax")

    labels = g.op.Gather(
        classes_init, class_idx, axis=0, name=f"{name}_labels", outputs=outputs[:1]
    )
    assert isinstance(labels, str)
    if not sts:
        out_itype = (
            onnx.TensorProto.INT64
            if np.issubdtype(classes_arr.dtype, np.integer)
            else onnx.TensorProto.STRING
        )
        g.set_type(labels, out_itype)

    if n_out < 2:
        return labels

    # ------------------------------------------------------------------
    # Probabilities – always uses discriminant scores (as predict_proba does)
    # ------------------------------------------------------------------
    scores = _compute_discriminant_scores(
        g, X, centroids, within_std, class_prior, itype, metric, f"{name}_proba"
    )

    # Numerically-stable softmax: proba = exp(s - max(s)) / sum(exp(s - max(s)))
    s_max = g.op.ReduceMax(
        scores, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_smax"
    )
    s_shifted = g.op.Sub(scores, s_max, name=f"{name}_sshift")
    s_exp = g.op.Exp(s_shifted, name=f"{name}_sexp")
    s_sum = g.op.ReduceSum(
        s_exp, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_ssum"
    )
    probabilities = g.op.Div(
        s_exp, s_sum, name=f"{name}_proba_out", outputs=outputs[1:2]
    )
    assert isinstance(probabilities, str)
    if not sts:
        g.set_type(probabilities, itype)

    return labels, probabilities
