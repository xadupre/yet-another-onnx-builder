from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from .kneighbors import _compute_pairwise_distances


@register_sklearn_converter(LocalOutlierFactor)
def sklearn_local_outlier_factor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: LocalOutlierFactor,
    X: str,
    name: str = "lof",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.neighbors.LocalOutlierFactor` into ONNX.

    Only ``novelty=True`` is supported (novelty detection mode), which enables
    the :meth:`~sklearn.neighbors.LocalOutlierFactor.predict` and
    :meth:`~sklearn.neighbors.LocalOutlierFactor.score_samples` methods on new
    data.

    **Algorithm overview**

    For each query point *x*, the converter implements the exact LOF formula:

    1. Compute pairwise distances to all training points: ``dists`` (N, M).
    2. Find the *k* nearest training neighbours: ``topk_dists``, ``topk_idx``.
    3. Compute the *reachability distance* from *x* to each neighbour *x_i*::

           reach_dist(x, x_i) = max(dist(x, x_i), k_distance(x_i))

       where ``k_distance(x_i)`` is the distance from training point *x_i* to
       its own *k*-th nearest neighbour, precomputed as
       ``estimator._distances_fit_X_[:, n_neighbors_ - 1]``.

    4. Local Reachability Density of *x*::

           LRD(x) = 1 / (mean(reach_dist(x, x_i)) + 1e-10)

    5. LOF score::

           LOF(x) = mean(LRD(x_i)) / LRD(x)

    6. Anomaly score (``score_samples``)::

           score_samples(x) = -LOF(x)

    7. Decision function and label::

           decision_function(x) = score_samples(x) - offset_
           predict(x) = 1  if decision_function(x) >= 0 else -1

    **ONNX graph structure**

    .. code-block:: text

        X (N, F)
          │
          └─── pairwise distances ────────────────────────────────► dists (N, M)
                                                                          │
                       TopK(k, axis=1, largest=0) ──────────────► topk_dists (N,k), topk_idx (N,k)
                                                                          │
            Gather(k_distances_train, topk_idx) ───────────────► k_dists_nbrs (N, k)
                                                                          │
            Max(topk_dists, k_dists_nbrs) ─────────────────────► reach_dists (N, k)
                                                                          │
            ReduceMean(axis=1) + 1e-10 ─────────────────────────► mean_reach (N,)
                                                                          │
            Div(1, mean_reach) ─────────────────────────────────► lrd_query (N,)
                                                                          │
            Gather(lrd_train, topk_idx) ────────────────────────► lrd_nbrs (N, k)
                                                                          │
            Div(lrd_nbrs, Unsqueeze(lrd_query,1)) ──────────────► lrd_ratios (N, k)
                                                                          │
            Neg(ReduceMean(lrd_ratios, axis=1)) ────────────────► score_samples (N,)
                                                                          │
            Sub(score_samples, offset_) ────────────────────────► decision (N,)
                                                                          │
            Where(decision >= 0, 1, -1) ────────────────────────► label (N,)

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names; ``outputs[0]`` receives the
        predicted labels and ``outputs[1]`` the anomaly scores
    :param estimator: a fitted :class:`~sklearn.neighbors.LocalOutlierFactor`
        with ``novelty=True``
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: tuple ``(label, scores)``
    :raises ValueError: if ``estimator.novelty`` is ``False``
    :raises NotImplementedError: if the opset is below 18 (required for
        ``ReduceMean`` with axes as input) or the metric is unsupported
    """
    assert isinstance(
        estimator, LocalOutlierFactor
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    if not estimator.novelty:
        raise ValueError(
            "LocalOutlierFactor ONNX conversion only supports novelty=True. "
            "Set novelty=True when creating the estimator so that predict() "
            "and score_samples() are available for new data."
        )

    opset = g.get_opset("")
    if opset < 18:
        raise NotImplementedError(
            f"LocalOutlierFactor converter requires opset >= 18 "
            f"(ReduceMean with axes as input was added in opset 18), "
            f"but the graph builder has opset {opset}."
        )

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    k = int(estimator.n_neighbors_)
    offset = float(estimator.offset_)
    metric = estimator.effective_metric_
    metric_params = dict(estimator.effective_metric_params_)

    fit_X = estimator._fit_X.astype(dtype)
    # k-distance for each training point: distance to its k-th nearest neighbour
    k_distances_train = estimator._distances_fit_X_[:, k - 1].astype(dtype)  # (M,)
    lrd_train = estimator._lrd.astype(dtype)  # (M,)

    # 1. Pairwise distances between query points X and training data: (N, M)
    dists = _compute_pairwise_distances(
        g, X, fit_X, itype, metric, f"{name}_dist", **metric_params
    )

    # 2. k nearest neighbours: topk_dists (N, k), topk_idx (N, k)
    topk_dists, topk_idx = g.op.TopK(
        dists, np.array([k], dtype=np.int64), axis=1, largest=0, sorted=1, name=f"{name}_topk"
    )

    # 3. k-distances of the neighbours (from their own fit): (N, k)
    k_dists_init = g.make_initializer(f"{name}_k_distances_train", k_distances_train)
    k_dists_nbrs = g.op.Gather(k_dists_init, topk_idx, axis=0, name=f"{name}_k_dists_nbrs")

    # 4. Reachability distances: max(dist(x, x_i), k_distance(x_i)): (N, k)
    reach_dists = g.op.Max(topk_dists, k_dists_nbrs, name=f"{name}_reach_dists")

    # 5. Mean reachability distance + epsilon: (N,)
    mean_reach = g.op.ReduceMean(
        reach_dists, np.array([1], dtype=np.int64), keepdims=0, name=f"{name}_mean_reach"
    )
    eps = np.array([1e-10], dtype=dtype)
    mean_reach_eps = g.op.Add(mean_reach, eps, name=f"{name}_mean_reach_eps")

    # 6. LRD of query point: 1 / mean_reach: (N,)
    ones = np.array([1.0], dtype=dtype)
    lrd_query = g.op.Div(ones, mean_reach_eps, name=f"{name}_lrd_query")

    # 7. LRD values of the k neighbours: (N, k)
    lrd_init = g.make_initializer(f"{name}_lrd_train", lrd_train)
    lrd_nbrs = g.op.Gather(lrd_init, topk_idx, axis=0, name=f"{name}_lrd_nbrs")

    # 8. LRD ratios: lrd(x_i) / lrd(x): (N, k)
    lrd_query_2d = g.op.Unsqueeze(
        lrd_query, np.array([1], dtype=np.int64), name=f"{name}_lrd_query_2d"
    )
    lrd_ratios = g.op.Div(lrd_nbrs, lrd_query_2d, name=f"{name}_lrd_ratios")

    # 9. score_samples = -mean(lrd_ratios): (N,)
    mean_ratio = g.op.ReduceMean(
        lrd_ratios, np.array([1], dtype=np.int64), keepdims=0, name=f"{name}_mean_ratio"
    )
    score_samples = g.op.Neg(mean_ratio, name=f"{name}_score_samples")

    # 10. decision_function = score_samples - offset_: (N,)
    offset_arr = np.array([offset], dtype=dtype)
    decision = g.op.Sub(score_samples, offset_arr, name=f"{name}_decision")

    # 11. predict: 1 if decision >= 0 else -1: (N,)
    zero = np.array([0.0], dtype=dtype)
    is_inlier = g.op.GreaterOrEqual(decision, zero, name=f"{name}_ge")
    label = g.op.Where(
        is_inlier,
        np.array([1], dtype=np.int64),
        np.array([-1], dtype=np.int64),
        name=f"{name}_label",
        outputs=outputs[:1],
    )

    scores_out = g.op.Identity(decision, name=f"{name}_scores", outputs=outputs[1:2])
    return label, scores_out
