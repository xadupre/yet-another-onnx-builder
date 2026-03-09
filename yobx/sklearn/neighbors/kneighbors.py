from typing import Any, Dict, List, Tuple, Union

import numpy as np
import onnx
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype

# Metric name aliases used to canonicalise the value of ``effective_metric_``.
_METRIC_ALIASES: Dict[str, str] = {
    "l1": "manhattan",
    "cityblock": "manhattan",
    "l2": "euclidean",
}

# Metrics that can be delegated to ``com.microsoft.CDist``.
# Only ``euclidean`` and ``sqeuclidean`` are universally supported across
# all ORT releases; ``cosine`` was added later so we keep the set conservative.
_CDIST_METRICS = frozenset({"euclidean", "sqeuclidean"})

# All metrics with a standard-ONNX implementation.
_STD_METRICS = frozenset(
    {"sqeuclidean", "euclidean", "cosine", "manhattan", "chebyshev", "minkowski"}
)


def _compute_pairwise_distances(
    g: GraphBuilderExtendedProtocol,
    X: str,
    training_data: np.ndarray,
    itype: int,
    metric: str,
    name: str,
    **metric_params: Any,
) -> str:
    """
    Computes pairwise distances between *X* (shape ``(N, F)``) and
    *training_data* (shape ``(M, F)``).

    The *metric* argument selects the distance function:

    * ``"sqeuclidean"`` — squared Euclidean ``||x−c||²`` (default sklearn
      default after effective-metric normalisation with p=2).
    * ``"euclidean"`` — Euclidean ``||x−c||``.
    * ``"cosine"`` — ``1 − (x·c)/(||x||·||c||)`` ∈ [0, 2].
    * ``"manhattan"`` (also ``"cityblock"``, ``"l1"``) — ``||x−c||₁``.
    * ``"chebyshev"`` — ``||x−c||∞ = max_k |x_k − c_k|``.
    * ``"minkowski"`` — ``||x−c||_p``; pass ``p=<value>`` in *metric_params*.
      ``p=1`` and ``p=2`` are redirected to the ``manhattan`` and ``euclidean``
      branches respectively; other values use a generic Pow/ReduceSum/Pow graph.
      sklearn always provides ``p`` via ``effective_metric_params_``; if absent
      the implementation defaults to ``p=2`` (Euclidean).

    When the ``com.microsoft`` domain is registered in the graph builder **and**
    the metric is in :data:`_CDIST_METRICS`, the computation is delegated to
    ``com.microsoft.CDist``, which is hardware-accelerated by ONNX Runtime.
    For all other metrics, or when the CDist domain is absent, a pure
    standard-ONNX implementation is used.

    Minimum opset requirements (standard-ONNX path):

    * All metrics except ``"chebyshev"``: opset ≥ 13
      (``ReduceSum`` with axes as input).
    * ``"chebyshev"``: opset ≥ 18
      (``ReduceMax`` with axes as input).

    :param g: graph builder
    :param X: input tensor name – shape ``(N, F)``
    :param training_data: training feature matrix – shape ``(M, F)``
    :param itype: ONNX element type of *X*
    :param metric: distance metric name (canonical or aliased)
    :param name: node name prefix
    :param metric_params: extra keyword arguments forwarded to the metric
        (e.g. ``p=3`` for ``"minkowski"``).
    :return: name of the ``(N, M)`` distance tensor
    :raises NotImplementedError: if *metric* is not supported or the opset
        is too low for the chosen implementation.
    """
    # Normalise metric name aliases upfront so all subsequent branches
    # (including recursive calls for minkowski) see the canonical name.
    metric = _METRIC_ALIASES.get(metric, metric)

    if metric not in _STD_METRICS:
        raise NotImplementedError(
            f"Metric {metric!r} is not supported by the KNN ONNX converter. "
            f"Supported metrics: {sorted(_STD_METRICS)}."
        )

    dtype = tensor_dtype_to_np_dtype(itype)
    training_data = training_data.astype(dtype)

    # ------------------------------------------------------------------ CDist
    if g.has_opset("com.microsoft") and metric in _CDIST_METRICS:
        training_data_name = g.make_initializer(
            f"{name}_training_data", training_data
        )
        dists = g.make_node(
            "CDist",
            [X, training_data_name],
            domain="com.microsoft",
            metric=metric,
            name=f"{name}_cdist",
        )
        zero = np.array([0], dtype=dtype)
        return g.op.Max(dists, zero, name=f"{name}_clip")

    # ---------------------------------------------------- Standard ONNX path
    opset = g.get_opset("")
    if opset < 13:
        raise NotImplementedError(
            f"The standard-ONNX KNN converter requires opset >= 13 "
            f"(ReduceSum with axes as input was added in opset 13), "
            f"but the graph builder has opset {opset}. "
            f"Pass target_opset >= 13 or include 'com.microsoft': 1 to use CDist."
        )

    if metric == "chebyshev" and opset < 18:
        raise NotImplementedError(
            f"The 'chebyshev' metric requires opset >= 18 "
            f"(ReduceMax with axes as input was added in opset 18), "
            f"but the graph builder has opset {opset}."
        )

    zero = np.array([0], dtype=dtype)

    if metric in ("sqeuclidean", "euclidean"):
        # Efficient O(N·F + M·F + N·M) implementation:
        # ||x − c||² = ||x||² − 2·x·cᵀ + ||c||²
        training_T = training_data.T.astype(dtype)  # (F, M)
        c_sq = np.sum(training_data**2, axis=1, keepdims=True).T.astype(dtype)  # (1, M)

        x_sq = g.op.Mul(X, X, name=f"{name}_x_sq")
        x_sq_sum = g.op.ReduceSum(
            x_sq,
            np.array([1], dtype=np.int64),
            keepdims=1,
            name=f"{name}_x_sq_sum",
        )  # (N, 1)
        cross = g.op.MatMul(X, training_T, name=f"{name}_cross")  # (N, M)
        two = np.array([2], dtype=dtype)
        two_cross = g.op.Mul(two, cross, name=f"{name}_two_cross")
        sq_plus = g.op.Add(x_sq_sum, c_sq, name=f"{name}_sq_plus")
        sq_dists = g.op.Sub(sq_plus, two_cross, name=f"{name}_sq_dists")
        sq_dists = g.op.Max(sq_dists, zero, name=f"{name}_sq_clip")
        if metric == "sqeuclidean":
            return sq_dists
        return g.op.Sqrt(sq_dists, name=f"{name}_sqrt")

    if metric == "cosine":
        # cosine_dist(x, c) = 1 − (x · c) / (||x|| · ||c||)
        norm_eps = np.array([1e-12], dtype=dtype)  # minimum norm to avoid division by zero
        x_sq = g.op.Mul(X, X, name=f"{name}_x_sq")
        x_sq_sum = g.op.ReduceSum(
            x_sq,
            np.array([1], dtype=np.int64),
            keepdims=1,
            name=f"{name}_x_sq_sum",
        )  # (N, 1)
        x_norm = g.op.Sqrt(
            g.op.Max(x_sq_sum, norm_eps, name=f"{name}_x_sq_clip"),
            name=f"{name}_x_norm",
        )  # (N, 1)
        x_normalized = g.op.Div(X, x_norm, name=f"{name}_x_normd")  # (N, F)

        # Normalise C as a constant (M, F) → (F, M) for matmul
        c_norm = np.maximum(
            np.linalg.norm(training_data, axis=1, keepdims=True), 1e-12
        )  # (M, 1)
        c_normalized_T = (training_data / c_norm).T.astype(dtype)  # (F, M)

        cos_sim = g.op.MatMul(x_normalized, c_normalized_T, name=f"{name}_cos_sim")
        one = np.array([1.0], dtype=dtype)
        cos_dist = g.op.Sub(one, cos_sim, name=f"{name}_cos_dist")
        # Cosine distance is theoretically in [0, 2]; clip for float safety.
        two = np.array([2.0], dtype=dtype)
        cos_dist = g.op.Min(
            g.op.Max(cos_dist, zero, name=f"{name}_cos_lo"),
            two,
            name=f"{name}_cos_hi",
        )
        return cos_dist

    if metric == "manhattan":
        # ||x − c||₁  — expand to (N, M, F), take abs, reduce-sum over features
        M, F = training_data.shape
        c_3d = training_data.reshape(1, M, F).astype(dtype)  # (1, M, F) constant
        x_3d = g.op.Unsqueeze(
            X, np.array([1], dtype=np.int64), name=f"{name}_unsq"
        )  # (N, 1, F)
        diff = g.op.Sub(x_3d, c_3d, name=f"{name}_diff")  # (N, M, F)
        abs_diff = g.op.Abs(diff, name=f"{name}_abs")  # (N, M, F)
        return g.op.ReduceSum(
            abs_diff,
            np.array([2], dtype=np.int64),
            keepdims=0,
            name=f"{name}_l1",
        )  # (N, M)

    if metric == "chebyshev":
        # ||x − c||∞ = max_k |x_k − c_k|  — requires opset >= 18 for ReduceMax
        M, F = training_data.shape
        c_3d = training_data.reshape(1, M, F).astype(dtype)  # (1, M, F)
        x_3d = g.op.Unsqueeze(
            X, np.array([1], dtype=np.int64), name=f"{name}_unsq"
        )  # (N, 1, F)
        diff = g.op.Sub(x_3d, c_3d, name=f"{name}_diff")  # (N, M, F)
        abs_diff = g.op.Abs(diff, name=f"{name}_abs")  # (N, M, F)
        return g.op.ReduceMax(
            abs_diff,
            np.array([2], dtype=np.int64),
            keepdims=0,
            name=f"{name}_linf",
        )  # (N, M)

    if metric == "minkowski":
        # sklearn always provides 'p' in effective_metric_params_ for minkowski;
        # default to p=2 (euclidean) only as a safety fallback.
        p = float(metric_params.get("p", 2))
        if p == 2:
            # Use the efficient euclidean branch (no extra recursion since
            # metric has already been alias-resolved above).
            metric = "euclidean"
            return _compute_pairwise_distances(
                g, X, training_data, itype, metric, name
            )
        if p == 1:
            metric = "manhattan"
            return _compute_pairwise_distances(
                g, X, training_data, itype, metric, name
            )
        # General p: (sum |x_k - c_k|^p)^(1/p)
        M, F = training_data.shape
        c_3d = training_data.reshape(1, M, F).astype(dtype)
        x_3d = g.op.Unsqueeze(
            X, np.array([1], dtype=np.int64), name=f"{name}_unsq"
        )
        diff = g.op.Sub(x_3d, c_3d, name=f"{name}_diff")
        abs_diff = g.op.Abs(diff, name=f"{name}_abs")
        p_arr = np.array([p], dtype=dtype)
        inv_p_arr = np.array([1.0 / p], dtype=dtype)
        powered = g.op.Pow(abs_diff, p_arr, name=f"{name}_pow")
        sum_pow = g.op.ReduceSum(
            powered,
            np.array([2], dtype=np.int64),
            keepdims=0,
            name=f"{name}_sum",
        )
        return g.op.Pow(sum_pow, inv_p_arr, name=f"{name}_root")

    # Should not be reached.
    raise NotImplementedError(f"Metric {metric!r} is not implemented.")


@register_sklearn_converter(KNeighborsClassifier)
def sklearn_knn_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: KNeighborsClassifier,
    X: str,
    name: str = "knn_clf",
) -> Union[str, Tuple[str, str]]:
    """
    Converts a :class:`sklearn.neighbors.KNeighborsClassifier` into ONNX.

    The converter supports all metrics implemented by
    :func:`_compute_pairwise_distances`.  The effective metric (and any extra
    parameters such as the Minkowski exponent ``p``) are read from
    ``estimator.effective_metric_`` and ``estimator.effective_metric_params_``.

    Supported metrics: ``"sqeuclidean"``, ``"euclidean"``, ``"cosine"``,
    ``"manhattan"`` (aliases: ``"cityblock"``, ``"l1"``), ``"chebyshev"``,
    ``"minkowski"``.  The ``"euclidean"`` and ``"sqeuclidean"`` metrics use
    ``com.microsoft.CDist`` when that domain is registered; all other metrics
    use the standard-ONNX path.

    Full graph structure (standard-ONNX path):

    .. code-block:: text

        X (N, F)
          │
          └─── pairwise distances ────────────────────────────────────► dists (N, M)
                                                                               │
                                                     TopK(k, axis=1, largest=0) ──► indices (N, k)
                                                                               │
                                        Gather(training_labels_encoded) ──────► neighbor_labels (N, k)
                                                                               │
                                Reshape(-1) → OneHot(n_classes) → Reshape(N, k, n_classes) ──►
                                                                               │
                                                           ReduceSum(axis=1) ──► votes (N, n_classes)
                                                                               │
                                             ArgMax(axis=1) → Gather(classes_) ──► labels (N,)
                                                                               │
                                           Div(votes, ReduceSum(votes, axis=1)) ──► probabilities (N, n_classes)

    :param g: graph builder
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names; ``outputs[0]`` receives the predicted
        labels and ``outputs[1]`` (if present) receives the class probabilities
    :param estimator: a fitted ``KNeighborsClassifier``
    :param X: input tensor name
    :param name: prefix names for the added nodes
    :return: predicted label tensor (and optionally probability tensor as second output)
    """
    assert isinstance(estimator, KNeighborsClassifier)
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    # ReduceSum with axes-as-input requires opset >= 13.
    opset = g.get_opset("")
    if opset < 13:
        raise NotImplementedError(
            f"KNeighborsClassifier converter requires opset >= 13 "
            f"(ReduceSum with axes as input was added in opset 13), "
            f"but the graph builder has opset {opset}."
        )

    # In older sklearn, outputs_2d_ exists; for newer sklearn it may not.
    # _y is 1-D for a single-output estimator and 2-D for multi-output.
    raw_y = estimator._y
    assert raw_y.ndim == 1 or raw_y.shape[1] == 1, (
        "Multi-output KNeighborsClassifier is not yet supported."
    )

    itype = g.get_type(X)
    k = estimator.n_neighbors
    training_data = estimator._fit_X  # (M, F)
    metric = estimator.effective_metric_
    metric_params = estimator.effective_metric_params_

    training_labels_encoded = (
        raw_y.ravel() if raw_y.ndim == 1 else raw_y[:, 0]
    ).astype(np.int64)  # (M,)

    # classes_ may be a plain ndarray (single output) or a list of arrays
    # (multi-output).  Normalise to a plain ndarray.
    classes_raw = estimator.classes_
    classes_arr = classes_raw[0] if isinstance(classes_raw, list) else classes_raw
    n_classes = len(classes_arr)

    # 1. Pairwise distances: (N, M)
    dists = _compute_pairwise_distances(
        g, X, training_data, itype, metric, f"{name}_dist", **metric_params
    )

    # 2. k nearest neighbours – values (N, k) and indices (N, k)
    _topk_values, nn_indices = g.op.TopK(
        dists,
        np.array([k], dtype=np.int64),
        axis=1,
        largest=0,
        sorted=1,
        name=f"{name}_topk",
    )

    # 3. Gather encoded labels for k neighbours: (N, k)
    neighbor_labels = g.op.Gather(
        training_labels_encoded,
        nn_indices,
        axis=0,
        name=f"{name}_gather",
    )

    # 4. Flatten to (N*k,) then one-hot encode to (N*k, n_classes)
    flat_labels = g.op.Reshape(
        neighbor_labels,
        np.array([-1], dtype=np.int64),
        name=f"{name}_flat",
    )
    one_hot = g.op.OneHot(
        flat_labels,
        np.array(n_classes, dtype=np.int64),
        np.array([0, 1], dtype=np.float32),
        axis=1,
        name=f"{name}_onehot",
    )  # (N*k, n_classes)

    # 5. Reshape to (N, k, n_classes) and sum votes: (N, n_classes)
    one_hot_3d = g.op.Reshape(
        one_hot,
        np.array([-1, k, n_classes], dtype=np.int64),
        name=f"{name}_3d",
    )
    vote_counts = g.op.ReduceSum(
        one_hot_3d,
        np.array([1], dtype=np.int64),
        keepdims=0,
        name=f"{name}_votes",
    )  # (N, n_classes)
    # Ensure float type is propagated for downstream Div node.
    assert isinstance(vote_counts, str)
    g.set_type(vote_counts, onnx.TensorProto.FLOAT)

    # 6. Predicted class index: (N,) int64
    class_idx = g.op.ArgMax(
        vote_counts,
        axis=1,
        keepdims=0,
        name=f"{name}_argmax",
    )

    # 7. Map back to the actual class labels: (N,)
    if np.issubdtype(classes_arr.dtype, np.integer):
        classes_init = classes_arr.astype(np.int64)
    else:
        classes_init = classes_arr

    labels = g.op.Gather(
        classes_init,
        class_idx,
        axis=0,
        name=f"{name}_labels",
        outputs=outputs[:1],
    )
    assert isinstance(labels, str)
    if not sts:
        out_itype = (
            onnx.TensorProto.INT64
            if np.issubdtype(classes_arr.dtype, np.integer)
            else onnx.TensorProto.STRING
        )
        g.set_type(labels, out_itype)

    n_out = len(outputs)
    if n_out >= 2:
        # Probabilities: normalise vote counts → (N, n_classes) in [0, 1]
        total = g.op.ReduceSum(
            vote_counts,
            np.array([1], dtype=np.int64),
            keepdims=1,
            name=f"{name}_total",
        )  # (N, 1)
        probabilities = g.op.Div(
            vote_counts,
            total,
            name=f"{name}_proba",
            outputs=outputs[1:2],
        )
        assert isinstance(probabilities, str)
        if not sts:
            g.set_type(probabilities, onnx.TensorProto.FLOAT)
        return labels, probabilities

    return labels


@register_sklearn_converter(KNeighborsRegressor)
def sklearn_knn_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: KNeighborsRegressor,
    X: str,
    name: str = "knn_reg",
) -> str:
    """
    Converts a :class:`sklearn.neighbors.KNeighborsRegressor` into ONNX.

    The converter supports all metrics implemented by
    :func:`_compute_pairwise_distances`.  The effective metric (and any extra
    parameters such as the Minkowski exponent ``p``) are read from
    ``estimator.effective_metric_`` and ``estimator.effective_metric_params_``.

    Supported metrics: ``"sqeuclidean"``, ``"euclidean"``, ``"cosine"``,
    ``"manhattan"`` (aliases: ``"cityblock"``, ``"l1"``), ``"chebyshev"``,
    ``"minkowski"``.  The ``"euclidean"`` and ``"sqeuclidean"`` metrics use
    ``com.microsoft.CDist`` when that domain is registered; all other metrics
    use the standard-ONNX path.

    :param g: graph builder
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a fitted ``KNeighborsRegressor``
    :param X: input tensor name
    :param name: prefix names for the added nodes
    :return: predicted value tensor
    """
    assert isinstance(estimator, KNeighborsRegressor)
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    # ReduceMean with axes-as-input requires opset >= 18.
    opset = g.get_opset("")
    if opset < 18:
        raise NotImplementedError(
            f"KNeighborsRegressor converter requires opset >= 18 "
            f"(ReduceMean with axes as input was added in opset 18), "
            f"but the graph builder has opset {opset}."
        )

    # _y is 1-D for a single-output estimator and 2-D for multi-output.
    raw_y = estimator._y
    assert raw_y.ndim == 1 or raw_y.shape[1] == 1, (
        "Multi-output KNeighborsRegressor is not yet supported."
    )

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)
    k = estimator.n_neighbors
    training_data = estimator._fit_X  # (M, F)
    metric = estimator.effective_metric_
    metric_params = estimator.effective_metric_params_

    training_targets = (
        raw_y.ravel() if raw_y.ndim == 1 else raw_y[:, 0]
    ).astype(dtype)  # (M,)

    # 1. Pairwise distances: (N, M)
    dists = _compute_pairwise_distances(
        g, X, training_data, itype, metric, f"{name}_dist", **metric_params
    )

    # 2. k nearest neighbour indices: (N, k)
    _topk_values, nn_indices = g.op.TopK(
        dists,
        np.array([k], dtype=np.int64),
        axis=1,
        largest=0,
        sorted=1,
        name=f"{name}_topk",
    )

    # 3. Gather regression targets for k neighbours: (N, k)
    neighbor_targets = g.op.Gather(
        training_targets,
        nn_indices,
        axis=0,
        name=f"{name}_gather",
    )

    # 4. Average over k neighbours: (N,)
    predictions = g.op.ReduceMean(
        neighbor_targets,
        np.array([1], dtype=np.int64),
        keepdims=0,
        name=f"{name}_mean",
        outputs=outputs[:1],
    )
    assert isinstance(predictions, str)
    if not sts:
        g.set_type(predictions, itype)

    return predictions
