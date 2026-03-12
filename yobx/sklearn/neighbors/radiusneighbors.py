from typing import Dict, List, Tuple, Union

import numpy as np
import onnx
from sklearn.neighbors import RadiusNeighborsClassifier, RadiusNeighborsRegressor

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from .kneighbors import _compute_pairwise_distances


@register_sklearn_converter(RadiusNeighborsClassifier)
def sklearn_radius_neighbors_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: RadiusNeighborsClassifier,
    X: str,
    name: str = "rnn_clf",
) -> Union[str, Tuple[str, str]]:
    """
    Converts a :class:`sklearn.neighbors.RadiusNeighborsClassifier` into ONNX.

    Only ``weights='uniform'`` is supported.  The converter produces a static
    ONNX graph by including the training data as graph initializers and
    evaluating classification for all training points simultaneously.

    Supported metrics are the same as for
    :func:`yobx.sklearn.neighbors.kneighbors.sklearn_knn_classifier` (see
    :func:`yobx.sklearn.neighbors.kneighbors._compute_pairwise_distances`).

    Full graph structure (standard-ONNX path):

    .. code-block:: text

        X (N, F)
          │
          └─── pairwise distances ──────────────────────────────────────► dists (N, M)
                                                                                │
                                           in_radius = (dists <= radius) ──► mask (N, M) bool
                                                                                │
                            Cast(float32) ──► mask_f32 (N, M) float32          │
                                                                                │
              OneHot(training_labels, n_classes) ──────────────────────► oh (M, n_classes)
                                                                                │
                    Unsqueeze(axis=2) → mask_3d (N, M, 1)                      │
                                   ╲                                            │
                              Mul ─────────────────────────────────► votes_3d (N, M, n_classes)
                                                                                │
                                          ReduceSum(axis=1) ──────► votes (N, n_classes)
                                                                                │
                                ArgMax(axis=1) → Gather(classes_) ──► labels (N,)
                                                                                │
                 Div(votes, Max(ReduceSum(votes, axis=1), 1)) ──────► proba (N, n_classes)

    For query points with no neighbor within the radius (outlier points):

    * If ``estimator.outlier_label_`` is set, a ``Where`` node substitutes the
      outlier label for those points.
    * If ``estimator.outlier_label_`` is ``None``, the output for outlier
      points is undefined (``ArgMax`` returns the first class when all votes
      are zero).

    :param g: graph builder
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names; ``outputs[0]`` receives the predicted
        labels and ``outputs[1]`` (if present) receives the class probabilities.
        Probabilities for outlier points are clamped to uniform (all zeros /
        total 0 → total is clipped to 1 before division).
    :param estimator: a fitted ``RadiusNeighborsClassifier``
    :param X: input tensor name
    :param name: prefix for node names
    :return: predicted label tensor (and optionally probability tensor)
    :raises NotImplementedError: if opset < 13 or ``weights != 'uniform'``
    """
    assert isinstance(estimator, RadiusNeighborsClassifier)
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    opset = g.get_opset("")
    if opset < 13:
        raise NotImplementedError(
            f"RadiusNeighborsClassifier converter requires opset >= 13 "
            f"(ReduceSum with axes as input was added in opset 13), "
            f"but the graph builder has opset {opset}."
        )

    if estimator.weights != "uniform":
        raise NotImplementedError(
            f"Only 'uniform' weights are supported by the "
            f"RadiusNeighborsClassifier ONNX converter; "
            f"got {estimator.weights!r}."
        )

    raw_y = estimator._y
    assert (
        raw_y.ndim == 1 or raw_y.shape[1] == 1
    ), "Multi-output RadiusNeighborsClassifier is not yet supported."

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)
    radius = float(estimator.radius)
    training_data = estimator._fit_X  # (M, F)
    metric = estimator.effective_metric_
    metric_params = estimator.effective_metric_params_

    training_labels_encoded = (raw_y.ravel() if raw_y.ndim == 1 else raw_y[:, 0]).astype(
        np.int64
    )  # (M,)

    classes_raw = estimator.classes_
    classes_arr = classes_raw[0] if isinstance(classes_raw, list) else classes_raw
    n_classes = len(classes_arr)

    # 1. Pairwise distances: (N, M)
    dists = _compute_pairwise_distances(
        g, X, training_data, itype, metric, f"{name}_dist", **metric_params
    )

    # 2. Radius mask: (N, M) bool — True where dist <= radius
    radius_const = np.array([radius], dtype=dtype)
    not_in_radius = g.op.Greater(dists, radius_const, name=f"{name}_gt")
    in_radius = g.op.Not(not_in_radius, name=f"{name}_mask")  # (N, M) bool

    # 3. Cast mask to float32 for voting (one-hot values are always float32)
    mask_f32 = g.op.Cast(
        in_radius, to=onnx.TensorProto.FLOAT, name=f"{name}_mask_f32"
    )  # (N, M) float32

    # 4. One-hot encode training labels: (M, n_classes) float32
    one_hot = g.op.OneHot(
        training_labels_encoded,
        np.array(n_classes, dtype=np.int64),
        np.array([0, 1], dtype=np.float32),
        axis=1,
        name=f"{name}_onehot",
    )  # (M, n_classes)

    # 5. Expand mask to (N, M, 1) and multiply with one_hot (M, n_classes):
    #    (N, M, 1) * (M, n_classes) broadcasts to (N, M, n_classes)
    mask_3d = g.op.Unsqueeze(
        mask_f32, np.array([2], dtype=np.int64), name=f"{name}_mask3d"
    )  # (N, M, 1)
    votes_3d = g.op.Mul(mask_3d, one_hot, name=f"{name}_votes3d")  # (N, M, n_classes)

    # 6. Sum votes over the neighbours axis: (N, n_classes) float32
    vote_counts = g.op.ReduceSum(
        votes_3d,
        np.array([1], dtype=np.int64),
        keepdims=0,
        name=f"{name}_votes",
    )  # (N, n_classes)
    assert isinstance(vote_counts, str)
    g.set_type(vote_counts, onnx.TensorProto.FLOAT)

    # 7. Predicted class index: (N,) int64
    class_idx = g.op.ArgMax(vote_counts, axis=1, keepdims=0, name=f"{name}_argmax")

    # 8. Map back to actual class labels: (N,)
    if np.issubdtype(classes_arr.dtype, np.integer):
        classes_init = classes_arr.astype(np.int64)
    else:
        classes_init = classes_arr

    predicted_label = g.op.Gather(classes_init, class_idx, axis=0, name=f"{name}_pred")

    # 9. Handle outlier points (no neighbour within radius)
    #    outlier_label_ is a list [label] or None when outlier_label=None.
    outlier_label_list = getattr(estimator, "outlier_label_", None)

    if outlier_label_list is not None:
        outlier_label_ = outlier_label_list[0]
        # Count neighbours per query point: (N,) float32
        neighbor_count = g.op.ReduceSum(
            mask_f32,
            np.array([1], dtype=np.int64),
            keepdims=0,
            name=f"{name}_count",
        )
        # is_outlier: True where count < 0.5 (i.e. == 0 for integer counts)
        half = np.array([0.5], dtype=np.float32)
        is_outlier = g.op.Less(neighbor_count, half, name=f"{name}_is_outlier")

        if np.issubdtype(classes_arr.dtype, np.integer):
            outlier_val = np.array([int(outlier_label_)], dtype=np.int64)
        else:
            outlier_val = np.array([str(outlier_label_)])

        labels = g.op.Where(
            is_outlier,
            outlier_val,
            predicted_label,
            name=f"{name}_labels",
            outputs=outputs[:1],
        )
    else:
        labels = g.op.Identity(predicted_label, name=f"{name}_labels", outputs=outputs[:1])

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
        # Normalise vote counts → probabilities in [0, 1]
        total = g.op.ReduceSum(
            vote_counts,
            np.array([1], dtype=np.int64),
            keepdims=1,
            name=f"{name}_total",
        )  # (N, 1) float32
        # Clamp denominator to 1 to avoid NaN for outlier points (total == 0)
        safe_total = g.op.Max(total, np.array([1.0], dtype=np.float32), name=f"{name}_safe_total")
        probabilities = g.op.Div(
            vote_counts,
            safe_total,
            name=f"{name}_proba",
            outputs=outputs[1:2],
        )
        assert isinstance(probabilities, str)
        if not sts:
            g.set_type(probabilities, onnx.TensorProto.FLOAT)
        return labels, probabilities

    return labels


@register_sklearn_converter(RadiusNeighborsRegressor)
def sklearn_radius_neighbors_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: RadiusNeighborsRegressor,
    X: str,
    name: str = "rnn_reg",
) -> str:
    """
    Converts a :class:`sklearn.neighbors.RadiusNeighborsRegressor` into ONNX.

    Only ``weights='uniform'`` is supported.  The converter produces a static
    ONNX graph by including the training data as graph initializers.

    Supported metrics are the same as for
    :func:`yobx.sklearn.neighbors.kneighbors.sklearn_knn_regressor` (see
    :func:`yobx.sklearn.neighbors.kneighbors._compute_pairwise_distances`).

    Full graph structure (standard-ONNX path):

    .. code-block:: text

        X (N, F)
          │
          └─── pairwise distances ──────────────────────────────────────► dists (N, M)
                                                                                │
                                           in_radius = (dists <= radius) ──► mask (N, M) bool
                                                                                │
                          Cast(itype) ──► mask_float (N, M)                    │
                                                                                │
              training_targets (M,) ─────────────────────────────────────────► │
                                   Mul (broadcast) ─────────────────► masked (N, M)
                                                                                │
                              ReduceSum(axis=1) ──────────────────► sum_t (N,)
                                                                                │
              ReduceSum(mask, axis=1) ──────────────────────────── count (N,)
                                                                                │
                                          Div ─────────────────────► pred (N,)

    For query points with no neighbour within the radius the prediction is
    ``NaN`` (float division ``0 / 0``), which mirrors the fact that sklearn
    raises ``ValueError`` for such points at runtime.

    :param g: graph builder
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a fitted ``RadiusNeighborsRegressor``
    :param X: input tensor name
    :param name: prefix for node names
    :return: predicted value tensor
    :raises NotImplementedError: if opset < 13 or ``weights != 'uniform'``
    """
    assert isinstance(estimator, RadiusNeighborsRegressor)
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    opset = g.get_opset("")
    if opset < 13:
        raise NotImplementedError(
            f"RadiusNeighborsRegressor converter requires opset >= 13 "
            f"(ReduceSum with axes as input was added in opset 13), "
            f"but the graph builder has opset {opset}."
        )

    if estimator.weights != "uniform":
        raise NotImplementedError(
            f"Only 'uniform' weights are supported by the "
            f"RadiusNeighborsRegressor ONNX converter; "
            f"got {estimator.weights!r}."
        )

    raw_y = estimator._y
    assert (
        raw_y.ndim == 1 or raw_y.shape[1] == 1
    ), "Multi-output RadiusNeighborsRegressor is not yet supported."

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)
    radius = float(estimator.radius)
    training_data = estimator._fit_X  # (M, F)
    metric = estimator.effective_metric_
    metric_params = estimator.effective_metric_params_

    training_targets = (raw_y.ravel() if raw_y.ndim == 1 else raw_y[:, 0]).astype(dtype)  # (M,)

    # 1. Pairwise distances: (N, M)
    dists = _compute_pairwise_distances(
        g, X, training_data, itype, metric, f"{name}_dist", **metric_params
    )

    # 2. Radius mask: (N, M) bool — True where dist <= radius
    radius_const = np.array([radius], dtype=dtype)
    not_in_radius = g.op.Greater(dists, radius_const, name=f"{name}_gt")
    in_radius = g.op.Not(not_in_radius, name=f"{name}_mask")  # (N, M) bool

    # 3. Cast mask to the input dtype for arithmetic
    mask_float = g.op.Cast(in_radius, to=itype, name=f"{name}_mask_f")  # (N, M)

    # 4. Masked targets: (N, M) — broadcast training_targets (M,) over N rows
    masked_targets = g.op.Mul(mask_float, training_targets, name=f"{name}_masked")

    # 5. Sum targets within radius: (N,)
    sum_targets = g.op.ReduceSum(
        masked_targets,
        np.array([1], dtype=np.int64),
        keepdims=0,
        name=f"{name}_sum",
    )

    # 6. Count neighbours within radius: (N,)
    count = g.op.ReduceSum(
        mask_float,
        np.array([1], dtype=np.int64),
        keepdims=0,
        name=f"{name}_count",
    )

    # 7. Average: (N,) — NaN for outlier points (0 / 0 = NaN in float)
    predictions = g.op.Div(
        sum_targets,
        count,
        name=f"{name}_pred",
        outputs=outputs[:1],
    )
    assert isinstance(predictions, str)
    if not sts:
        g.set_type(predictions, itype)

    return predictions
