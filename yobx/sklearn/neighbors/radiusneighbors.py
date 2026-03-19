from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from sklearn.neighbors import RadiusNeighborsClassifier, RadiusNeighborsRegressor

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from .kneighbors import _compute_pairwise_distances


def _compute_vote_weights(
    g: GraphBuilderExtendedProtocol,
    dists: str,
    in_radius: str,
    itype: int,
    dtype: np.dtype,
    use_distance: bool,
    name: str,
) -> str:
    """
    Returns a float32 ``(N, M)`` weight matrix for radius-neighbor voting.

    * ``use_distance=False`` (uniform): ``Cast(in_radius, FLOAT32)`` — each
      in-radius neighbour contributes weight 1.
    * ``use_distance=True`` (distance): ``mask * 1 / max(dists, eps)`` — each
      in-radius neighbour contributes weight inversely proportional to its
      distance.  A tiny epsilon (``1e-12``) clips exact-zero distances so
      that the closest neighbour (or the query point itself when it appears in
      the training set) dominates the vote with an extremely large weight
      rather than producing a division-by-zero NaN.
    """
    mask_f32 = g.op.Cast(in_radius, to=onnx.TensorProto.FLOAT, name=f"{name}_mask_f32")

    if not use_distance:
        return mask_f32

    eps = np.array([1e-12], dtype=dtype)
    safe_dists = g.op.Max(dists, eps, name=f"{name}_safe_dists")
    one = np.array([1.0], dtype=dtype)
    inv_dists = g.op.Div(one, safe_dists, name=f"{name}_inv_dists")
    inv_dists_f32 = g.op.Cast(inv_dists, to=onnx.TensorProto.FLOAT, name=f"{name}_inv_f32")
    weights = g.op.Mul(mask_f32, inv_dists_f32, name=f"{name}_weights")
    return weights


def _classify_single_output(
    g: GraphBuilderExtendedProtocol,
    weights: str,
    training_labels_encoded: np.ndarray,
    classes_arr: np.ndarray,
    outputs: List[str],
    outlier_label_list: Optional[list],
    name: str,
    sts: Dict,
) -> Union[str, Tuple[str, str]]:
    """
    Builds the voting → label (and optionally probability) nodes for a
    **single-output** radius-neighbours classifier.

    ``weights`` is the ``(N, M)`` float32 matrix returned by
    :func:`_compute_vote_weights`.
    """
    n_classes = len(classes_arr)

    # One-hot encode training labels: (M, n_classes) float32
    one_hot = g.op.OneHot(
        training_labels_encoded,
        np.array(n_classes, dtype=np.int64),
        np.array([0, 1], dtype=np.float32),
        axis=1,
        name=f"{name}_onehot",
    )  # (M, n_classes)

    # Weighted votes: (N, M) @ (M, n_classes) = (N, n_classes)
    vote_counts = g.op.MatMul(weights, one_hot, name=f"{name}_votes")

    # Predicted class index: (N,) int64
    class_idx = g.op.ArgMax(vote_counts, axis=1, keepdims=0, name=f"{name}_argmax")

    # Map back to actual class labels: (N,)
    classes_init = (
        classes_arr.astype(np.int64)
        if np.issubdtype(classes_arr.dtype, np.integer)
        else classes_arr
    )

    predicted_label = g.op.Gather(classes_init, class_idx, axis=0, name=f"{name}_pred")

    # Handle outlier points (no neighbour within radius)
    if outlier_label_list is not None:
        outlier_label_ = outlier_label_list[0]
        # Count neighbours per query point: (N,) float32
        neighbor_count = g.op.ReduceSum(
            weights, np.array([1], dtype=np.int64), keepdims=0, name=f"{name}_nbcount"
        )
        half = np.array([0.5], dtype=np.float32)
        is_outlier = g.op.Less(neighbor_count, half, name=f"{name}_is_outlier")

        if np.issubdtype(classes_arr.dtype, np.integer):
            outlier_val = np.array([int(outlier_label_)], dtype=np.int64)
        else:
            outlier_val = np.array([str(outlier_label_)])

        labels = g.op.Where(
            is_outlier, outlier_val, predicted_label, name=f"{name}_labels", outputs=outputs[:1]
        )
    else:
        labels = g.op.Identity(predicted_label, name=f"{name}_labels", outputs=outputs[:1])

    n_out = len(outputs)
    if n_out >= 2:
        # Normalise vote counts → probabilities in [0, 1]
        total = g.op.ReduceSum(
            vote_counts, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_total"
        )  # (N, 1) float32
        # Clamp denominator to 1 to avoid NaN for outlier points (total == 0)
        safe_total = g.op.Max(total, np.array([1.0], dtype=np.float32), name=f"{name}_safe_total")
        probabilities = g.op.Div(
            vote_counts, safe_total, name=f"{name}_proba", outputs=outputs[1:2]
        )
        return labels, probabilities

    return labels


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

    Both ``weights='uniform'`` and ``weights='distance'`` are supported.
    Single-output and multi-output estimators are supported.

    For ``weights='distance'``, each in-radius neighbour is weighted by
    ``1 / max(distance, 1e-12)``.  When a query point coincides with a
    training point (distance ≈ 0), that training point's vote dominates.

    For multi-output estimators (``_y.shape[1] > 1``), each output is
    predicted independently.  Only the predicted labels are returned
    (probability output is not available in multi-output mode).

    Supported metrics are the same as for
    :func:`yobx.sklearn.neighbors.kneighbors.sklearn_knn_classifier` (see
    :func:`yobx.sklearn.neighbors.kneighbors._compute_pairwise_distances`).

    :param g: graph builder
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names; ``outputs[0]`` receives the predicted
        labels and ``outputs[1]`` (if present) receives the class probabilities
        (single-output mode only).
    :param estimator: a fitted ``RadiusNeighborsClassifier``
    :param X: input tensor name
    :param name: prefix for node names
    :return: predicted label tensor (and optionally probability tensor for
        single-output estimators)
    :raises NotImplementedError: if opset < 13
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

    raw_y = estimator._y  # (M,) or (M, n_outputs)
    is_multi_output = raw_y.ndim == 2 and raw_y.shape[1] > 1

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)
    radius = float(estimator.radius)
    training_data = estimator._fit_X  # (M, F)
    metric = estimator.effective_metric_
    metric_params = estimator.effective_metric_params_
    use_distance = estimator.weights == "distance"

    classes_raw = estimator.classes_

    # 1. Pairwise distances: (N, M)
    dists = _compute_pairwise_distances(
        g, X, training_data, itype, metric, f"{name}_dist", **metric_params
    )

    # 2. Radius mask: (N, M) bool — True where dist <= radius
    radius_const = np.array([radius], dtype=dtype)
    not_in_radius = g.op.Greater(dists, radius_const, name=f"{name}_gt")
    in_radius = g.op.Not(not_in_radius, name=f"{name}_mask")  # (N, M) bool

    # 3. Voting weights: (N, M) float32
    weights = _compute_vote_weights(
        g, dists, in_radius, itype, dtype, use_distance, f"{name}_wts"
    )

    if not is_multi_output:
        # Single-output path
        training_labels_encoded = (raw_y.ravel() if raw_y.ndim == 1 else raw_y[:, 0]).astype(
            np.int64
        )  # (M,)
        classes_arr = classes_raw[0] if isinstance(classes_raw, list) else classes_raw
        outlier_label_list = getattr(estimator, "outlier_label_", None)

        return _classify_single_output(
            g,
            weights,
            training_labels_encoded,
            classes_arr,
            outputs,
            outlier_label_list,
            name,
            sts,
        )

    # Multi-output path: predict each output independently, then concat
    # classes_ is always a list for multi-output estimators
    n_outputs = raw_y.shape[1]
    per_output_labels = []
    per_output_probas = []
    for t in range(n_outputs):
        labels_t = raw_y[:, t].astype(np.int64)  # (M,) encoded class indices
        classes_t = classes_raw[t]  # array of classes for output t
        n_classes_t = len(classes_t)

        classes_init_t = (
            classes_t.astype(np.int64)
            if np.issubdtype(classes_t.dtype, np.integer)
            else classes_t
        )

        if n_classes_t == 1:
            # Degenerate case: only one class in training data for this output.
            # The prediction is always that single class.  Build a (N,) constant
            # by broadcasting and an (N, 1) probability output.

            # Need N (batch size) to broadcast — derive from weights shape.
            # weights has shape (N, M); ReduceSum along axis=1 yields a (N,) vector.
            # The values are meaningless (they hold "total weight per query"); we only
            # need the shape.  Multiplying by 0.0 and casting to INT64 gives (N,) zeros,
            # which serve as gather indices for the single class.  Adding 1.0 (before the
            # final cast to FLOAT) gives (N,) ones for the trivial probability output.
            # A ZerosLike / ConstantOfShape approach is not used here because the
            # GraphBuilder helper methods abstract away ONNX opset differences in those
            # ops, so arithmetic-based broadcasting is the most portable choice.
            n_vec = g.op.ReduceSum(
                weights, np.array([1], dtype=np.int64), keepdims=0, name=f"{name}_n_{t}"
            )  # (N,) float32 — used only for its shape
            # (N,) int64 zeros: multiply by 0.0 then cast — portable shape broadcast
            zeros_like = g.op.Cast(
                g.op.Mul(n_vec, np.array([0.0], dtype=np.float32), name=f"{name}_z_{t}"),
                to=onnx.TensorProto.INT64,
                name=f"{name}_zidx_{t}",
            )  # (N,) zeros — index into single-class array
            pred_t = g.op.Gather(classes_init_t, zeros_like, axis=0, name=f"{name}_pred_{t}")
            # (N, 1) float32 all-ones probability (trivial: only one class)
            proba_t_1d = g.op.Cast(
                g.op.Add(n_vec, np.array([1.0], dtype=np.float32), name=f"{name}_one_{t}"),
                to=onnx.TensorProto.FLOAT,
                name=f"{name}_pfloat_{t}",
            )  # (N,) all-ones — add 1.0 to zero vector to get ones portably
            proba_t = g.op.Unsqueeze(
                proba_t_1d, np.array([1], dtype=np.int64), name=f"{name}_proba2d_{t}"
            )  # (N, 1)
        else:
            one_hot_t = g.op.OneHot(
                labels_t,
                np.array(n_classes_t, dtype=np.int64),
                np.array([0, 1], dtype=np.float32),
                axis=1,
                name=f"{name}_onehot_{t}",
            )  # (M, n_classes_t)

            # votes_t: (N, M) @ (M, n_classes_t) = (N, n_classes_t)
            votes_t = g.op.MatMul(weights, one_hot_t, name=f"{name}_votes_{t}")
            assert isinstance(votes_t, str)
            g.set_type(votes_t, onnx.TensorProto.FLOAT)

            class_idx_t = g.op.ArgMax(votes_t, axis=1, keepdims=0, name=f"{name}_argmax_{t}")
            pred_t = g.op.Gather(classes_init_t, class_idx_t, axis=0, name=f"{name}_pred_{t}")

            # Probability: normalise votes → (N, n_classes_t)
            total_t = g.op.ReduceSum(
                votes_t, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_tot_{t}"
            )  # (N, 1)
            safe_total_t = g.op.Max(
                total_t, np.array([1.0], dtype=np.float32), name=f"{name}_stot_{t}"
            )
            proba_t = g.op.Div(votes_t, safe_total_t, name=f"{name}_proba_t_{t}")

        # Reshape to (N, 1) for concat
        pred_t_2d = g.op.Unsqueeze(pred_t, np.array([1], dtype=np.int64), name=f"{name}_uns_{t}")
        per_output_labels.append(pred_t_2d)
        per_output_probas.append(proba_t)

    # Concat labels along axis=1 to get (N, n_outputs)
    labels = g.op.Concat(*per_output_labels, axis=1, name=f"{name}_labels", outputs=outputs[:1])
    assert isinstance(labels, str)
    if not sts:
        classes_t0 = classes_raw[0]
        out_itype = (
            onnx.TensorProto.INT64
            if np.issubdtype(classes_t0.dtype, np.integer)
            else onnx.TensorProto.STRING
        )
        g.set_type(labels, out_itype)

    n_out = len(outputs)
    if n_out >= 2:
        # Concat probabilities along axis=1: (N, sum(n_classes_t))
        probabilities = g.op.Concat(
            *per_output_probas, axis=1, name=f"{name}_proba", outputs=outputs[1:2]
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

    Both ``weights='uniform'`` and ``weights='distance'`` are supported.
    Single-output and multi-output estimators are supported.

    For ``weights='distance'``, each in-radius neighbour is weighted by
    ``1 / max(distance, 1e-12)``.  When a query point coincides with a
    training point (distance ≈ 0), that training point's weight dominates.

    For multi-output estimators (``_y.shape[1] > 1``), the prediction is a
    ``(N, n_targets)`` tensor.

    For query points with no neighbour within the radius the prediction is
    ``NaN`` (float division ``0 / 0`` for uniform, or ``0 / 0`` for distance
    weights), which mirrors the fact that sklearn raises ``ValueError`` for
    such points at runtime.

    Supported metrics are the same as for
    :func:`yobx.sklearn.neighbors.kneighbors.sklearn_knn_regressor` (see
    :func:`yobx.sklearn.neighbors.kneighbors._compute_pairwise_distances`).

    :param g: graph builder
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a fitted ``RadiusNeighborsRegressor``
    :param X: input tensor name
    :param name: prefix for node names
    :return: predicted value tensor — shape ``(N,)`` for single-output or
        ``(N, n_targets)`` for multi-output estimators
    :raises NotImplementedError: if opset < 13
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

    raw_y = estimator._y  # (M,) or (M, n_targets)
    is_multi_output = raw_y.ndim == 2 and raw_y.shape[1] > 1

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)
    radius = float(estimator.radius)
    training_data = estimator._fit_X  # (M, F)
    metric = estimator.effective_metric_
    metric_params = estimator.effective_metric_params_
    use_distance = estimator.weights == "distance"

    # 1. Pairwise distances: (N, M)
    dists = _compute_pairwise_distances(
        g, X, training_data, itype, metric, f"{name}_dist", **metric_params
    )

    # 2. Radius mask: (N, M) bool — True where dist <= radius
    radius_const = np.array([radius], dtype=dtype)
    not_in_radius = g.op.Greater(dists, radius_const, name=f"{name}_gt")
    in_radius = g.op.Not(not_in_radius, name=f"{name}_mask")  # (N, M) bool

    # 3. Per-neighbour weights: (N, M), same dtype as X
    if use_distance:
        eps = np.array([1e-12], dtype=dtype)
        safe_dists = g.op.Max(dists, eps, name=f"{name}_safe_dists")
        one = np.array([1.0], dtype=dtype)
        inv_dists = g.op.Div(one, safe_dists, name=f"{name}_inv_dists")
        mask_float = g.op.Cast(in_radius, to=itype, name=f"{name}_mask_f")
        weights = g.op.Mul(mask_float, inv_dists, name=f"{name}_weights")  # (N, M)
    else:
        weights = g.op.Cast(in_radius, to=itype, name=f"{name}_weights")  # (N, M)

    if is_multi_output:
        # training_targets: (M, T)
        training_targets = raw_y.astype(dtype)  # (M, T)

        # weighted_sum: (N, M) @ (M, T) = (N, T)
        weighted_sum = g.op.MatMul(weights, training_targets, name=f"{name}_wsum")

        # weight_sum: (N, 1)
        weight_sum = g.op.ReduceSum(
            weights, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_wcount"
        )

        predictions = g.op.Div(weighted_sum, weight_sum, name=f"{name}_pred", outputs=outputs[:1])
    else:
        # Single-output path
        training_targets = (raw_y.ravel() if raw_y.ndim == 1 else raw_y[:, 0]).astype(
            dtype
        )  # (M,)

        # Masked/weighted targets: (N, M)
        weighted_targets = g.op.Mul(weights, training_targets, name=f"{name}_wt")

        # Sum over neighbours: (N,)
        sum_targets = g.op.ReduceSum(
            weighted_targets, np.array([1], dtype=np.int64), keepdims=0, name=f"{name}_sum"
        )

        # Sum of weights (= count for uniform, = sum(1/d) for distance): (N,)
        weight_sum = g.op.ReduceSum(
            weights, np.array([1], dtype=np.int64), keepdims=0, name=f"{name}_wsum"
        )

        # Average / weighted average: NaN when no in-radius neighbour (0/0)
        predictions = g.op.Div(sum_targets, weight_sum, name=f"{name}_pred", outputs=outputs[:1])

    return predictions
