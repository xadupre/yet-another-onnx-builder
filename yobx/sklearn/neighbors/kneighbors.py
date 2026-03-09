from typing import Dict, List, Tuple, Union

import numpy as np
import onnx
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


def _compute_pairwise_sq_distances(
    g: GraphBuilderExtendedProtocol,
    X: str,
    training_data: np.ndarray,
    itype: int,
    name: str,
) -> str:
    """
    Computes pairwise squared Euclidean distances between *X* and *training_data*.

    When the ``com.microsoft`` domain is registered in the graph builder the
    computation delegates to ``com.microsoft.CDist`` (with
    ``metric="sqeuclidean"``), which may be accelerated by ONNX Runtime.
    Otherwise a standard-ONNX implementation is used, based on the identity

    .. code-block:: text

        ||x − c||² = ||x||² − 2·x·cᵀ + ||c||²

    :param g: graph builder
    :param X: input tensor name – shape ``(N, F)``
    :param training_data: training feature matrix – shape ``(M, F)``
    :param itype: ONNX element type of *X*
    :param name: node name prefix
    :return: name of the ``(N, M)`` squared-distance tensor
    """
    dtype = tensor_dtype_to_np_dtype(itype)
    training_data = training_data.astype(dtype)

    if g.has_opset("com.microsoft"):
        # Use com.microsoft.CDist; sqeuclidean returns squared distances directly.
        training_data_name = g.make_initializer(
            f"{name}_training_data", training_data
        )
        sq_dists = g.make_node(
            "CDist",
            [X, training_data_name],
            domain="com.microsoft",
            metric="sqeuclidean",
            name=f"{name}_cdist",
        )
        zero = np.array([0], dtype=dtype)
        return g.op.Max(sq_dists, zero, name=f"{name}_clip")

    # Standard ONNX: ||x||² − 2·x·cᵀ + ||c||²
    # ReduceSum with axes as a second input requires opset >= 13.
    opset = g.get_opset("")
    if opset < 13:
        raise NotImplementedError(
            f"The standard-ONNX KNN converter requires opset >= 13 "
            f"(ReduceSum with axes as input was added in opset 13), "
            f"but the graph builder has opset {opset}. "
            f"Use target_opset >= 13 or pass target_opset={{..., 'com.microsoft': 1}} "
            f"to enable the CDist path."
        )
    training_T = training_data.T  # (F, M)
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

    zero = np.array([0], dtype=dtype)
    return g.op.Max(sq_dists, zero, name=f"{name}_clip")


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

    The converter uses brute-force nearest-neighbour search with *uniform*
    sample weights and supports a single output.  When the ``com.microsoft``
    domain is registered in the graph builder the pairwise distances are
    computed with ``CDist``; otherwise a standard-ONNX implementation is used.

    Full graph structure (standard-ONNX path):

    .. code-block:: text

        X (N, F)
          │
          └─── pairwise squared distances ───────────────────────────► sq_dists (N, M)
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

    training_labels_encoded = (
        raw_y.ravel() if raw_y.ndim == 1 else raw_y[:, 0]
    ).astype(np.int64)  # (M,)

    # classes_ may be a plain ndarray (single output) or a list of arrays
    # (multi-output).  Normalise to a plain ndarray.
    classes_raw = estimator.classes_
    classes_arr = classes_raw[0] if isinstance(classes_raw, list) else classes_raw
    n_classes = len(classes_arr)

    # 1. Pairwise squared distances: (N, M)
    sq_dists = _compute_pairwise_sq_distances(
        g, X, training_data, itype, f"{name}_dist"
    )

    # 2. k nearest neighbours – values (N, k) and indices (N, k)
    _topk_values, nn_indices = g.op.TopK(
        sq_dists,
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

    The converter uses brute-force nearest-neighbour search with *uniform*
    sample weights and supports a single output.  When the ``com.microsoft``
    domain is registered in the graph builder the pairwise distances are
    computed with ``CDist``; otherwise a standard-ONNX implementation is used.

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

    training_targets = (
        raw_y.ravel() if raw_y.ndim == 1 else raw_y[:, 0]
    ).astype(dtype)  # (M,)

    # 1. Pairwise squared distances: (N, M)
    sq_dists = _compute_pairwise_sq_distances(
        g, X, training_data, itype, f"{name}_dist"
    )

    # 2. k nearest neighbour indices: (N, k)
    _topk_values, nn_indices = g.op.TopK(
        sq_dists,
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
