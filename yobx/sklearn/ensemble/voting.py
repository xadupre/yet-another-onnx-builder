from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from sklearn.ensemble import VotingClassifier, VotingRegressor

from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from ...typing import GraphBuilderExtendedProtocol
from ..register import get_sklearn_converter, register_sklearn_converter
from ..sklearn_helper import get_n_expected_outputs


def _label_to_class_index(
    g: GraphBuilderExtendedProtocol, label: str, classes_arr: np.ndarray, name: str
) -> str:
    """
    Maps a tensor of class labels to class indices ``0 … C-1`` by broadcasting
    comparison against the ``classes_`` constant.

    Each predicted label must be present in *classes_arr* exactly once.
    The result is the argmax of the one-hot-like boolean row, which equals the
    position of the matching class.

    :param g: graph builder
    :param label: name of the ``(N,)`` label tensor
    :param classes_arr: 1-D numpy array of class values (int64 or string)
    :param name: node name prefix
    :return: name of the ``(N,)`` int64 class-index tensor
    """
    # Reshape label to (N, 1) for broadcasting.
    label_2d = g.op.Reshape(label, np.array([-1, 1], dtype=np.int64), name=f"{name}_reshape")
    # equal: (N, C) bool — True where label[i] == classes_[j]
    equal = g.op.Equal(label_2d, classes_arr, name=f"{name}_equal")
    # Cast to int64 and argmax over class axis → (N,) class indices.
    equal_int = g.op.Cast(equal, to=onnx.TensorProto.INT64, name=f"{name}_cast_eq")
    idx = g.op.ArgMax(equal_int, axis=1, keepdims=0, name=f"{name}_argmax_eq")
    return idx


def _build_label_output(
    g: GraphBuilderExtendedProtocol,
    winner_idx: str,
    classes: np.ndarray,
    outputs: List[str],
    name: str,
) -> str:
    """
    Builds the final label output by gathering from the ``classes_`` constant.

    :param g: graph builder
    :param winner_idx: ``(N,)`` int64 tensor with the winning class index
    :param classes: 1-D numpy array of class labels (int or string)
    :param outputs: output name list; the first entry is used for the label
    :param name: node name prefix
    :return: name of the label output tensor
    """
    if np.issubdtype(classes.dtype, np.integer):
        classes_arr = classes.astype(np.int64)
        label = g.op.Gather(
            classes_arr, winner_idx, axis=0, name=f"{name}_label", outputs=outputs[:1]
        )
        if not g.get_type(label):
            g.set_type(label, onnx.TensorProto.INT64)
    else:
        classes_arr = np.array(classes.astype(str))
        label = g.op.Gather(
            classes_arr, winner_idx, axis=0, name=f"{name}_label_str", outputs=outputs[:1]
        )
        if not g.get_type(label):
            g.set_type(label, onnx.TensorProto.STRING)
    return label


@register_sklearn_converter(VotingRegressor)
def sklearn_voting_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: VotingRegressor,
    X: str,
    name: str = "voting_regressor",
) -> str:
    """
    Converts a :class:`sklearn.ensemble.VotingRegressor` into ONNX.

    Each sub-estimator's predictions are averaged (optionally weighted).

    Graph structure (equal weights, two sub-estimators as an example):

    .. code-block:: text

        X ──[sub-est 0]──► pred_0 (N,)
        X ──[sub-est 1]──► pred_1 (N,)
                 Unsqueeze(axis=1) ──► pred_0 (N,1), pred_1 (N,1)
                        Concat(axis=1) ──► stacked (N, E)
                            ReduceMean(axis=1) ──► predictions (N,)

    With weights the mean is replaced by a weighted sum followed by division
    by the sum of weights.

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names (one entry: predictions)
    :param estimator: a fitted :class:`~sklearn.ensemble.VotingRegressor`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: name of the predictions output tensor
    """
    assert isinstance(
        estimator, VotingRegressor
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    weights: Optional[List] = estimator.weights

    preds: List[str] = []
    for i, sub_est in enumerate(estimator.estimators_):
        sub_name = f"{name}__est{i}"
        sub_out = g.unique_name(f"{sub_name}_pred")
        fct = get_sklearn_converter(type(sub_est))
        fct(g, sts, [sub_out], sub_est, X, name=sub_name)

        # Ensure shape (N, 1) for stacking.
        pred_2d = g.op.Reshape(
            sub_out, np.array([-1, 1], dtype=np.int64), name=f"{sub_name}_reshape"
        )
        preds.append(pred_2d)

    stacked = g.op.Concat(*preds, axis=1, name=f"{name}_stack")  # (N, E)

    if weights is None:
        # Simple average: ReduceMean over estimator axis.
        result = g.op.ReduceMean(
            stacked,
            np.array([1], dtype=np.int64),
            keepdims=0,
            name=f"{name}_mean",
            outputs=outputs[:1],
        )
    else:
        # Weighted average: dot product with weights / sum(weights).
        w = np.array(weights, dtype=dtype).reshape(1, -1)  # (1, E)
        weighted = g.op.Mul(stacked, w, name=f"{name}_weighted")
        sum_w = g.op.ReduceSum(
            weighted, np.array([1], dtype=np.int64), keepdims=0, name=f"{name}_wsum"
        )
        total_w = np.array(float(sum(weights)), dtype=dtype)
        result = g.op.Div(sum_w, total_w, name=f"{name}_wdiv", outputs=outputs[:1])

    return result


@register_sklearn_converter(VotingClassifier)
def sklearn_voting_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: VotingClassifier,
    X: str,
    name: str = "voting_classifier",
) -> Union[str, Tuple[str, str]]:
    """
    Converts a :class:`sklearn.ensemble.VotingClassifier` into ONNX.

    Both ``voting='soft'`` and ``voting='hard'`` are supported.

    **Soft voting** — average (weighted) class probabilities:

    .. code-block:: text

        X ──[sub-est 0]──► (_, proba_0) (N, C)
        X ──[sub-est 1]──► (_, proba_1) (N, C)
                Unsqueeze(axis=0) ──► proba_0 (1, N, C), proba_1 (1, N, C)
                    Concat(axis=0) ──► stacked (E, N, C)
                        ReduceMean(axis=0) ──► avg_proba (N, C)
                            ArgMax(axis=1) ──Cast──Gather(classes_) ──► label

    **Hard voting** — majority vote via one-hot vote accumulation:

    .. code-block:: text

        X ──[sub-est 0]──► label_0 (N,)
        X ──[sub-est 1]──► label_1 (N,)
          label_to_index ──► idx_0 (N,), idx_1 (N,)
              OneHot ──► votes_0 (N, C), votes_1 (N, C)
                  Add ──► total_votes (N, C)
                      ArgMax(axis=1) ──Cast──Gather(classes_) ──► label

    With non-None ``weights``, the averaging (soft) or one-hot accumulation
    (hard) is replaced by a weighted version.

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names; two entries for soft voting
        (label + probabilities), one entry for hard voting (label only)
    :param estimator: a fitted :class:`~sklearn.ensemble.VotingClassifier`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: label tensor name (hard voting) or tuple ``(label, probabilities)``
        (soft voting)
    """
    assert isinstance(
        estimator, VotingClassifier
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    classes = estimator.classes_
    n_classes = len(classes)
    voting = estimator.voting
    weights: Optional[List] = estimator.weights
    emit_proba = len(outputs) > 1

    if np.issubdtype(classes.dtype, np.integer):
        classes_arr = classes.astype(np.int64)
    else:
        classes_arr = np.array(classes.astype(str))

    # ------------------------------------------------------------------
    # Soft voting
    # ------------------------------------------------------------------
    if voting == "soft":
        probas: List[str] = []
        for i, sub_est in enumerate(estimator.estimators_):
            sub_name = f"{name}__est{i}"
            n_sub_outputs = get_n_expected_outputs(sub_est)
            if n_sub_outputs < 2:
                raise NotImplementedError(
                    f"Sub-estimator {type(sub_est).__name__} does not expose "
                    "predict_proba. Only sub-estimators with predict_proba are "
                    "supported for soft voting."
                )
            sub_label = g.unique_name(f"{sub_name}_label")
            sub_proba = g.unique_name(f"{sub_name}_proba")
            fct = get_sklearn_converter(type(sub_est))
            fct(g, sts, [sub_label, sub_proba], sub_est, X, name=sub_name)

            if not g.has_type(sub_proba):
                g.set_type(sub_proba, onnx.TensorProto.FLOAT)

            # Unsqueeze to (1, N, C) for stacking along axis 0.
            proba_3d = g.op.Unsqueeze(
                sub_proba, np.array([0], dtype=np.int64), name=f"{sub_name}_unsqueeze"
            )
            probas.append(proba_3d)

        stacked = g.op.Concat(*probas, axis=0, name=f"{name}_stack")  # (E, N, C)

        if weights is None:
            avg_proba = g.op.ReduceMean(
                stacked, np.array([0], dtype=np.int64), keepdims=0, name=f"{name}_mean"
            )  # (N, C)
        else:
            w = np.array(weights, dtype=dtype).reshape(-1, 1, 1)  # (E, 1, 1)
            weighted = g.op.Mul(stacked, w, name=f"{name}_weighted")
            wsum = g.op.ReduceSum(
                weighted, np.array([0], dtype=np.int64), keepdims=0, name=f"{name}_wsum"
            )  # (N, C)
            total_w = np.array(float(sum(weights)), dtype=dtype)
            avg_proba = g.op.Div(wsum, total_w, name=f"{name}_wdiv")  # (N, C)

        label_idx_raw = g.op.ArgMax(avg_proba, axis=1, keepdims=0, name=f"{name}_argmax")
        label_idx = g.op.Cast(label_idx_raw, to=onnx.TensorProto.INT64, name=f"{name}_cast_idx")
        label = _build_label_output(g, label_idx, classes, outputs, name)

        if emit_proba:
            proba_out = g.op.Identity(avg_proba, name=f"{name}_proba", outputs=outputs[1:])
            return label, proba_out

        return label

    # ------------------------------------------------------------------
    # Hard voting
    # ------------------------------------------------------------------
    # Each estimator contributes one (weighted) vote for its predicted class.
    # Votes are accumulated via broadcasting comparison (Equal) and Cast to
    # float, then summed.  This avoids OneHot type-inference issues.
    range_classes = np.arange(n_classes, dtype=np.int64).reshape(1, -1)  # (1, C)

    vote_tensors: List[str] = []
    for i, sub_est in enumerate(estimator.estimators_):
        sub_name = f"{name}__est{i}"
        sub_label = g.unique_name(f"{sub_name}_label")
        # We only need the label output; ignore probabilities if any.
        n_sub_outputs = get_n_expected_outputs(sub_est)
        sub_outputs_i = [sub_label]
        if n_sub_outputs > 1:
            sub_outputs_i.append(g.unique_name(f"{sub_name}_proba"))
        fct = get_sklearn_converter(type(sub_est))
        fct(g, sts, sub_outputs_i, sub_est, X, name=sub_name)

        # Map sub_label (class value) → class index (0 … C-1).
        class_idx = _label_to_class_index(g, sub_label, classes_arr, sub_name)
        class_idx_i64 = g.op.Cast(
            class_idx, to=onnx.TensorProto.INT64, name=f"{sub_name}_cast_cidx"
        )

        # Reshape to (N, 1) and compare with (1, C) → (N, C) bool vote mask.
        class_idx_2d = g.op.Unsqueeze(
            class_idx_i64, np.array([1], dtype=np.int64), name=f"{sub_name}_unsqueeze_cidx"
        )
        vote_bool = g.op.Equal(class_idx_2d, range_classes, name=f"{sub_name}_vote_bool")
        vote = g.op.Cast(vote_bool, to=onnx.TensorProto.FLOAT, name=f"{sub_name}_vote_cast")
        # Set type explicitly so subsequent Add can infer its output type.
        g.set_type(vote, onnx.TensorProto.FLOAT)

        if weights is not None:
            w_i = np.array(float(weights[i]), dtype=np.float32)
            vote = g.op.Mul(vote, w_i, name=f"{sub_name}_weighted")
            g.set_type(vote, onnx.TensorProto.FLOAT)

        vote_tensors.append(vote)

    # Sum votes: reduce over estimators.
    if len(vote_tensors) == 1:
        total_votes = vote_tensors[0]
    else:
        total_votes = vote_tensors[0]
        for vt in vote_tensors[1:]:
            total_votes = g.op.Add(total_votes, vt, name=f"{name}_vote_add")
            g.set_type(total_votes, onnx.TensorProto.FLOAT)

    winner_raw = g.op.ArgMax(total_votes, axis=1, keepdims=0, name=f"{name}_argmax")
    winner_idx = g.op.Cast(winner_raw, to=onnx.TensorProto.INT64, name=f"{name}_cast_winner")

    label = _build_label_output(g, winner_idx, classes, outputs, name)
    return label
