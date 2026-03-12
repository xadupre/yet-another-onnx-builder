from typing import Dict, List, Tuple, Union

import numpy as np
import onnx
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from ...typing import GraphBuilderExtendedProtocol
from ..register import get_sklearn_converter, register_sklearn_converter
from ..sklearn_helper import get_n_expected_outputs
from .voting import _build_label_output


@register_sklearn_converter(AdaBoostClassifier)
def sklearn_adaboost_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: AdaBoostClassifier,
    X: str,
    name: str = "adaboost_classifier",
) -> Union[str, Tuple[str, str]]:
    """
    Converts a :class:`sklearn.ensemble.AdaBoostClassifier` into ONNX.

    Implements the SAMME (Stagewise Additive Modelling using a Multi-class
    Exponential loss function) algorithm that is the only algorithm
    supported by recent versions of scikit-learn.

    **Algorithm overview** — for each base estimator *i* with weight *wᵢ*:

    .. code-block:: text

        pred_i = estimator_i.predict(X)           # (N,) class values
        vote_i[j, k] = wᵢ          if pred_i[j] == classes_[k]
                     = -wᵢ/(C−1)   otherwise      # (N, C) float

        decision = Σᵢ vote_i / Σᵢ wᵢ             # (N, C)

    For **binary** classification (*C* = 2) the decision is folded to 1-D
    (``decision[:,0] *= -1`` then ``sum(axis=1)``) before computing
    probabilities.

    **Graph structure (multiclass, two base estimators as an example)**:

    .. code-block:: text

        X ──[base est 0]──► label_0 (N,)
        X ──[base est 1]──► label_1 (N,)
            label_i == classes_k ? w_i : -w_i/(C-1)  ──► vote_i (N, C)
                        Add votes ──► decision (N, C)
                    ArgMax(axis=1) ──Cast──Gather(classes_) ──► label
                decision/(C-1) ──Softmax ──► probabilities

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names; two entries
        (label + probabilities) or one (label only)
    :param estimator: a fitted :class:`~sklearn.ensemble.AdaBoostClassifier`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: label tensor name, or tuple ``(label, probabilities)``
    """
    assert isinstance(
        estimator, AdaBoostClassifier
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    classes = estimator.classes_
    n_classes = estimator.n_classes_
    weights = estimator.estimator_weights_
    weight_sum = np.array(weights.sum(), dtype=dtype)

    if np.issubdtype(classes.dtype, np.integer):
        classes_const = classes.astype(np.int64).reshape(1, -1)  # (1, C)
    else:
        classes_const = np.array(classes.astype(str)).reshape(1, -1)

    emit_proba = len(outputs) > 1

    # ------------------------------------------------------------------
    # Accumulate SAMME votes from each base estimator
    # ------------------------------------------------------------------
    vote_tensors: List[str] = []
    for i, (est, w) in enumerate(zip(estimator.estimators_, weights)):
        sub_name = f"{name}__est{i}"
        sub_label = g.unique_name(f"{sub_name}_label")

        n_sub_out = get_n_expected_outputs(est)
        sub_outs: List[str] = [sub_label]
        if n_sub_out > 1:
            sub_outs.append(g.unique_name(f"{sub_name}_proba"))

        fct = get_sklearn_converter(type(est))
        fct(g, sts, sub_outs, est, X, name=sub_name)

        # Reshape label (N,) → (N, 1) to broadcast-compare with classes (1, C)
        label_2d = g.op.Reshape(
            sub_label,
            np.array([-1, 1], dtype=np.int64),
            name=f"{sub_name}_reshape",
        )
        # one_hot_bool: (N, C) — True where label[i] matches classes_[k]
        one_hot_bool = g.op.Equal(label_2d, classes_const, name=f"{sub_name}_eq")
        one_hot = g.op.Cast(one_hot_bool, to=itype, name=f"{sub_name}_cast")
        g.set_type(one_hot, itype)

        # SAMME vote: wᵢ where predicted class, -wᵢ/(C-1) elsewhere.
        # vote = one_hot * (wᵢ + wᵢ/(C-1)) - wᵢ/(C-1)
        #      = one_hot * (wᵢ * C/(C-1))   - wᵢ/(C-1)
        pos_val = np.array(w * n_classes / (n_classes - 1), dtype=dtype)
        neg_val = np.array(-w / (n_classes - 1), dtype=dtype)
        weighted = g.op.Mul(one_hot, pos_val, name=f"{sub_name}_mul")
        vote = g.op.Add(weighted, neg_val, name=f"{sub_name}_vote")
        g.set_type(vote, itype)
        vote_tensors.append(vote)

    # Sum votes across all estimators → (N, C)
    total_votes: str = vote_tensors[0]
    for vt in vote_tensors[1:]:
        total_votes = g.op.Add(total_votes, vt, name=f"{name}_add_votes")  # type: ignore
        g.set_type(total_votes, itype)

    # Divide by total weight → decision (N, C)
    decision = g.op.Div(total_votes, weight_sum, name=f"{name}_decision")
    g.set_type(decision, itype)

    # ------------------------------------------------------------------
    # Binary case: collapse (N, 2) decision to 1-D
    #   decision[:,0] *= -1; decision_1d = decision.sum(axis=1)
    # ------------------------------------------------------------------
    if n_classes == 2:
        sign_mask = np.array([[-1.0, 1.0]], dtype=dtype)
        signed = g.op.Mul(decision, sign_mask, name=f"{name}_sign")
        g.set_type(signed, itype)
        decision_1d = g.op.ReduceSum(
            signed,
            np.array([1], dtype=np.int64),
            keepdims=0,
            name=f"{name}_reduce",
        )
        g.set_type(decision_1d, itype)

        # predict: classes_.take(decision_1d > 0)
        pred_gt0 = g.op.Greater(decision_1d, np.array(0.0, dtype=dtype), name=f"{name}_gt0")
        pred_idx = g.op.Cast(pred_gt0, to=onnx.TensorProto.INT64, name=f"{name}_idx")
        label = _build_label_output(g, pred_idx, classes, outputs, name)

        if emit_proba:
            # predict_proba: stack([-d, d]) / 2, then softmax
            dec_2d = g.op.Unsqueeze(
                decision_1d,
                np.array([1], dtype=np.int64),
                name=f"{name}_unsqueeze",
            )
            neg_dec = g.op.Neg(dec_2d, name=f"{name}_neg_dec")
            stacked = g.op.Concat(neg_dec, dec_2d, axis=1, name=f"{name}_stack")
            half = g.op.Div(stacked, np.array(2.0, dtype=dtype), name=f"{name}_half")
            proba = g.op.Softmax(half, axis=1, name=f"{name}_softmax", outputs=outputs[1:])
            assert isinstance(proba, str)
            return label, proba

        return label

    # ------------------------------------------------------------------
    # Multiclass: decision shape is already (N, C)
    # ------------------------------------------------------------------
    # predict: classes_.take(argmax(decision, axis=1))
    winner_raw = g.op.ArgMax(decision, axis=1, keepdims=0, name=f"{name}_argmax")
    winner_idx = g.op.Cast(winner_raw, to=onnx.TensorProto.INT64, name=f"{name}_cast_winner")
    label = _build_label_output(g, winner_idx, classes, outputs, name)

    if emit_proba:
        # predict_proba: softmax(decision / (C-1))
        proba_decision = g.op.Div(
            decision, np.array(n_classes - 1, dtype=dtype), name=f"{name}_proba_div"
        )
        proba = g.op.Softmax(proba_decision, axis=1, name=f"{name}_softmax", outputs=outputs[1:])
        assert isinstance(proba, str)
        return label, proba

    return label


@register_sklearn_converter(AdaBoostRegressor)
def sklearn_adaboost_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: AdaBoostRegressor,
    X: str,
    name: str = "adaboost_regressor",
) -> str:
    """
    Converts a :class:`sklearn.ensemble.AdaBoostRegressor` into ONNX.

    The prediction is the **weighted median** of the base estimators'
    predictions, following scikit-learn's R2 (AdaBoost.R2) algorithm.

    **Algorithm overview**:

    .. code-block:: text

        predictions = [est_i.predict(X) for i in range(E)]   # (E, N)
        sorted_idx  = argsort(predictions, axis=0)            # per-sample
        cumsum      = cumsum(weights[sorted_idx], axis=0)
        median_pos  = argmax(cumsum >= 0.5 * total_weight)
        output      = predictions[sorted_idx[median_pos]]

    **Graph structure** (three base estimators as an example):

    .. code-block:: text

        X ──[base est 0]──► pred_0 (N,)
        X ──[base est 1]──► pred_1 (N,)
        X ──[base est 2]──► pred_2 (N,)
               Concat(axis=1) ──► all_preds (N, E)
             TopK(k=E, asc) ──► sorted_vals (N, E), sorted_idx (N, E)
          Gather(weights, idx) ──► weights_sorted (N, E)
              CumSum(axis=1) ──► cumsum (N, E)
        cumsum >= 0.5*total ──► ArgMax ──► median_pos (N,)
         GatherElements ──► predictions (N,)

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names (one entry: predictions)
    :param estimator: a fitted :class:`~sklearn.ensemble.AdaBoostRegressor`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: name of the predictions output tensor
    """
    assert isinstance(
        estimator, AdaBoostRegressor
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    weights = estimator.estimator_weights_.astype(dtype)
    n_est = len(estimator.estimators_)
    half_total = np.array(0.5 * weights.sum(), dtype=dtype)

    # ------------------------------------------------------------------
    # Collect predictions from each base estimator, stack to (N, E)
    # ------------------------------------------------------------------
    pred_cols: List[str] = []
    for i, est in enumerate(estimator.estimators_):
        sub_name = f"{name}__est{i}"
        sub_pred = g.unique_name(f"{sub_name}_pred")
        fct = get_sklearn_converter(type(est))
        fct(g, sts, [sub_pred], est, X, name=sub_name)

        # Unsqueeze (N,) → (N, 1) for column-wise concat
        # (use Reshape to also handle predictions of shape (N, 1))
        pred_2d = g.op.Reshape(
            sub_pred,
            np.array([-1, 1], dtype=np.int64),
            name=f"{sub_name}_reshape",
        )
        pred_cols.append(pred_2d)

    all_preds = g.op.Concat(*pred_cols, axis=1, name=f"{name}_stack")  # (N, E)

    # ------------------------------------------------------------------
    # Sort predictions per sample in ascending order using TopK
    # ------------------------------------------------------------------
    k_val = np.array([n_est], dtype=np.int64)
    sorted_vals, sorted_idx = g.op.TopK(
        all_preds,
        k_val,
        largest=0,
        sorted=1,
        name=f"{name}_topk",
        outputs=2,
    )

    # ------------------------------------------------------------------
    # Weighted median via cumulative weight sum
    # ------------------------------------------------------------------
    # weights_sorted[i, j] = estimator_weights_[sorted_idx[i, j]]
    weights_const = weights  # shape (E,)
    weights_sorted = g.op.Gather(weights_const, sorted_idx, axis=0, name=f"{name}_gather_w")

    # Cumulative sum along estimator axis → (N, E)
    cumsum = g.op.CumSum(
        weights_sorted,
        np.array(1, dtype=np.int64),
        name=f"{name}_cumsum",
    )

    # Find first position where cumsum >= 0.5 * total_weight
    at_or_above = g.op.GreaterOrEqual(cumsum, half_total, name=f"{name}_ge")
    at_or_above_int = g.op.Cast(at_or_above, to=onnx.TensorProto.INT64, name=f"{name}_cast_bool")
    median_pos = g.op.ArgMax(
        at_or_above_int, axis=1, keepdims=0, name=f"{name}_median_pos"
    )  # (N,)

    # Gather the prediction value at each sample's median position
    median_pos_2d = g.op.Unsqueeze(
        median_pos,
        np.array([1], dtype=np.int64),
        name=f"{name}_median_unsqueeze",
    )  # (N, 1)
    result_2d = g.op.GatherElements(
        sorted_vals, median_pos_2d, axis=1, name=f"{name}_gather_pred"
    )  # (N, 1)
    result = g.op.Reshape(
        result_2d,
        np.array([-1], dtype=np.int64),
        name=f"{name}_reshape",
        outputs=outputs[:1],
    )
    assert isinstance(result, str)
    return result
