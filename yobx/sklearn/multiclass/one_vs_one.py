from typing import Dict, List, Union

import numpy as np
import onnx
from sklearn.multiclass import OneVsOneClassifier

from ..register import register_sklearn_converter, get_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ..sklearn_helper import get_n_expected_outputs
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(OneVsOneClassifier)
def sklearn_one_vs_one_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: OneVsOneClassifier,
    X: str,
    name: str = "one_vs_one",
) -> str:
    """
    Converts a :class:`sklearn.multiclass.OneVsOneClassifier` into ONNX.

    The converter replicates :meth:`~sklearn.multiclass.OneVsOneClassifier.predict`,
    which is equivalent to :meth:`~sklearn.multiclass.OneVsOneClassifier.decision_function`
    followed by ``argmax``.  For K classes there are K*(K-1)/2 binary
    sub-estimators, one for each pair (i, j) with i < j.

    For each pair k = (i, j):

    * ``confidence_k = predict_proba_k[:, 1]``  — probability that class *j* wins.
    * ``binary_pred_k = (confidence_k >= 0.5)``  — 1 if class j wins, 0 if class i wins.

    The vote and confidence accumulators for class *c* are:

    .. code-block:: text

        vote_c      = Σ_{k: c==i} (1 − pred_k)  +  Σ_{k: c==j} pred_k
        sum_conf_c  = Σ_{k: c==i} (−conf_k)      +  Σ_{k: c==j} conf_k

    The continuous tie-breaking transform is:

    .. code-block:: text

        transformed_conf_c = sum_conf_c / (3 * (|sum_conf_c| + 1))

    The final per-class score is:

    .. code-block:: text

        score_c = vote_c + transformed_conf_c

    The predicted label is ``classes_[argmax(score)]``.

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names (label only; OvO has no predict_proba)
    :param estimator: a fitted :class:`~sklearn.multiclass.OneVsOneClassifier`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: label tensor name
    :raises NotImplementedError: when ``estimator.pairwise_indices_`` is not ``None``
        or when a sub-estimator does not expose :meth:`predict_proba`
    """
    assert isinstance(
        estimator, OneVsOneClassifier
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    if estimator.pairwise_indices_ is not None:
        raise NotImplementedError(
            "OneVsOneClassifier with pairwise_indices_ != None is not yet supported."
        )

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)
    classes = estimator.classes_
    K = len(classes)

    # Build the ordered pair list: (i, j) for 0 <= i < j < K.
    # sklearn iterates in the same order to build estimators_.
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    assert len(pairs) == len(
        estimator.estimators_
    ), f"Expected {len(pairs)} estimators, got {len(estimator.estimators_)}"

    # ------------------------------------------------------------------
    # Step 1: obtain per-pair confidence (proba[:, 1]) and binary prediction
    # ------------------------------------------------------------------
    pair_confs: List[str] = []  # [N, 1] tensors matching the input dtype
    pair_preds: List[str] = []  # [N, 1] tensors (0.0 or 1.0) matching the input dtype

    for k, (ci, cj) in enumerate(pairs):
        sub_est = estimator.estimators_[k]
        sub_name = f"{name}__pair{k}_{ci}_{cj}"

        n_sub_outputs = get_n_expected_outputs(sub_est)
        if n_sub_outputs < 2:
            raise NotImplementedError(
                f"Sub-estimator {type(sub_est).__name__} does not expose predict_proba. "
                "Only sub-estimators with predict_proba are supported."
            )

        sub_label = g.unique_name(f"{sub_name}_label")
        sub_proba = g.unique_name(f"{sub_name}_proba")
        fct = get_sklearn_converter(type(sub_est))
        fct(g, sts, [sub_label, sub_proba], sub_est, X, name=sub_name)

        # Some converters do not register the output type; fall back to FLOAT.
        if not g.has_type(sub_proba):
            g.set_type(sub_proba, onnx.TensorProto.FLOAT)

        # Extract column 1 (positive-class probability): [N, 2] → [N, 1].
        conf_k = g.op.Slice(
            sub_proba,
            np.array([1], dtype=np.int64),  # starts
            np.array([2], dtype=np.int64),  # ends
            np.array([1], dtype=np.int64),  # axes
            name=f"{sub_name}_conf",
        )
        pair_confs.append(conf_k)

        # Binary prediction: 1 if class j wins (conf >= 0.5), 0 if class i wins.
        pred_k = g.op.Cast(
            g.op.GreaterOrEqual(
                conf_k,
                np.array([[0.5]], dtype=dtype),
                name=f"{sub_name}_pred_ge",
            ),
            to=itype,
            name=f"{sub_name}_pred",
        )
        pair_preds.append(pred_k)

    # ------------------------------------------------------------------
    # Step 2: accumulate votes and sum-of-confidences for each class
    # ------------------------------------------------------------------
    vote_cols: List[str] = []
    conf_cols: List[str] = []

    for c in range(K):
        vote_c: Union[str, None] = None
        conf_c: Union[str, None] = None

        for k, (ci, cj) in enumerate(pairs):
            if c == ci:
                # vote contribution: (1 - pred_k)  →  1 if class i won
                v_contrib = g.op.Sub(
                    np.array([[1.0]], dtype=dtype),
                    pair_preds[k],
                    name=f"{name}__v_c{c}_k{k}",
                )
                # confidence contribution: -conf_k
                c_contrib: str = g.op.Neg(
                    pair_confs[k],
                    name=f"{name}__c_c{c}_k{k}",
                )
            elif c == cj:
                # vote contribution: pred_k  →  1 if class j won
                v_contrib = pair_preds[k]
                # confidence contribution: +conf_k
                c_contrib = pair_confs[k]
            else:
                continue

            if vote_c is None:
                vote_c = v_contrib
                conf_c = c_contrib
            else:
                vote_c = g.op.Add(vote_c, v_contrib, name=f"{name}__vacc_c{c}_k{k}")
                conf_c = g.op.Add(conf_c, c_contrib, name=f"{name}__cacc_c{c}_k{k}")

        assert vote_c is not None, f"Class {c} was not covered by any pair."
        vote_cols.append(vote_c)
        conf_cols.append(conf_c)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Step 3: stack into [N, K] matrices and compute final scores
    # ------------------------------------------------------------------
    votes = g.op.Concat(*vote_cols, axis=1, name=f"{name}_votes")  # [N, K]
    sum_conf = g.op.Concat(*conf_cols, axis=1, name=f"{name}_sum_conf")  # [N, K]

    # Monotonic tie-breaking transform: f(x) = x / (3 * (|x| + 1))
    abs_conf = g.op.Abs(sum_conf, name=f"{name}_abs_conf")
    denom_inner = g.op.Add(
        abs_conf,
        np.array([1.0], dtype=dtype),
        name=f"{name}_denom_inner",
    )
    denom = g.op.Mul(
        np.array([3.0], dtype=dtype),
        denom_inner,
        name=f"{name}_denom",
    )
    transformed_conf = g.op.Div(sum_conf, denom, name=f"{name}_transformed_conf")

    # Final per-class scores.
    final_scores = g.op.Add(votes, transformed_conf, name=f"{name}_scores")

    # ------------------------------------------------------------------
    # Step 4: argmax → class label
    # ------------------------------------------------------------------
    label_idx_raw = g.op.ArgMax(final_scores, axis=1, keepdims=0, name=f"{name}_argmax")
    label_idx = g.op.Cast(label_idx_raw, to=onnx.TensorProto.INT64, name=f"{name}_cast")

    if np.issubdtype(classes.dtype, np.integer):
        classes_arr = classes.astype(np.int64)
        label = g.op.Gather(
            classes_arr,
            label_idx,
            axis=0,
            name=f"{name}_label",
            outputs=outputs[:1],
        )
        assert isinstance(label, str)
        if not sts:
            g.set_type(label, onnx.TensorProto.INT64)
    else:
        classes_arr = np.array(classes.astype(str))
        label = g.op.Gather(
            classes_arr,
            label_idx,
            axis=0,
            name=f"{name}_label_string",
            outputs=outputs[:1],
        )
        assert isinstance(label, str)
        if not sts:
            g.set_type(label, onnx.TensorProto.STRING)

    return label
