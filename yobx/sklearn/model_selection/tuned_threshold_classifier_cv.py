from typing import Dict, List, Tuple

import numpy as np
import onnx
from sklearn.model_selection import TunedThresholdClassifierCV

from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from ...typing import GraphBuilderExtendedProtocol
from ..register import get_sklearn_converter, register_sklearn_converter


@register_sklearn_converter(TunedThresholdClassifierCV, sklearn_version="1.5")
def sklearn_tuned_threshold_classifier_cv(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: TunedThresholdClassifierCV,
    X: str,
    name: str = "tuned_threshold_classifier_cv",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.model_selection.TunedThresholdClassifierCV`
    into ONNX.

    :class:`~sklearn.model_selection.TunedThresholdClassifierCV` wraps a binary
    classifier and adjusts the decision threshold for the positive class to
    optimise a scoring metric.  At inference time it:

    1. Obtains the positive-class probability from the inner estimator via
       ``predict_proba``.
    2. Compares that probability against ``best_threshold_``.
    3. Returns the positive-class label when the probability is **≥**
       ``best_threshold_``, and the negative-class label otherwise.

    The ONNX graph replicates this logic:

    .. code-block:: text

        X ──[estimator_ converter]──► (inner_label, probas (N, 2))
                                               │
                                   Gather(axis=1, idx=pos_label_idx)──► y_score (N,)
                                               │
                                   y_score >= best_threshold_ ──► bool (N,)
                                               │
                                   Cast(INT64) ──► 0 or 1
                                               │
                           Gather(map_thresholded_score_to_label) ──► label_idx (N,)
                                               │
                           Gather(classes_) ──► label (N,)

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names for the result
    :param estimator: a fitted
        :class:`~sklearn.model_selection.TunedThresholdClassifierCV`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: tuple ``(label, probabilities)``
    :raises NotImplementedError: if the inner estimator does not expose
        ``predict_proba`` (i.e. ``response_method='decision_function'`` was
        used) — only probability-based thresholding is currently supported
    """
    assert isinstance(
        estimator, TunedThresholdClassifierCV
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    inner = estimator.estimator_
    classes = estimator.classes_  # shape (2,) for binary

    if len(classes) != 2:
        raise NotImplementedError(
            f"TunedThresholdClassifierCV converter only supports binary classification "
            f"(got {len(classes)} classes: {classes})."
        )

    # Determine the positive-class index in classes_.
    # ``TunedThresholdClassifierCV`` stores the curve scorer as the private
    # attribute ``_curve_scorer``, and ``_get_pos_label()`` returns the
    # user-supplied ``pos_label`` (or ``None`` when the default is used).
    # There is no public API equivalent in scikit-learn ≥ 1.5; if this
    # changes in a future version the converter will need to be updated.
    # When pos_label is None, sklearn uses ``classes_[-1]`` as the positive
    # label, so ``map_thresholded_score_to_label = [0, 1]``.
    pos_label = estimator._curve_scorer._get_pos_label()
    if pos_label is None:
        pos_label_idx = len(classes) - 1  # last class
        neg_label_idx = 0
    else:
        pos_label_idx = int(np.flatnonzero(classes == pos_label)[0])
        neg_label_idx = int(np.flatnonzero(classes != pos_label)[0])

    # map_thresholded_score_to_label[i] is the index into classes_ for
    # binary outcome i (0 = below threshold, 1 = at-or-above threshold).
    map_idx = np.array([neg_label_idx, pos_label_idx], dtype=np.int64)

    # Obtain probabilities from the inner estimator.
    inner_fct = get_sklearn_converter(type(inner))
    inner_label_name = f"{name}__inner_label"
    inner_proba_name = f"{name}__inner_probas"
    inner_result = inner_fct(
        g, sts, [inner_label_name, inner_proba_name], inner, X, name=f"{name}__inner"
    )
    # inner_result is (label_str, proba_str) for a classifier
    _, probas = inner_result

    # Extract the positive-class probability column → shape (N,).
    y_score = g.op.Gather(
        probas,
        np.array(pos_label_idx, dtype=np.int64),
        axis=1,
        name=f"{name}_gather_pos",
    )  # (N,)

    # Compare against the tuned threshold → bool (N,).
    threshold = np.array(estimator.best_threshold_, dtype=dtype)
    score_ge = g.op.GreaterOrEqual(y_score, threshold, name=f"{name}_ge")  # (N,) bool

    # Cast bool → int64 (0 or 1).
    score_int = g.op.Cast(score_ge, to=onnx.TensorProto.INT64, name=f"{name}_cast_int")

    # Look up the classes_ index (0=neg, 1=pos) via map_idx.
    label_classes_idx = g.op.Gather(map_idx, score_int, axis=0, name=f"{name}_map_idx")

    # Look up the actual class label from classes_.
    if np.issubdtype(classes.dtype, np.integer):
        classes_arr = classes.astype(np.int64)
        label = g.op.Gather(
            classes_arr, label_classes_idx, axis=0, name=f"{name}_label", outputs=outputs[:1]
        )
        if not sts:
            g.set_type(label, onnx.TensorProto.INT64)
    else:
        classes_arr = np.array(classes.astype(str))
        label = g.op.Gather(
            classes_arr,
            label_classes_idx,
            axis=0,
            name=f"{name}_label_str",
            outputs=outputs[:1],
        )
        if not sts:
            g.set_type(label, onnx.TensorProto.STRING)

    assert isinstance(label, str)

    # Probabilities are unchanged (delegate to inner estimator's predict_proba).
    proba = g.op.Identity(probas, name=f"{name}_proba", outputs=outputs[1:])
    assert isinstance(proba, str)

    return label, proba
