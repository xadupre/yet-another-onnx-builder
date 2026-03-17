from typing import Dict, List, Tuple, Union

import numpy as np
import onnx
from imblearn.ensemble import BalancedBaggingClassifier

from ...typing import GraphBuilderExtendedProtocol
from ..register import get_sklearn_converter, register_sklearn_converter
from ..sklearn_helper import get_n_expected_outputs


@register_sklearn_converter(BalancedBaggingClassifier)
def sklearn_balanced_bagging_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: BalancedBaggingClassifier,
    X: str,
    name: str = "balanced_bagging_classifier",
) -> Union[str, Tuple[str, str]]:
    """
    Converts an :class:`imblearn.ensemble.BalancedBaggingClassifier` into ONNX.

    :class:`~imblearn.ensemble.BalancedBaggingClassifier` is a bagging
    ensemble where each sub-estimator is trained on a balanced bootstrap
    sample (obtained via random under-sampling).  At inference time, the
    resampling step is inactive; each sub-estimator is therefore effectively
    a plain classifier.

    Probabilities from all sub-estimators are averaged (soft aggregation)
    and the winning class is determined by an argmax over the averaged
    probability vector.  Each sub-estimator is applied to the feature subset
    recorded in ``estimators_features_`` (all features by default).

    The sub-estimators stored in ``estimators_`` are
    :class:`imblearn.pipeline.Pipeline` instances wrapping a resampler and a
    classifier.  This converter dispatches to the registered converter for
    :class:`imblearn.pipeline.Pipeline` (defined in
    :mod:`yobx.sklearn.imblearn.easy_ensemble`), which transparently drops
    the resampler at inference time.

    Graph structure (two sub-estimators as an example):

    .. code-block:: text

        X ──Gather(cols_0)──[sub-pipeline 0]──► (_, proba_0) (N, C)
        X ──Gather(cols_1)──[sub-pipeline 1]──► (_, proba_1) (N, C)
                Unsqueeze(axis=0) ──► proba_0 (1, N, C), proba_1 (1, N, C)
                    Concat(axis=0) ──► stacked (E, N, C)
                        ReduceMean(axis=0) ──► avg_proba (N, C)
                            ArgMax(axis=1) ──Cast──Gather(classes_) ──► label

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names; two entries for
        ``(label, probabilities)``, one entry for label only
    :param estimator: a fitted
        :class:`~imblearn.ensemble.BalancedBaggingClassifier`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: label tensor name, or tuple ``(label, probabilities)``
    """
    assert isinstance(
        estimator, BalancedBaggingClassifier
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    classes = estimator.classes_
    emit_proba = len(outputs) > 1

    if np.issubdtype(classes.dtype, np.integer):
        classes_arr = classes.astype(np.int64)
    else:
        classes_arr = np.array(classes.astype(str))

    probas: List[str] = []
    for i, (sub_est, feat_idx) in enumerate(
        zip(estimator.estimators_, estimator.estimators_features_)
    ):
        sub_name = f"{name}__est{i}"

        # Select the subset of features used for this sub-estimator.
        idx = np.array(feat_idx, dtype=np.int64)
        sub_X = g.op.Gather(X, idx, axis=1, name=f"{sub_name}_feature_gather")
        if g.has_type(X):
            g.set_type(sub_X, g.get_type(X))

        n_sub_outputs = get_n_expected_outputs(sub_est)
        if n_sub_outputs < 2:
            raise NotImplementedError(
                f"Sub-estimator {type(sub_est).__name__} does not expose "
                "predict_proba. Only sub-estimators with predict_proba are "
                "supported for BalancedBaggingClassifier conversion."
            )

        sub_label = g.unique_name(f"{sub_name}_label")
        sub_proba = g.unique_name(f"{sub_name}_proba")
        fct = get_sklearn_converter(type(sub_est))
        fct(g, sts, [sub_label, sub_proba], sub_est, sub_X, name=sub_name)

        if not g.has_type(sub_proba):
            g.set_type(sub_proba, onnx.TensorProto.FLOAT)

        # Unsqueeze to (1, N, C) for stacking along axis 0.
        proba_3d = g.op.Unsqueeze(
            sub_proba, np.array([0], dtype=np.int64), name=f"{sub_name}_unsqueeze"
        )
        probas.append(proba_3d)

    stacked = g.op.Concat(*probas, axis=0, name=f"{name}_stack")  # (E, N, C)
    avg_proba = g.op.ReduceMean(
        stacked, np.array([0], dtype=np.int64), keepdims=0, name=f"{name}_mean"
    )  # (N, C)

    label_idx_raw = g.op.ArgMax(avg_proba, axis=1, keepdims=0, name=f"{name}_argmax")
    label_idx = g.op.Cast(
        label_idx_raw, to=onnx.TensorProto.INT64, name=f"{name}_cast_idx"
    )

    # Build the label output by gathering from classes_.
    if np.issubdtype(classes.dtype, np.integer):
        label = g.op.Gather(
            classes_arr, label_idx, axis=0, name=f"{name}_label", outputs=outputs[:1]
        )
        assert isinstance(label, str)
        if not g.has_type(label):
            g.set_type(label, onnx.TensorProto.INT64)
    else:
        label = g.op.Gather(
            classes_arr, label_idx, axis=0, name=f"{name}_label_str", outputs=outputs[:1]
        )
        assert isinstance(label, str)
        if not g.has_type(label):
            g.set_type(label, onnx.TensorProto.STRING)

    if emit_proba:
        proba_out = g.op.Identity(avg_proba, name=f"{name}_proba", outputs=outputs[1:])
        assert isinstance(proba_out, str)
        return label, proba_out

    return label
