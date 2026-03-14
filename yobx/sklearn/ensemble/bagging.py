from typing import Dict, List, Tuple, Union

import numpy as np
import onnx
from sklearn.ensemble import BaggingClassifier, BaggingRegressor

from ...typing import GraphBuilderExtendedProtocol
from ..register import get_sklearn_converter, register_sklearn_converter
from ..sklearn_helper import get_n_expected_outputs


def _select_features(
    g: GraphBuilderExtendedProtocol, X: str, feature_indices: np.ndarray, name: str
) -> str:
    """
    Selects a subset of columns from *X* using a :onnx:`Gather` node.

    :param g: graph builder
    :param X: name of the ``(N, F)`` input tensor
    :param feature_indices: 1-D int64 array of column indices to select
    :param name: node name prefix
    :return: name of the ``(N, len(feature_indices))`` output tensor
    """
    idx = np.array(feature_indices, dtype=np.int64)
    sub_X = g.op.Gather(X, idx, axis=1, name=f"{name}_feature_gather")
    # Propagate the input type to the output of Gather.
    if g.has_type(X):
        g.set_type(sub_X, g.get_type(X))
    return sub_X


@register_sklearn_converter(BaggingRegressor)
def sklearn_bagging_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: BaggingRegressor,
    X: str,
    name: str = "bagging_regressor",
) -> str:
    """
    Converts a :class:`sklearn.ensemble.BaggingRegressor` into ONNX.

    Each sub-estimator's predictions (computed on the feature subset
    recorded in ``estimators_features_``) are averaged to produce the
    final prediction.

    Graph structure (two sub-estimators as an example):

    .. code-block:: text

        X ──Gather(cols_0)──[sub-est 0]──► pred_0 (N,)
        X ──Gather(cols_1)──[sub-est 1]──► pred_1 (N,)
                           Reshape(N,1) ──► pred_0, pred_1
                               Concat(axis=1) ──► stacked (N, E)
                                   ReduceMean(axis=1) ──► predictions (N,)

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names (one entry: predictions)
    :param estimator: a fitted :class:`~sklearn.ensemble.BaggingRegressor`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: name of the predictions output tensor
    """
    assert isinstance(
        estimator, BaggingRegressor
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    preds: List[str] = []
    for i, (sub_est, feat_idx) in enumerate(
        zip(estimator.estimators_, estimator.estimators_features_)
    ):
        sub_name = f"{name}__est{i}"

        # Select the subset of features used for this sub-estimator.
        sub_X = _select_features(g, X, feat_idx, sub_name)

        sub_out = g.unique_name(f"{sub_name}_pred")
        fct = get_sklearn_converter(type(sub_est))
        fct(g, sts, [sub_out], sub_est, sub_X, name=sub_name)

        # Reshape (N,) → (N, 1) for concatenation.
        pred_2d = g.op.Reshape(
            sub_out, np.array([-1, 1], dtype=np.int64), name=f"{sub_name}_reshape"
        )
        preds.append(pred_2d)

    stacked = g.op.Concat(*preds, axis=1, name=f"{name}_stack")  # (N, E)
    result = g.op.ReduceMean(
        stacked,
        np.array([1], dtype=np.int64),
        keepdims=0,
        name=f"{name}_mean",
        outputs=outputs[:1],
    )
    assert isinstance(result, str)
    return result


@register_sklearn_converter(BaggingClassifier)
def sklearn_bagging_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: BaggingClassifier,
    X: str,
    name: str = "bagging_classifier",
) -> Union[str, Tuple[str, str]]:
    """
    Converts a :class:`sklearn.ensemble.BaggingClassifier` into ONNX.

    Probabilities from all sub-estimators are averaged (soft aggregation)
    and the winning class is determined by an argmax over the averaged
    probability vector.  Each sub-estimator is applied to the feature
    subset recorded in ``estimators_features_``.

    Graph structure (two sub-estimators as an example):

    .. code-block:: text

        X ──Gather(cols_0)──[sub-est 0]──► (_, proba_0) (N, C)
        X ──Gather(cols_1)──[sub-est 1]──► (_, proba_1) (N, C)
                Unsqueeze(axis=0) ──► proba_0 (1, N, C), proba_1 (1, N, C)
                    Concat(axis=0) ──► stacked (E, N, C)
                        ReduceMean(axis=0) ──► avg_proba (N, C)
                            ArgMax(axis=1) ──Cast──Gather(classes_) ──► label

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names; two entries for
        ``(label, probabilities)``, one entry for label only
    :param estimator: a fitted :class:`~sklearn.ensemble.BaggingClassifier`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: label tensor name, or tuple ``(label, probabilities)``
    """
    assert isinstance(
        estimator, BaggingClassifier
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
        sub_X = _select_features(g, X, feat_idx, sub_name)

        n_sub_outputs = get_n_expected_outputs(sub_est)
        if n_sub_outputs < 2:
            raise NotImplementedError(
                f"Sub-estimator {type(sub_est).__name__} does not expose "
                "predict_proba. Only sub-estimators with predict_proba are "
                "supported for BaggingClassifier conversion."
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
    label_idx = g.op.Cast(label_idx_raw, to=onnx.TensorProto.INT64, name=f"{name}_cast_idx")

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
