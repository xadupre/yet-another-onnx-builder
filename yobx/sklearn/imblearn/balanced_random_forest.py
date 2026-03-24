from typing import Dict, List, Tuple, Union

import numpy as np
from imblearn.ensemble import BalancedRandomForestClassifier

from ...typing import GraphBuilderExtendedProtocol
from ..sklearn_helper import extract_step_name
from ..ensemble.random_forest import (
    _emit_decision_leaf_for_estimators,
    _emit_decision_path_for_estimators,
    _extract_forest_attributes_legacy,
    _sklearn_random_forest_classifier_v5,
)
from ..register import register_sklearn_converter


@register_sklearn_converter(BalancedRandomForestClassifier)
def sklearn_balanced_random_forest_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: BalancedRandomForestClassifier,
    X: str,
    name: str = "balanced_random_forest_classifier",
) -> Union[str, Tuple[str, str]]:
    """
    Converts an :class:`imblearn.ensemble.BalancedRandomForestClassifier`
    into ONNX.

    :class:`~imblearn.ensemble.BalancedRandomForestClassifier` is a subclass
    of :class:`~sklearn.ensemble.RandomForestClassifier` that trains each tree
    on a balanced bootstrap sample (obtained by random under-sampling).  At
    inference time the resampling step is inactive; the resulting forest has
    exactly the same structure as a regular ``RandomForestClassifier`` and is
    therefore converted using the same helpers.

    When ``ai.onnx.ml`` opset 5 (or later) is active in the graph builder
    the unified ``TreeEnsemble`` operator is used; otherwise the legacy
    ``TreeEnsembleClassifier`` operator is emitted.

    The forest is encoded as a single multi-tree ONNX node where each
    estimator's leaf weights are divided by ``n_estimators`` so that the
    ``SUM`` aggregate (or ``NONE`` post-transform in the legacy path) yields
    the averaged class-probability vector used by :meth:`predict_proba`.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names (label, probabilities)
    :param estimator: a fitted ``BalancedRandomForestClassifier``
    :param X: input tensor name
    :param name: prefix for node names added to the graph
    :return: label tensor name, or tuple ``(label_result_name, proba_result_name)``
    """
    assert isinstance(
        estimator, BalancedRandomForestClassifier
    ), f"Unexpected type {type(estimator)} for estimator."

    ml_opset = g.get_opset("ai.onnx.ml")
    classes = estimator.classes_
    n_classes = len(classes)
    n_estimators = estimator.n_estimators
    estimators = estimator.estimators_

    if ml_opset >= 5:
        return _sklearn_random_forest_classifier_v5(
            g,
            sts,
            outputs,
            estimator,
            X,
            name,
            classes,
            n_classes,
            n_estimators,
            estimators,
            itype=g.get_type(X),
        )

    # Legacy path: TreeEnsembleClassifier (ai.onnx.ml opset <= 4)
    attrs = _extract_forest_attributes_legacy(
        estimators, n_classes, is_classifier=True, n_estimators=n_estimators
    )

    if np.issubdtype(classes.dtype, np.integer):  # type: ignore
        classlabels = classes.astype(np.int64).tolist()  # type: ignore
        label_kwargs = {"classlabels_int64s": classlabels}
    else:
        classlabels = classes.astype(str).tolist()  # type: ignore
        label_kwargs = {"classlabels_strings": classlabels}

    g.make_node(
        "TreeEnsembleClassifier",
        [X],
        outputs=outputs[:2],
        domain="ai.onnx.ml",
        name=name,
        post_transform="NONE",
        **attrs,  # type: ignore
        **label_kwargs,
    )

    extra_idx = 2
    if g.convert_options.has("decision_path", estimator, extract_step_name(name)):
        assert len(outputs) > extra_idx, f"Missing output for decision_path in {outputs}"
        _emit_decision_path_for_estimators(g, estimators, X, outputs[extra_idx], f"{name}_dp")
        extra_idx += 1
    if g.convert_options.has("decision_leaf", estimator, extract_step_name(name)):
        assert len(outputs) > extra_idx, f"Missing output for decision_leaf in {outputs}"
        _emit_decision_leaf_for_estimators(g, estimators, X, outputs[extra_idx], f"{name}_dl")
    return outputs[0] if len(outputs) == 1 else tuple(outputs)
