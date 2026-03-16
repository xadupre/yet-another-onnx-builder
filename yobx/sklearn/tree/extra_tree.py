from typing import Tuple, Dict, List
import numpy as np
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from .decision_tree import (
    _extract_tree_attributes,
    _extract_tree_attributes_v5,
    _sklearn_decision_tree_classifier_v5,
    _emit_decision_path_for_tree,
    _emit_decision_leaf_for_tree,
)


@register_sklearn_converter((ExtraTreeClassifier,))
def sklearn_extra_tree_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: ExtraTreeClassifier,
    X: str,
    name: str = "extra_tree_classifier",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.tree.ExtraTreeClassifier` into ONNX.

    :class:`sklearn.tree.ExtraTreeClassifier` is a single randomised tree
    that inherits from :class:`sklearn.tree.DecisionTreeClassifier` and
    shares the same ``tree_`` attribute layout.  This converter therefore
    delegates to the same helpers used by
    :func:`sklearn_decision_tree_classifier`.

    When ``ai.onnx.ml`` opset 5 (or later) is active in the graph builder
    the unified ``TreeEnsemble`` operator is used; otherwise the legacy
    ``TreeEnsembleClassifier`` operator is emitted.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``ExtraTreeClassifier``
    :param outputs: desired names (label, probabilities)
    :param X: input tensor name
    :param name: prefix names for the added nodes
    :return: tuple ``(label_result_name, proba_result_name)``
    """
    assert isinstance(
        estimator, ExtraTreeClassifier
    ), f"Unexpected type {type(estimator)} for estimator."

    ml_opset = g.get_opset("ai.onnx.ml")
    classes = estimator.classes_
    n_classes = len(classes)
    tree = estimator.tree_
    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    if ml_opset >= 5:
        return _sklearn_decision_tree_classifier_v5(
            g, sts, outputs, estimator, X, name, classes, n_classes, tree, dtype=dtype
        )

    # Legacy path: TreeEnsembleClassifier (ai.onnx.ml opset <= 4)
    attrs = _extract_tree_attributes(tree, n_classes, is_classifier=True)

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
    if g.convert_options.has("decision_path", estimator):
        assert len(outputs) > extra_idx, f"Missing output for decision_path in {outputs}"
        _emit_decision_path_for_tree(g, tree, X, outputs[extra_idx], f"{name}_dp")
        extra_idx += 1
    if g.convert_options.has("decision_leaf", estimator):
        assert len(outputs) > extra_idx, f"Missing output for decision_leaf in {outputs}"
        _emit_decision_leaf_for_tree(g, tree, X, outputs[extra_idx], f"{name}_dl")
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


@register_sklearn_converter((ExtraTreeRegressor,))
def sklearn_extra_tree_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: ExtraTreeRegressor,
    X: str,
    name: str = "extra_tree_regressor",
) -> str:
    """
    Converts a :class:`sklearn.tree.ExtraTreeRegressor` into ONNX.

    :class:`sklearn.tree.ExtraTreeRegressor` is a single randomised tree
    that inherits from :class:`sklearn.tree.DecisionTreeRegressor` and
    shares the same ``tree_`` attribute layout.  This converter therefore
    delegates to the same helpers used by
    :func:`sklearn_decision_tree_regressor`.

    When ``ai.onnx.ml`` opset 5 (or later) is active in the graph builder
    the unified ``TreeEnsemble`` operator is used; otherwise the legacy
    ``TreeEnsembleRegressor`` operator is emitted.

    When the input tensor is double (``float64``), a ``Cast`` node is appended
    after the tree operator to ensure the output dtype matches the input dtype.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``ExtraTreeRegressor``
    :param outputs: desired output names (predictions)
    :param X: input tensor name
    :param name: prefix names for the added nodes
    :return: output tensor name
    """
    assert isinstance(
        estimator, ExtraTreeRegressor
    ), f"Unexpected type {type(estimator)} for estimator."

    ml_opset = g.get_opset("ai.onnx.ml")
    tree = estimator.tree_

    if ml_opset >= 5:
        attrs = _extract_tree_attributes_v5(
            tree, n_classes=1, is_classifier=False, itype=g.get_type(X)
        )
        g.make_node(
            "TreeEnsemble",
            [X],
            outputs=outputs[:1],
            domain="ai.onnx.ml",
            name=f"{name}_te",
            post_transform=0,  # NONE
            aggregate_function=1,  # SUM
            **attrs,  # type: ignore
        )
        extra_idx = 1
        if g.convert_options.has("decision_path", estimator):
            assert len(outputs) > extra_idx, f"Missing output for decision_path in {outputs}"
            _emit_decision_path_for_tree(g, tree, X, outputs[extra_idx], f"{name}_dp")
            extra_idx += 1
        if g.convert_options.has("decision_leaf", estimator):
            assert len(outputs) > extra_idx, f"Missing output for decision_leaf in {outputs}"
            _emit_decision_leaf_for_tree(g, tree, X, outputs[extra_idx], f"{name}_dl")
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    # Legacy path: TreeEnsembleRegressor (ai.onnx.ml opset <= 4)
    attrs = _extract_tree_attributes(tree, n_classes=1, is_classifier=False)
    tree_outputs = [f"{outputs[0]}_cast"]
    tree_result = g.make_node(
        "TreeEnsembleRegressor",
        [X],
        outputs=tree_outputs,
        domain="ai.onnx.ml",
        name=name,
        n_targets=1,
        post_transform="NONE",
        **attrs,  # type: ignore
    )
    g.make_node(
        "Cast", [tree_result], outputs=outputs[:1], name=f"{name}_cast_f64", to=g.get_type(X)
    )
    extra_idx = 1
    if g.convert_options.has("decision_path", estimator):
        assert len(outputs) > extra_idx, f"Missing output for decision_path in {outputs}"
        _emit_decision_path_for_tree(g, tree, X, outputs[extra_idx], f"{name}_dp")
        extra_idx += 1
    if g.convert_options.has("decision_leaf", estimator):
        assert len(outputs) > extra_idx, f"Missing output for decision_leaf in {outputs}"
        _emit_decision_leaf_for_tree(g, tree, X, outputs[extra_idx], f"{name}_dl")
    return outputs[0] if len(outputs) == 1 else tuple(outputs)
