from typing import Tuple, Dict, List
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..register import register_sklearn_converter
from ...xbuilder import GraphBuilder
from ...helpers.onnx_helper import choose_consistent_domain_opset

_LEAF = -1  # sklearn's TREE_LEAF constant


def _ensure_ml_domain(g: GraphBuilder) -> None:
    """Ensures the 'ai.onnx.ml' domain is registered in the graph builder."""
    if "ai.onnx.ml" not in g.opsets:
        g.opsets["ai.onnx.ml"] = choose_consistent_domain_opset("ai.onnx.ml", g.opsets)


def _extract_tree_attributes(tree, n_classes: int, is_classifier: bool):
    """
    Extracts the attributes needed by ONNX TreeEnsemble* operators
    from a fitted sklearn tree object.

    :param tree: ``estimator.tree_`` attribute
    :param n_classes: number of classes (for classifiers) or targets
    :param is_classifier: True for classification, False for regression
    :return: dict of ONNX attribute arrays

    .. note::
        Only single-output trees are supported (``tree.n_outputs == 1``).
        Multi-output trees would require multiple TreeEnsemble nodes.
    """
    assert (
        tree.n_outputs == 1
    ), f"Only single-output decision trees are supported, got n_outputs={tree.n_outputs}."
    n_nodes = tree.node_count
    feature = tree.feature
    threshold = tree.threshold.astype(np.float32)
    children_left = tree.children_left
    children_right = tree.children_right
    value = tree.value  # shape: (n_nodes, n_outputs, max_n_classes)

    nodes_featureids = []
    nodes_values = []
    nodes_modes = []
    nodes_truenodeids = []
    nodes_falsenodeids = []
    nodes_nodeids = []
    nodes_treeids = []
    nodes_hitrates = []
    nodes_missing_value_tracks_true = []

    target_nodeids = []
    target_treeids = []
    target_ids = []
    target_weights = []

    for node_id in range(n_nodes):
        nodes_nodeids.append(node_id)
        nodes_treeids.append(0)
        nodes_hitrates.append(1.0)
        nodes_missing_value_tracks_true.append(0)

        left = children_left[node_id]
        right = children_right[node_id]

        if left == _LEAF:
            # Leaf node
            nodes_featureids.append(0)
            nodes_values.append(0.0)
            nodes_modes.append("LEAF")
            nodes_truenodeids.append(0)
            nodes_falsenodeids.append(0)

            node_value = value[
                node_id, 0
            ]  # shape: (max_n_classes,); index 0 = first (only) output
            if is_classifier:
                total = node_value.sum()
                # total > 0 for all valid fitted nodes; guard against degenerate cases
                for class_idx in range(n_classes):
                    prob = float(node_value[class_idx]) / float(total) if total > 0 else 0.0
                    target_nodeids.append(node_id)
                    target_treeids.append(0)
                    target_ids.append(class_idx)
                    target_weights.append(prob)
            else:
                target_nodeids.append(node_id)
                target_treeids.append(0)
                target_ids.append(0)
                target_weights.append(float(node_value[0]))
        else:
            # Internal node
            nodes_featureids.append(int(feature[node_id]))
            nodes_values.append(float(threshold[node_id]))
            nodes_modes.append("BRANCH_LEQ")
            nodes_truenodeids.append(int(left))
            nodes_falsenodeids.append(int(right))

    if is_classifier:
        return dict(
            nodes_featureids=nodes_featureids,
            nodes_values=nodes_values,
            nodes_modes=nodes_modes,
            nodes_truenodeids=nodes_truenodeids,
            nodes_falsenodeids=nodes_falsenodeids,
            nodes_nodeids=nodes_nodeids,
            nodes_treeids=nodes_treeids,
            nodes_hitrates=nodes_hitrates,
            nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
            class_nodeids=target_nodeids,
            class_treeids=target_treeids,
            class_ids=target_ids,
            class_weights=target_weights,
        )
    return dict(
        nodes_featureids=nodes_featureids,
        nodes_values=nodes_values,
        nodes_modes=nodes_modes,
        nodes_truenodeids=nodes_truenodeids,
        nodes_falsenodeids=nodes_falsenodeids,
        nodes_nodeids=nodes_nodeids,
        nodes_treeids=nodes_treeids,
        nodes_hitrates=nodes_hitrates,
        nodes_missing_value_tracks_true=nodes_missing_value_tracks_true,
        target_nodeids=target_nodeids,
        target_treeids=target_treeids,
        target_ids=target_ids,
        target_weights=target_weights,
    )


@register_sklearn_converter((DecisionTreeClassifier,))
def sklearn_decision_tree_classifier(
    g: GraphBuilder,
    sts: Dict,
    outputs: List[str],
    estimator: DecisionTreeClassifier,
    X: str,
    name: str = "decision_tree_classifier",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.tree.DecisionTreeClassifier` into ONNX.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``DecisionTreeClassifier``
    :param outputs: desired names (label, probabilities)
    :param X: input tensor name
    :param name: prefix names for the added nodes
    :return: tuple ``(label_result_name, proba_result_name)``
    """
    assert isinstance(
        estimator, DecisionTreeClassifier
    ), f"Unexpected type {type(estimator)} for estimator."

    _ensure_ml_domain(g)
    classes = estimator.classes_
    n_classes = len(classes)
    tree = estimator.tree_
    attrs = _extract_tree_attributes(tree, n_classes, is_classifier=True)

    if np.issubdtype(classes.dtype, np.integer):
        classlabels = classes.astype(np.int64).tolist()
        label_kwargs = {"classlabels_int64s": classlabels}
    else:
        classlabels = classes.astype(str).tolist()
        label_kwargs = {"classlabels_strings": classlabels}

    result = g.make_node(
        "TreeEnsembleClassifier",
        [X],
        outputs=outputs,
        domain="ai.onnx.ml",
        name=name,
        post_transform="NONE",
        **attrs,
        **label_kwargs,
    )

    if isinstance(result, str):
        return result, result
    return result[0], result[1]


@register_sklearn_converter((DecisionTreeRegressor,))
def sklearn_decision_tree_regressor(
    g: GraphBuilder,
    sts: Dict,
    outputs: List[str],
    estimator: DecisionTreeRegressor,
    X: str,
    name: str = "decision_tree_regressor",
) -> str:
    """
    Converts a :class:`sklearn.tree.DecisionTreeRegressor` into ONNX.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``DecisionTreeRegressor``
    :param outputs: desired output names (predictions)
    :param X: input tensor name
    :param name: prefix names for the added nodes
    :return: output tensor name
    """
    assert isinstance(
        estimator, DecisionTreeRegressor
    ), f"Unexpected type {type(estimator)} for estimator."

    _ensure_ml_domain(g)
    tree = estimator.tree_
    attrs = _extract_tree_attributes(tree, n_classes=1, is_classifier=False)

    result = g.make_node(
        "TreeEnsembleRegressor",
        [X],
        outputs=outputs,
        domain="ai.onnx.ml",
        name=name,
        n_targets=1,
        post_transform="NONE",
        **attrs,
    )

    return result if isinstance(result, str) else result[0]
