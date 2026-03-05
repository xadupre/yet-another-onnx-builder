from typing import Tuple, Dict, List
import numpy as np
import onnx
import onnx.helper as oh
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from ..register import register_sklearn_converter
from ...xbuilder import GraphBuilder
from ...helpers.onnx_helper import choose_consistent_domain_opset

_LEAF = -1  # sklearn's TREE_LEAF constant

# Mode encoding for the TreeEnsemble operator (ai.onnx.ml opset 5)
_NODE_MODE_LEQ = np.uint8(0)


def _ensure_ml_domain(g: GraphBuilder) -> None:
    """Ensures the 'ai.onnx.ml' domain is registered in the graph builder."""
    if "ai.onnx.ml" not in g.opsets:
        g.opsets["ai.onnx.ml"] = choose_consistent_domain_opset("ai.onnx.ml", g.opsets)


def _get_ml_opset(g: GraphBuilder) -> int:
    """Returns the resolved ``ai.onnx.ml`` opset version from the builder."""
    _ensure_ml_domain(g)
    return g.opsets.get("ai.onnx.ml", 1)


def _extract_tree_attributes(tree, n_classes: int, is_classifier: bool):
    """
    Extracts the attributes needed by the legacy ONNX ``TreeEnsembleClassifier``
    / ``TreeEnsembleRegressor`` operators (``ai.onnx.ml`` opset <= 4).

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


def _extract_tree_attributes_v5(tree, n_classes: int, is_classifier: bool):
    """
    Extracts the attributes needed by the unified ``TreeEnsemble`` operator
    introduced in ``ai.onnx.ml`` opset 5.

    The new operator separates *interior* nodes (stored in ``nodes_*``
    arrays) from *leaf* nodes (stored in ``leaf_*`` arrays).  For
    classification, ``n_classes`` independent trees are emitted — one per
    class — each outputting the probability for that class.  Aggregation
    (SUM) over all trees then yields the full ``[N, n_classes]`` score
    matrix.  For regression a single tree with one target is emitted.

    :param tree: ``estimator.tree_`` attribute
    :param n_classes: number of classes (classifiers) or 1 (regressors)
    :param is_classifier: True for classification, False for regression
    :return: dict of ONNX attributes ready to be passed to ``make_node``

    .. note::
        Only single-output trees are supported (``tree.n_outputs == 1``).
    """
    assert (
        tree.n_outputs == 1
    ), f"Only single-output decision trees are supported, got n_outputs={tree.n_outputs}."

    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold.astype(np.float32)
    value = tree.value  # shape: (n_nodes, n_outputs, max_n_classes)
    n_nodes = tree.node_count

    # Partition nodes into internal (has children) vs leaf (no children).
    internal_nodes: List[int] = []
    leaf_nodes: List[int] = []
    for nid in range(n_nodes):
        if children_left[nid] == _LEAF:
            leaf_nodes.append(nid)
        else:
            internal_nodes.append(nid)

    internal_node_to_idx = {nid: idx for idx, nid in enumerate(internal_nodes)}
    leaf_node_to_idx = {nid: idx for idx, nid in enumerate(leaf_nodes)}

    n_internal = len(internal_nodes)
    n_leaves = len(leaf_nodes)

    # ------------------------------------------------------------------ #
    # Build nodes_* arrays.                                               #
    # For classification: n_classes copies of the internal-node structure #
    # (one per class / virtual tree).                                     #
    # For regression: a single copy.                                      #
    # ------------------------------------------------------------------ #
    n_trees = n_classes if is_classifier else 1

    nodes_featureids_: List[int] = []
    nodes_splits_: List[float] = []
    nodes_modes_: List[int] = []
    nodes_truenodeids_: List[int] = []
    nodes_trueleafs_: List[int] = []
    nodes_falsenodeids_: List[int] = []
    nodes_falseleafs_: List[int] = []

    for tree_idx in range(n_trees):
        node_offset = tree_idx * n_internal
        leaf_offset = tree_idx * n_leaves

        if not internal_nodes:
            # Degenerate tree: root is the only node and is a leaf.
            # Represent it as a single dummy internal node where both branches
            # point to the same leaf (index 0 in this tree's leaf block).
            # The leaf entry at `leaf_offset` is populated in the leaf-building
            # loop below, which runs immediately after this node-building loop.
            nodes_featureids_.append(0)
            nodes_splits_.append(0.0)
            nodes_modes_.append(int(_NODE_MODE_LEQ))
            nodes_truenodeids_.append(leaf_offset)
            nodes_trueleafs_.append(1)
            nodes_falsenodeids_.append(leaf_offset)
            nodes_falseleafs_.append(1)
        else:
            for nid in internal_nodes:
                left = int(children_left[nid])
                right = int(children_right[nid])

                nodes_featureids_.append(int(feature[nid]))
                nodes_splits_.append(float(threshold[nid]))
                nodes_modes_.append(int(_NODE_MODE_LEQ))

                # True branch → left child (sklearn: feature <= threshold)
                left_is_leaf = children_left[left] == _LEAF
                if left_is_leaf:
                    nodes_truenodeids_.append(leaf_offset + leaf_node_to_idx[left])
                else:
                    nodes_truenodeids_.append(node_offset + internal_node_to_idx[left])
                nodes_trueleafs_.append(1 if left_is_leaf else 0)

                # False branch → right child (sklearn: feature > threshold)
                right_is_leaf = children_left[right] == _LEAF
                if right_is_leaf:
                    nodes_falsenodeids_.append(leaf_offset + leaf_node_to_idx[right])
                else:
                    nodes_falsenodeids_.append(node_offset + internal_node_to_idx[right])
                nodes_falseleafs_.append(1 if right_is_leaf else 0)

    # tree_roots: index of each tree's root in the nodes_* arrays.
    # For a degenerate tree (no internal nodes) we still create one dummy
    # node per tree, so the offsets are identical to the regular case.
    effective_n_internal = max(n_internal, 1)  # 1 dummy node when degenerate
    tree_roots = [tree_idx * effective_n_internal for tree_idx in range(n_trees)]

    # ------------------------------------------------------------------ #
    # Build leaf_* arrays.                                                #
    # ------------------------------------------------------------------ #
    leaf_targetids_: List[int] = []
    leaf_weights_: List[float] = []

    for tree_idx in range(n_trees):
        for nid in leaf_nodes:
            node_value = value[nid, 0]  # shape: (max_n_classes,)
            if is_classifier:
                total = float(node_value.sum())
                prob = float(node_value[tree_idx]) / total if total > 0.0 else 0.0
                leaf_targetids_.append(tree_idx)
                leaf_weights_.append(prob)
            else:
                leaf_targetids_.append(0)
                leaf_weights_.append(float(node_value[0]))

        if not leaf_nodes:
            # Degenerate tree: single leaf derived from the root node.
            node_value = value[0, 0]
            if is_classifier:
                total = float(node_value.sum())
                prob = float(node_value[tree_idx]) / total if total > 0.0 else 0.0
                leaf_targetids_.append(tree_idx)
                leaf_weights_.append(prob)
            else:
                leaf_targetids_.append(0)
                leaf_weights_.append(float(node_value[0]))

    # Pack tensor attributes (nodes_splits, nodes_modes, leaf_weights must
    # be ONNX tensors in the opset-5 encoding).
    nodes_splits_tensor = oh.make_tensor(
        "nodes_splits",
        onnx.TensorProto.FLOAT,
        (len(nodes_splits_),),
        np.array(nodes_splits_, dtype=np.float32),
    )
    nodes_modes_tensor = oh.make_tensor(
        "nodes_modes",
        onnx.TensorProto.UINT8,
        (len(nodes_modes_),),
        np.array(nodes_modes_, dtype=np.uint8),
    )
    leaf_weights_tensor = oh.make_tensor(
        "leaf_weights",
        onnx.TensorProto.FLOAT,
        (len(leaf_weights_),),
        np.array(leaf_weights_, dtype=np.float32),
    )

    return dict(
        tree_roots=tree_roots,
        n_targets=n_classes if is_classifier else 1,
        nodes_featureids=nodes_featureids_,
        nodes_splits=nodes_splits_tensor,
        nodes_modes=nodes_modes_tensor,
        nodes_truenodeids=nodes_truenodeids_,
        nodes_trueleafs=nodes_trueleafs_,
        nodes_falsenodeids=nodes_falsenodeids_,
        nodes_falseleafs=nodes_falseleafs_,
        leaf_targetids=leaf_targetids_,
        leaf_weights=leaf_weights_tensor,
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

    When ``ai.onnx.ml`` opset 5 (or later) is active in the graph builder
    the unified ``TreeEnsemble`` operator is used; otherwise the legacy
    ``TreeEnsembleClassifier`` operator is emitted.

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

    ml_opset = _get_ml_opset(g)
    classes = estimator.classes_
    n_classes = len(classes)
    tree = estimator.tree_

    if ml_opset >= 5:
        return _sklearn_decision_tree_classifier_v5(
            g, sts, outputs, estimator, X, name, classes, n_classes, tree
        )

    # Legacy path: TreeEnsembleClassifier (ai.onnx.ml opset <= 4)
    attrs = _extract_tree_attributes(tree, n_classes, is_classifier=True)

    if np.issubdtype(classes.dtype, np.integer):  # type: ignore
        classlabels = classes.astype(np.int64).tolist()  # type: ignore
        label_kwargs = {"classlabels_int64s": classlabels}
    else:
        classlabels = classes.astype(str).tolist()  # type: ignore
        label_kwargs = {"classlabels_strings": classlabels}

    result = g.make_node(
        "TreeEnsembleClassifier",
        [X],
        outputs=outputs,
        domain="ai.onnx.ml",
        name=name,
        post_transform="NONE",
        **attrs,  # type: ignore
        **label_kwargs,
    )

    if isinstance(result, str):
        return result, result
    return result[0], result[1]


def _sklearn_decision_tree_classifier_v5(
    g: GraphBuilder,
    sts: Dict,
    outputs: List[str],
    estimator: DecisionTreeClassifier,
    X: str,
    name: str,
    classes,
    n_classes: int,
    tree,
) -> Tuple[str, str]:
    """
    Emits a ``TreeEnsemble`` node (``ai.onnx.ml`` opset 5) for a classifier.

    The ``TreeEnsemble`` output is a float ``[N, n_classes]`` score matrix.
    An ``ArgMax`` + ``Gather`` post-processing step derives the integer (or
    string) class label, mirroring the pattern used by the logistic-regression
    converter.

    :return: tuple ``(label_result_name, proba_result_name)``
    """
    attrs = _extract_tree_attributes_v5(tree, n_classes, is_classifier=True)

    # scores: [N, n_classes] float32 – class probabilities
    scores = g.make_node(
        "TreeEnsemble",
        [X],
        outputs=1,
        domain="ai.onnx.ml",
        name=f"{name}_te",
        post_transform=0,  # NONE
        aggregate_function=1,  # SUM
        **attrs,  # type: ignore
    )
    assert isinstance(scores, str)

    # Rename scores to the desired probabilities output name.
    proba = g.op.Identity(scores, name=f"{name}_proba", outputs=outputs[1:])
    assert isinstance(proba, str)

    # Derive the predicted label via ArgMax over the class axis.
    label_idx = g.op.ArgMax(scores, axis=1, keepdims=0, name=f"{name}_argmax")
    label_idx_cast = g.op.Cast(label_idx, to=onnx.TensorProto.INT64, name=f"{name}_cast")

    if np.issubdtype(classes.dtype, np.integer):  # type: ignore
        classes_arr = classes.astype(np.int64)
        label = g.op.Gather(
            classes_arr,
            label_idx_cast,
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
            label_idx_cast,
            axis=0,
            name=f"{name}_label_string",
            outputs=outputs[:1],
        )
        assert isinstance(label, str)
        if not sts:
            g.set_type(label, onnx.TensorProto.STRING)

    return label, proba


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

    When ``ai.onnx.ml`` opset 5 (or later) is active in the graph builder
    the unified ``TreeEnsemble`` operator is used; otherwise the legacy
    ``TreeEnsembleRegressor`` operator is emitted.

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

    ml_opset = _get_ml_opset(g)
    tree = estimator.tree_

    if ml_opset >= 5:
        return _sklearn_decision_tree_regressor_v5(g, sts, outputs, estimator, X, name, tree)

    # Legacy path: TreeEnsembleRegressor (ai.onnx.ml opset <= 4)
    attrs = _extract_tree_attributes(tree, n_classes=1, is_classifier=False)

    result = g.make_node(
        "TreeEnsembleRegressor",
        [X],
        outputs=outputs,
        domain="ai.onnx.ml",
        name=name,
        n_targets=1,
        post_transform="NONE",
        **attrs,  # type: ignore
    )

    return result if isinstance(result, str) else result[0]


def _sklearn_decision_tree_regressor_v5(
    g: GraphBuilder,
    sts: Dict,
    outputs: List[str],
    estimator: DecisionTreeRegressor,
    X: str,
    name: str,
    tree,
) -> str:
    """
    Emits a ``TreeEnsemble`` node (``ai.onnx.ml`` opset 5) for a regressor.

    The output is a float ``[N, 1]`` tensor of predictions, matching the
    shape produced by the legacy ``TreeEnsembleRegressor``.

    :return: output tensor name
    """
    attrs = _extract_tree_attributes_v5(tree, n_classes=1, is_classifier=False)

    result = g.make_node(
        "TreeEnsemble",
        [X],
        outputs=outputs,
        domain="ai.onnx.ml",
        name=f"{name}_te",
        post_transform=0,  # NONE
        aggregate_function=1,  # SUM
        **attrs,  # type: ignore
    )

    return result if isinstance(result, str) else result[0]
