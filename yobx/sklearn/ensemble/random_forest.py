from typing import Tuple, Dict, List
import numpy as np
import onnx
import onnx.helper as oh
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from ..register import register_sklearn_converter
from ..tree.decision_tree import (
    _LEAF,
    _NODE_MODE_LEQ,
    _emit_decision_path_for_tree,
    _emit_decision_leaf_for_tree,
)


def _extract_forest_attributes_legacy(
    estimators: list, n_classes: int, is_classifier: bool, n_estimators: int
):
    """
    Extracts combined attributes for all trees in a forest for use with
    the legacy ``TreeEnsembleClassifier`` / ``TreeEnsembleRegressor``
    operators (``ai.onnx.ml`` opset <= 4).

    Each tree in the forest is assigned a unique ``tree_id``.
    For classifiers, leaf weights are divided by ``n_estimators`` so that
    the ``TreeEnsembleClassifier`` output (with ``post_transform="NONE"``)
    is the averaged class-probability vector.  For regressors, raw leaf
    values are stored and ``aggregate_function="AVERAGE"`` is used.

    :param estimators: list of fitted base estimators (``estimator.estimators_``)
    :param n_classes: number of output classes (classifiers) or 1 (regressors)
    :param is_classifier: True for classification, False for regression
    :param n_estimators: total number of trees in the forest
    :return: dict of ONNX attribute arrays ready to be passed to ``make_node``
    """
    all_featureids: List[int] = []
    all_values: List[float] = []
    all_modes: List[str] = []
    all_truenodeids: List[int] = []
    all_falsenodeids: List[int] = []
    all_nodeids: List[int] = []
    all_treeids: List[int] = []
    all_hitrates: List[float] = []
    all_mvt: List[int] = []

    all_target_nodeids: List[int] = []
    all_target_treeids: List[int] = []
    all_target_ids: List[int] = []
    all_target_weights: List[float] = []

    for tree_id, base_estimator in enumerate(estimators):
        tree = base_estimator.tree_
        n_nodes = tree.node_count
        feature = tree.feature
        threshold = tree.threshold.astype(np.float32)
        children_left = tree.children_left
        children_right = tree.children_right
        value = tree.value  # shape: (n_nodes, n_outputs, max_n_classes)

        for node_id in range(n_nodes):
            all_nodeids.append(node_id)
            all_treeids.append(tree_id)
            all_hitrates.append(1.0)
            all_mvt.append(0)

            left = children_left[node_id]
            right = children_right[node_id]

            if left == _LEAF:
                # Leaf node
                all_featureids.append(0)
                all_values.append(0.0)
                all_modes.append("LEAF")
                all_truenodeids.append(0)
                all_falsenodeids.append(0)

                node_value = value[node_id, 0]  # shape: (max_n_classes,)
                if is_classifier:
                    total = float(node_value.sum())
                    for class_idx in range(n_classes):
                        prob = (
                            float(node_value[class_idx]) / total / n_estimators
                            if total > 0.0
                            else 0.0
                        )
                        all_target_nodeids.append(node_id)
                        all_target_treeids.append(tree_id)
                        all_target_ids.append(class_idx)
                        all_target_weights.append(prob)
                else:
                    all_target_nodeids.append(node_id)
                    all_target_treeids.append(tree_id)
                    all_target_ids.append(0)
                    all_target_weights.append(float(node_value[0]))
            else:
                # Internal node
                all_featureids.append(int(feature[node_id]))
                all_values.append(float(threshold[node_id]))
                all_modes.append("BRANCH_LEQ")
                all_truenodeids.append(int(left))
                all_falsenodeids.append(int(right))

    if is_classifier:
        return dict(
            nodes_featureids=all_featureids,
            nodes_values=all_values,
            nodes_modes=all_modes,
            nodes_truenodeids=all_truenodeids,
            nodes_falsenodeids=all_falsenodeids,
            nodes_nodeids=all_nodeids,
            nodes_treeids=all_treeids,
            nodes_hitrates=all_hitrates,
            nodes_missing_value_tracks_true=all_mvt,
            class_nodeids=all_target_nodeids,
            class_treeids=all_target_treeids,
            class_ids=all_target_ids,
            class_weights=all_target_weights,
        )
    return dict(
        nodes_featureids=all_featureids,
        nodes_values=all_values,
        nodes_modes=all_modes,
        nodes_truenodeids=all_truenodeids,
        nodes_falsenodeids=all_falsenodeids,
        nodes_nodeids=all_nodeids,
        nodes_treeids=all_treeids,
        nodes_hitrates=all_hitrates,
        nodes_missing_value_tracks_true=all_mvt,
        target_nodeids=all_target_nodeids,
        target_treeids=all_target_treeids,
        target_ids=all_target_ids,
        target_weights=all_target_weights,
    )


def _append_leaf_entry_v5(
    all_leaf_targetids: List[int],
    all_leaf_weights: List[float],
    node_value,
    tree_idx: int,
    is_classifier: bool,
    n_estimators: int,
) -> None:
    """Appends one leaf entry to the flat ``leaf_*`` arrays for the v5 operator.

    For classifiers, the weight is ``prob(class=tree_idx) / n_estimators``.
    For regressors, the weight is ``raw_value / n_estimators``.
    """
    if is_classifier:
        total = float(node_value.sum())
        prob = float(node_value[tree_idx]) / total / n_estimators if total > 0.0 else 0.0
        all_leaf_targetids.append(tree_idx)
        all_leaf_weights.append(prob)
    else:
        all_leaf_targetids.append(0)
        all_leaf_weights.append(float(node_value[0]) / n_estimators)


def _extract_forest_attributes_v5(
    estimators: list, n_classes: int, is_classifier: bool, n_estimators: int, itype: int
):
    """
    Extracts combined attributes for all trees in a forest for use with the
    unified ``TreeEnsemble`` operator introduced in ``ai.onnx.ml`` opset 5.

    The approach mirrors the single-tree v5 encoding in
    :func:`~yobx.sklearn.tree.decision_tree._extract_tree_attributes_v5`:

    * **Classifiers** — ``n_classes`` virtual trees per estimator.
      Virtual tree ``est * n_classes + cls`` stores only the probability for
      class ``cls`` (divided by ``n_estimators``).  With
      ``aggregate_function=SUM`` and ``n_targets=n_classes`` the output
      ``[N, n_classes]`` is the averaged class-probability matrix.

    * **Regressors** — one virtual tree per estimator.  Leaf weights are the
      raw predicted values divided by ``n_estimators``; ``aggregate_function=SUM``
      then gives the averaged regression target.

    :param estimators: list of fitted base estimators
    :param n_classes: number of classes (classifiers) or 1 (regressors)
    :param is_classifier: True for classification, False for regression
    :param n_estimators: total number of trees in the forest
    :param itype: onnx type
    :return: dict of ONNX attributes ready to be passed to ``make_node``
    """
    dtype = tensor_dtype_to_np_dtype(itype)
    all_nodes_featureids: List[int] = []
    all_nodes_splits: List[float] = []
    all_nodes_modes: List[int] = []
    all_nodes_truenodeids: List[int] = []
    all_nodes_trueleafs: List[int] = []
    all_nodes_falsenodeids: List[int] = []
    all_nodes_falseleafs: List[int] = []

    all_leaf_targetids: List[int] = []
    all_leaf_weights: List[float] = []

    all_tree_roots: List[int] = []

    # Cumulative offsets into the flat nodes_* and leaf_* arrays.
    cumulative_internal_offset = 0
    cumulative_leaf_offset = 0

    for base_estimator in estimators:
        tree = base_estimator.tree_
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold.astype(dtype)
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

        # Each estimator contributes n_trees_per_est virtual trees.
        n_trees_per_est = n_classes if is_classifier else 1

        # ------------------------------------------------------------------ #
        # tree_roots                                                           #
        # ------------------------------------------------------------------ #
        effective_n_internal = max(n_internal, 1)  # 1 dummy node when degenerate
        for tree_idx in range(n_trees_per_est):
            all_tree_roots.append(cumulative_internal_offset + tree_idx * effective_n_internal)

        # ------------------------------------------------------------------ #
        # nodes_* arrays                                                       #
        # ------------------------------------------------------------------ #
        if not internal_nodes:
            # Degenerate tree: one dummy internal node per virtual tree.
            for tree_idx in range(n_trees_per_est):
                this_leaf_offset = cumulative_leaf_offset + tree_idx * 1
                all_nodes_featureids.append(0)
                all_nodes_splits.append(0.0)
                all_nodes_modes.append(int(_NODE_MODE_LEQ))
                all_nodes_truenodeids.append(this_leaf_offset)
                all_nodes_trueleafs.append(1)
                all_nodes_falsenodeids.append(this_leaf_offset)
                all_nodes_falseleafs.append(1)
        else:
            for tree_idx in range(n_trees_per_est):
                node_offset = cumulative_internal_offset + tree_idx * n_internal
                leaf_offset = cumulative_leaf_offset + tree_idx * n_leaves

                for nid in internal_nodes:
                    left = int(children_left[nid])
                    right = int(children_right[nid])

                    all_nodes_featureids.append(int(feature[nid]))
                    all_nodes_splits.append(float(threshold[nid]))
                    all_nodes_modes.append(int(_NODE_MODE_LEQ))

                    # True branch → left child (sklearn: feature <= threshold)
                    left_is_leaf = children_left[left] == _LEAF
                    if left_is_leaf:
                        all_nodes_truenodeids.append(leaf_offset + leaf_node_to_idx[left])
                        all_nodes_trueleafs.append(1)
                    else:
                        all_nodes_truenodeids.append(node_offset + internal_node_to_idx[left])
                        all_nodes_trueleafs.append(0)

                    # False branch → right child (sklearn: feature > threshold)
                    right_is_leaf = children_left[right] == _LEAF
                    if right_is_leaf:
                        all_nodes_falsenodeids.append(leaf_offset + leaf_node_to_idx[right])
                        all_nodes_falseleafs.append(1)
                    else:
                        all_nodes_falsenodeids.append(node_offset + internal_node_to_idx[right])
                        all_nodes_falseleafs.append(0)

        # ------------------------------------------------------------------ #
        # leaf_* arrays                                                        #
        # ------------------------------------------------------------------ #
        if not leaf_nodes:
            # Degenerate tree: single leaf derived from the root node.
            for tree_idx in range(n_trees_per_est):
                _append_leaf_entry_v5(
                    all_leaf_targetids,
                    all_leaf_weights,
                    value[0, 0],
                    tree_idx,
                    is_classifier,
                    n_estimators,
                )
        else:
            for tree_idx in range(n_trees_per_est):
                for nid in leaf_nodes:
                    _append_leaf_entry_v5(
                        all_leaf_targetids,
                        all_leaf_weights,
                        value[nid, 0],
                        tree_idx,
                        is_classifier,
                        n_estimators,
                    )

        # ------------------------------------------------------------------ #
        # Advance cumulative offsets.                                          #
        # ------------------------------------------------------------------ #
        cumulative_internal_offset += n_trees_per_est * max(n_internal, 1)
        cumulative_leaf_offset += n_trees_per_est * max(n_leaves, 1)

    # Pack tensor attributes required by the opset-5 operator.
    # nodes_splits and leaf_weights use the same float type as the input.
    nodes_splits_tensor = oh.make_tensor(
        "nodes_splits", itype, (len(all_nodes_splits),), np.array(all_nodes_splits, dtype=dtype)
    )
    nodes_modes_tensor = oh.make_tensor(
        "nodes_modes",
        onnx.TensorProto.UINT8,
        (len(all_nodes_modes),),
        np.array(all_nodes_modes, dtype=np.uint8),
    )
    leaf_weights_tensor = oh.make_tensor(
        "leaf_weights", itype, (len(all_leaf_weights),), np.array(all_leaf_weights, dtype=dtype)
    )

    return dict(
        tree_roots=all_tree_roots,
        n_targets=n_classes if is_classifier else 1,
        nodes_featureids=all_nodes_featureids,
        nodes_splits=nodes_splits_tensor,
        nodes_modes=nodes_modes_tensor,
        nodes_truenodeids=all_nodes_truenodeids,
        nodes_trueleafs=all_nodes_trueleafs,
        nodes_falsenodeids=all_nodes_falsenodeids,
        nodes_falseleafs=all_nodes_falseleafs,
        leaf_targetids=all_leaf_targetids,
        leaf_weights=leaf_weights_tensor,
    )


def _emit_decision_path_for_estimators(
    g: GraphBuilderExtendedProtocol, estimators: list, X: str, output_name: str, name: str
) -> str:
    """Emits ONNX nodes to compute the ``decision_path`` output for an ensemble.

    Computes a per-tree path string for each estimator and concatenates them
    along axis 1 to produce a ``(N, n_estimators)`` string tensor.

    :param g: graph builder
    :param estimators: list of fitted base estimators (``estimator.estimators_``)
    :param X: input tensor name
    :param output_name: desired output tensor name
    :param name: prefix for the internal node names
    :return: *output_name*
    """
    per_tree_paths: List[str] = []
    for i, base_est in enumerate(estimators):
        path_i = g.unique_name(f"{name}_t{i}_path")
        _emit_decision_path_for_tree(g, base_est.tree_, X, path_i, name=f"{name}_t{i}")
        per_tree_paths.append(path_i)
    g.make_node(
        "Concat", per_tree_paths, outputs=[output_name], name=f"{name}_path_concat", axis=1
    )
    return output_name


def _emit_decision_leaf_for_estimators(
    g: GraphBuilderExtendedProtocol, estimators: list, X: str, output_name: str, name: str
) -> str:
    """Emit ONNX nodes to compute the ``decision_leaf`` output for an ensemble.

    Computes a per-tree leaf index for each estimator and concatenates them
    along axis 1 to produce a ``(N, n_estimators)`` int64 tensor.

    :param g: graph builder
    :param estimators: list of fitted base estimators (``estimator.estimators_``)
    :param X: input tensor name
    :param output_name: desired output tensor name
    :param name: prefix for the internal node names
    :return: *output_name*
    """
    per_tree_leaves: List[str] = []
    for i, base_est in enumerate(estimators):
        leaf_i = g.unique_name(f"{name}_t{i}_leaf")
        _emit_decision_leaf_for_tree(g, base_est.tree_, X, leaf_i, name=f"{name}_t{i}")
        per_tree_leaves.append(leaf_i)
    g.make_node(
        "Concat", per_tree_leaves, outputs=[output_name], name=f"{name}_leaf_concat", axis=1
    )
    return output_name


@register_sklearn_converter((RandomForestClassifier,))
def sklearn_random_forest_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: RandomForestClassifier,
    X: str,
    name: str = "random_forest_classifier",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.ensemble.RandomForestClassifier` into ONNX.

    When ``ai.onnx.ml`` opset 5 (or later) is active in the graph builder
    the unified ``TreeEnsemble`` operator is used; otherwise the legacy
    ``TreeEnsembleClassifier`` operator is emitted.

    The forest is encoded as a single multi-tree ONNX node where each
    estimator's leaf weights are divided by ``n_estimators`` so that the
    ``SUM`` aggregate (or ``NONE`` post-transform in the legacy path) yields
    the averaged class-probability vector used by :meth:`predict_proba`.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``RandomForestClassifier``
    :param outputs: desired output names (label, probabilities)
    :param X: input tensor name
    :param name: prefix for node names added to the graph
    :return: tuple ``(label_result_name, proba_result_name)``
    """
    assert isinstance(
        estimator, RandomForestClassifier
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
    if g.convert_options.has("decision_path", estimator):
        assert len(outputs) > extra_idx, f"Missing output for decision_path in {outputs}"
        _emit_decision_path_for_estimators(g, estimators, X, outputs[extra_idx], f"{name}_dp")
        extra_idx += 1
    if g.convert_options.has("decision_leaf", estimator):
        assert len(outputs) > extra_idx, f"Missing output for decision_path in {outputs}"
        _emit_decision_leaf_for_estimators(g, estimators, X, outputs[extra_idx], f"{name}_dl")
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def _sklearn_random_forest_classifier_v5(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: RandomForestClassifier,
    X: str,
    name: str,
    classes,
    n_classes: int,
    n_estimators: int,
    estimators: list,
    itype: int,
) -> Tuple[str, str]:
    """
    Emits a ``TreeEnsemble`` node (``ai.onnx.ml`` opset 5) for a
    :class:`sklearn.ensemble.RandomForestClassifier`.

    Mirrors :func:`~yobx.sklearn.tree.decision_tree._sklearn_decision_tree_classifier_v5`
    but encodes all ``n_estimators`` trees into a single node.

    :param dtype: numpy float dtype for ``nodes_splits`` / ``leaf_weights``
        tensors; uses ``np.float32`` when ``None``
    :return: tuple ``(label_result_name, proba_result_name)``
    """
    attrs = _extract_forest_attributes_v5(
        estimators, n_classes, is_classifier=True, n_estimators=n_estimators, itype=itype
    )

    # scores: [N, n_classes] float32 - averaged class probabilities
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
    proba = g.op.Identity(scores, name=f"{name}_proba", outputs=outputs[1:2])
    assert isinstance(proba, str)

    # Derive the predicted label via ArgMax over the class axis.
    label_idx = g.op.ArgMax(scores, axis=1, keepdims=0, name=f"{name}_argmax")
    label_idx_cast = g.op.Cast(label_idx, to=onnx.TensorProto.INT64, name=f"{name}_cast")

    if np.issubdtype(classes.dtype, np.integer):  # type: ignore
        classes_arr = classes.astype(np.int64)
        label = g.op.Gather(
            classes_arr, label_idx_cast, axis=0, name=f"{name}_label", outputs=outputs[:1]
        )
        assert isinstance(label, str)
        if not sts:
            g.set_type(label, onnx.TensorProto.INT64)
    else:
        classes_arr = np.array(classes.astype(str))
        label = g.op.Gather(
            classes_arr, label_idx_cast, axis=0, name=f"{name}_label_string", outputs=outputs[:1]
        )
        assert isinstance(label, str)
        if not sts:
            g.set_type(label, onnx.TensorProto.STRING)

    extra_idx = 2
    if g.convert_options.has("decision_path", estimator):
        assert len(outputs) > extra_idx, f"Missing output for decision_path in {outputs}"
        _emit_decision_path_for_estimators(g, estimators, X, outputs[extra_idx], f"{name}_dp")
        extra_idx += 1
    if g.convert_options.has("decision_leaf", estimator):
        assert len(outputs) > extra_idx, f"Missing output for decision_path in {outputs}"
        _emit_decision_leaf_for_estimators(g, estimators, X, outputs[extra_idx], f"{name}_dl")
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


@register_sklearn_converter((RandomForestRegressor,))
def sklearn_random_forest_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: RandomForestRegressor,
    X: str,
    name: str = "random_forest_regressor",
) -> str:
    """
    Converts a :class:`sklearn.ensemble.RandomForestRegressor` into ONNX.

    When ``ai.onnx.ml`` opset 5 (or later) is active in the graph builder
    the unified ``TreeEnsemble`` operator is used (leaf weights pre-divided
    by ``n_estimators`` so that ``SUM`` aggregation yields the average);
    otherwise the legacy ``TreeEnsembleRegressor`` operator is emitted with
    ``aggregate_function="AVERAGE"``.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``RandomForestRegressor``
    :param outputs: desired output names (predictions)
    :param X: input tensor name
    :param name: prefix for node names added to the graph
    :return: output tensor name
    """
    assert isinstance(
        estimator, RandomForestRegressor
    ), f"Unexpected type {type(estimator)} for estimator."

    ml_opset = g.get_opset("ai.onnx.ml")
    n_estimators = estimator.n_estimators
    estimators = estimator.estimators_

    if ml_opset >= 5:
        return _sklearn_random_forest_regressor_v5(
            g, sts, outputs, estimator, X, name, n_estimators, estimators, itype=g.get_type(X)
        )

    # Detect float64 input so we can cast the output back to double after the
    # tree node (TreeEnsembleRegressor / TreeEnsemble always output float32).
    itype = g.get_type(X)

    # When a cast is needed, direct the tree node into a temporary intermediate name.
    tree_outputs = [f"{outputs[0]}_tree_out"]

    # Legacy path: TreeEnsembleRegressor (ai.onnx.ml opset <= 4)
    attrs = _extract_forest_attributes_legacy(
        estimators, n_classes=1, is_classifier=False, n_estimators=n_estimators
    )

    node_result = g.make_node(
        "TreeEnsembleRegressor",
        [X],
        outputs=tree_outputs,
        domain="ai.onnx.ml",
        name=name,
        n_targets=1,
        aggregate_function="AVERAGE",
        post_transform="NONE",
        **attrs,  # type: ignore
    )
    tree_result = node_result if isinstance(node_result, str) else node_result[0]

    # Cast float32 output back to float64 to match the input dtype.
    g.make_node("Cast", [tree_result], outputs=outputs[:1], name=f"{name}_cast_f64", to=itype)

    extra_idx = 1
    if g.convert_options.has("decision_path", estimator):
        assert len(outputs) > extra_idx, f"Missing output for decision_path in {outputs}"
        _emit_decision_path_for_estimators(g, estimators, X, outputs[extra_idx], f"{name}_dp")
        extra_idx += 1
    if g.convert_options.has("decision_leaf", estimator):
        assert len(outputs) > extra_idx, f"Missing output for decision_path in {outputs}"
        _emit_decision_leaf_for_estimators(g, estimators, X, outputs[extra_idx], f"{name}_dl")
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def _sklearn_random_forest_regressor_v5(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: RandomForestRegressor,
    X: str,
    name: str,
    n_estimators: int,
    estimators: list,
    itype: int,
) -> str:
    """
    Emits a ``TreeEnsemble`` node (``ai.onnx.ml`` opset 5) for a
    :class:`sklearn.ensemble.RandomForestRegressor`.

    Leaf weights are pre-divided by ``n_estimators`` so that the ``SUM``
    aggregate gives the averaged prediction.

    :param dtype: numpy float dtype for ``nodes_splits`` / ``leaf_weights``
        tensors; uses ``np.float32`` when ``None``
    :return: output tensor name
    """
    attrs = _extract_forest_attributes_v5(
        estimators, n_classes=1, is_classifier=False, n_estimators=n_estimators, itype=itype
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
    if g.convert_options.has("decision_path", estimator) and len(outputs) > extra_idx:
        assert len(outputs) > extra_idx, f"Missing output for decision_path in {outputs}"
        _emit_decision_path_for_estimators(g, estimators, X, outputs[extra_idx], f"{name}_dp")
        extra_idx += 1
    if g.convert_options.has("decision_leaf", estimator) and len(outputs) > extra_idx:
        assert len(outputs) > extra_idx, f"Missing output for decision_leaf in {outputs}"
        _emit_decision_leaf_for_estimators(g, estimators, X, outputs[extra_idx], f"{name}_dl")
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


@register_sklearn_converter((ExtraTreesClassifier,))
def sklearn_extra_trees_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: ExtraTreesClassifier,
    X: str,
    name: str = "extra_trees_classifier",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.ensemble.ExtraTreesClassifier` into ONNX.

    Extra Trees share the same internal tree structure as Random Forests
    (a list of fitted base estimators with a ``tree_`` attribute).  This
    converter therefore delegates entirely to the same attribute-extraction
    helpers used by :func:`sklearn_random_forest_classifier`.

    When ``ai.onnx.ml`` opset 5 (or later) is active in the graph builder
    the unified ``TreeEnsemble`` operator is used; otherwise the legacy
    ``TreeEnsembleClassifier`` operator is emitted.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``ExtraTreesClassifier``
    :param outputs: desired output names (label, probabilities)
    :param X: input tensor name
    :param name: prefix for node names added to the graph
    :return: tuple ``(label_result_name, proba_result_name)``
    """
    assert isinstance(
        estimator, ExtraTreesClassifier
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
    if g.convert_options.has("decision_path", estimator) and len(outputs) > extra_idx:
        assert len(outputs) > extra_idx, f"Missing output for decision_path in {outputs}"
        _emit_decision_path_for_estimators(g, estimators, X, outputs[extra_idx], f"{name}_dp")
        extra_idx += 1
    if g.convert_options.has("decision_leaf", estimator) and len(outputs) > extra_idx:
        assert len(outputs) > extra_idx, f"Missing output for decision_path in {outputs}"
        _emit_decision_leaf_for_estimators(g, estimators, X, outputs[extra_idx], f"{name}_dl")
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


@register_sklearn_converter((ExtraTreesRegressor,))
def sklearn_extra_trees_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: ExtraTreesRegressor,
    X: str,
    name: str = "extra_trees_regressor",
) -> str:
    """
    Converts a :class:`sklearn.ensemble.ExtraTreesRegressor` into ONNX.

    Extra Trees share the same internal tree structure as Random Forests.
    This converter delegates to the same attribute-extraction helpers used
    by :func:`sklearn_random_forest_regressor`.

    When ``ai.onnx.ml`` opset 5 (or later) is active in the graph builder
    the unified ``TreeEnsemble`` operator is used (leaf weights pre-divided
    by ``n_estimators`` so that ``SUM`` aggregation yields the average);
    otherwise the legacy ``TreeEnsembleRegressor`` operator is emitted with
    ``aggregate_function="AVERAGE"``.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``ExtraTreesRegressor``
    :param outputs: desired output names (predictions)
    :param X: input tensor name
    :param name: prefix for node names added to the graph
    :return: output tensor name
    """
    assert isinstance(
        estimator, ExtraTreesRegressor
    ), f"Unexpected type {type(estimator)} for estimator."

    ml_opset = g.get_opset("ai.onnx.ml")
    n_estimators = estimator.n_estimators
    estimators = estimator.estimators_

    if ml_opset >= 5:
        return _sklearn_random_forest_regressor_v5(
            g, sts, outputs, estimator, X, name, n_estimators, estimators, itype=g.get_type(X)
        )

    # Detect float64 input so we can cast the output back to double after the
    # tree node (TreeEnsembleRegressor / TreeEnsemble always output float32).
    itype = g.get_type(X)

    # When a cast is needed, direct the tree node into a temporary intermediate name.
    tree_outputs = [f"{outputs[0]}_tree_out"]

    # Legacy path: TreeEnsembleRegressor (ai.onnx.ml opset <= 4)
    attrs = _extract_forest_attributes_legacy(
        estimators, n_classes=1, is_classifier=False, n_estimators=n_estimators
    )

    node_result = g.make_node(
        "TreeEnsembleRegressor",
        [X],
        outputs=tree_outputs,
        domain="ai.onnx.ml",
        name=name,
        n_targets=1,
        aggregate_function="AVERAGE",
        post_transform="NONE",
        **attrs,  # type: ignore
    )
    tree_result = node_result if isinstance(node_result, str) else node_result[0]

    # Cast float32 output back to float64 to match the input dtype.
    g.make_node("Cast", [tree_result], outputs=outputs[:1], name=f"{name}_cast_f64", to=itype)

    extra_idx = 1
    if g.convert_options.has("decision_path", estimator) and len(outputs) > extra_idx:
        assert len(outputs) > extra_idx, f"Missing output for decision_path in {outputs}"
        _emit_decision_path_for_estimators(g, estimators, X, outputs[extra_idx], f"{name}_dp")
        extra_idx += 1
    if g.convert_options.has("decision_leaf", estimator) and len(outputs) > extra_idx:
        assert len(outputs) > extra_idx, f"Missing output for decision_path in {outputs}"
        _emit_decision_leaf_for_estimators(g, estimators, X, outputs[extra_idx], f"{name}_dl")
    return outputs[0] if len(outputs) == 1 else tuple(outputs)
