from typing import Dict, List, Tuple, Union

import numpy as np
import onnx
import onnx.helper as oh
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length

from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter

_LEAF = -1  # sklearn's TREE_LEAF constant

# Mode encoding for the TreeEnsemble operator (ai.onnx.ml opset 5)
_NODE_MODE_LEQ = np.uint8(0)


def _compute_node_depths(tree) -> np.ndarray:
    """
    Computes the depth of every node in a sklearn tree (root = depth 1).

    Depth 1 is used for the root so that the path length contribution for a
    leaf at depth *d* equals ``d - 1 + c(n_leaf)``, which matches sklearn's
    formula: ``node_indicator.sum(axis=1) - 1 + c(n_leaf)`` (where the
    indicator sum counts all nodes on the path including root and leaf).

    :param tree: ``estimator.tree_`` attribute
    :return: float64 array of shape ``(node_count,)`` with per-node depths
    """
    depths = np.zeros(tree.node_count, dtype=np.float64)
    stack = [(0, 1)]
    while stack:
        node_id, depth = stack.pop()
        depths[node_id] = depth
        if tree.children_left[node_id] != _LEAF:
            stack.append((int(tree.children_left[node_id]), depth + 1))
            stack.append((int(tree.children_right[node_id]), depth + 1))
    return depths


def _extract_iforest_attributes_legacy(
    estimators: list,
    estimators_features: list,
    n_estimators: int,
) -> dict:
    """
    Extracts combined tree attributes for all isolation trees, encoding each
    leaf with the precomputed path-length contribution used by
    :meth:`sklearn.ensemble.IsolationForest.score_samples`.

    For each leaf node the stored weight is::

        depth(leaf) + _average_path_length(n_node_samples[leaf]) - 1

    Feature indices are remapped from the per-tree local indices to
    absolute indices in the original feature space using
    ``estimators_features_``.

    When a ``TreeEnsembleRegressor`` is built with
    ``aggregate_function="AVERAGE"`` the output equals::

        (1 / n_estimators) * sum_over_trees( depth(leaf) + c(n_leaf) - 1 )

    which is exactly the *average* isolation-path length needed to compute
    the anomaly score.

    :param estimators: ``IsolationForest.estimators_``
    :param estimators_features: ``IsolationForest.estimators_features_``
    :param n_estimators: ``IsolationForest.n_estimators``
    :return: dict of arrays ready to be passed to ``make_node`` as
        ``TreeEnsembleRegressor`` attributes
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

    for tree_id, (base_estimator, features) in enumerate(zip(estimators, estimators_features)):
        tree = base_estimator.tree_
        n_nodes = tree.node_count
        feature = tree.feature
        threshold = tree.threshold.astype(np.float32)
        children_left = tree.children_left
        children_right = tree.children_right
        n_node_samples = tree.n_node_samples

        depths = _compute_node_depths(tree)

        for node_id in range(n_nodes):
            all_nodeids.append(node_id)
            all_treeids.append(tree_id)
            all_hitrates.append(1.0)
            all_mvt.append(0)

            left = children_left[node_id]
            right = children_right[node_id]

            if left == _LEAF:
                # Leaf: store precomputed path-length contribution.
                all_featureids.append(0)
                all_values.append(0.0)
                all_modes.append("LEAF")
                all_truenodeids.append(0)
                all_falsenodeids.append(0)

                leaf_val = float(
                    depths[node_id]
                    + _average_path_length([int(n_node_samples[node_id])])[0]
                    - 1.0
                )
                all_target_nodeids.append(node_id)
                all_target_treeids.append(tree_id)
                all_target_ids.append(0)
                all_target_weights.append(leaf_val)
            else:
                # Internal node: remap local feature index to global one.
                remapped_feat = int(features[int(feature[node_id])])
                all_featureids.append(remapped_feat)
                all_values.append(float(threshold[node_id]))
                all_modes.append("BRANCH_LEQ")
                all_truenodeids.append(int(left))
                all_falsenodeids.append(int(right))

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


def _extract_iforest_attributes_v5(
    estimators: list,
    estimators_features: list,
    n_estimators: int,
    itype: int,
) -> dict:
    """
    Extracts combined tree attributes for all isolation trees in the
    ``ai.onnx.ml`` opset-5 ``TreeEnsemble`` format.

    Leaf weights are the precomputed path-length contributions divided by
    ``n_estimators`` so that the ``SUM`` aggregate yields the *average*
    isolation-path length::

        leaf_weight = (depth(leaf) + c(n_leaf) - 1) / n_estimators

    Feature indices are remapped from per-tree local indices to absolute
    indices in the original feature space via ``estimators_features_``.

    :param estimators: ``IsolationForest.estimators_``
    :param estimators_features: ``IsolationForest.estimators_features_``
    :param n_estimators: ``IsolationForest.n_estimators``
    :param itype: ONNX tensor type for ``nodes_splits`` / ``leaf_weights``
    :return: dict of attributes ready to be passed to the ``TreeEnsemble``
        ``make_node`` call
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

    cumulative_internal_offset = 0
    cumulative_leaf_offset = 0

    for base_estimator, features in zip(estimators, estimators_features):
        tree = base_estimator.tree_
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold.astype(dtype)
        n_node_samples = tree.n_node_samples
        n_nodes = tree.node_count

        depths = _compute_node_depths(tree)

        # Partition nodes into internal vs leaf.
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
        effective_n_internal = max(n_internal, 1)

        # tree_roots: one virtual tree per estimator.
        all_tree_roots.append(cumulative_internal_offset)

        # nodes_* arrays
        if not internal_nodes:
            # Degenerate tree: one dummy internal node pointing to the single leaf.
            all_nodes_featureids.append(0)
            all_nodes_splits.append(0.0)
            all_nodes_modes.append(int(_NODE_MODE_LEQ))
            all_nodes_truenodeids.append(cumulative_leaf_offset)
            all_nodes_trueleafs.append(1)
            all_nodes_falsenodeids.append(cumulative_leaf_offset)
            all_nodes_falseleafs.append(1)
        else:
            node_offset = cumulative_internal_offset
            leaf_offset = cumulative_leaf_offset

            for nid in internal_nodes:
                left = int(children_left[nid])
                right = int(children_right[nid])

                # Remap local feature index to global index.
                remapped_feat = int(features[int(feature[nid])])
                all_nodes_featureids.append(remapped_feat)
                all_nodes_splits.append(float(threshold[nid]))
                all_nodes_modes.append(int(_NODE_MODE_LEQ))

                # In sklearn, a node is a leaf iff children_left[node] == TREE_LEAF.
                left_is_leaf = children_left[left] == _LEAF
                if left_is_leaf:
                    all_nodes_truenodeids.append(leaf_offset + leaf_node_to_idx[left])
                    all_nodes_trueleafs.append(1)
                else:
                    all_nodes_truenodeids.append(node_offset + internal_node_to_idx[left])
                    all_nodes_trueleafs.append(0)

                right_is_leaf = children_left[right] == _LEAF
                if right_is_leaf:
                    all_nodes_falsenodeids.append(leaf_offset + leaf_node_to_idx[right])
                    all_nodes_falseleafs.append(1)
                else:
                    all_nodes_falsenodeids.append(node_offset + internal_node_to_idx[right])
                    all_nodes_falseleafs.append(0)

        # leaf_* arrays: precomputed path-length contribution / n_estimators
        if not leaf_nodes:
            # Degenerate: single leaf from the root node.
            leaf_val = float(depths[0] + _average_path_length([int(n_node_samples[0])])[0] - 1.0)
            all_leaf_targetids.append(0)
            all_leaf_weights.append(leaf_val / n_estimators)
        else:
            for nid in leaf_nodes:
                leaf_val = float(
                    depths[nid] + _average_path_length([int(n_node_samples[nid])])[0] - 1.0
                )
                all_leaf_targetids.append(0)
                all_leaf_weights.append(leaf_val / n_estimators)

        cumulative_internal_offset += effective_n_internal
        cumulative_leaf_offset += max(n_leaves, 1)

    nodes_splits_tensor = oh.make_tensor(
        "nodes_splits",
        itype,
        (len(all_nodes_splits),),
        np.array(all_nodes_splits, dtype=dtype),
    )
    nodes_modes_tensor = oh.make_tensor(
        "nodes_modes",
        onnx.TensorProto.UINT8,
        (len(all_nodes_modes),),
        np.array(all_nodes_modes, dtype=np.uint8),
    )
    leaf_weights_tensor = oh.make_tensor(
        "leaf_weights",
        itype,
        (len(all_leaf_weights),),
        np.array(all_leaf_weights, dtype=dtype),
    )

    return dict(
        tree_roots=all_tree_roots,
        n_targets=1,
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


@register_sklearn_converter((IsolationForest,))
def sklearn_isolation_forest(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: IsolationForest,
    X: str,
    name: str = "isolation_forest",
) -> Union[str, Tuple[str, str]]:
    """
    Converts a :class:`sklearn.ensemble.IsolationForest` into ONNX.

    **Algorithm overview**

    Each isolation tree assigns a sample a *path-length contribution*
    equal to ``depth(leaf) + c(n_leaf) - 1``, where ``c(n)`` is
    :func:`sklearn.ensemble._iforest._average_path_length`.  These
    contributions are averaged across trees to give an *average isolation
    path length* ``avg_depth``.  The anomaly score is then:

    .. code-block:: text

        score_samples(X) = -2^(-avg_depth / c(max_samples))
        decision_function(X) = score_samples(X) - offset_
        predict(X) = 1 if decision_function(X) >= 0 else -1

    **ONNX graph structure**

    The path-length contributions are precomputed at conversion time and
    stored as leaf weights in a ``TreeEnsembleRegressor`` (opset ≤ 4) or
    ``TreeEnsemble`` (opset 5+) node.  Feature indices are remapped from
    per-tree local indices to the original feature space via
    ``estimators_features_``.

    .. code-block:: text

        X ──TreeEnsembleRegressor(AVERAGE) / TreeEnsemble(SUM)──► avg_depth
                Reshape──► avg_depth (N,)
                Mul(-log(2)/c(max_samples))──► exponent (N,)
                Exp──► score (N,)          [= 2^(-avg/c)]
                Neg──► score_samples (N,)
                Sub(offset_)──► decision (N,)
                Where(>=0, 1, -1)──► label (N,)

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names; two entries
        ``(label, scores)`` or one entry ``(label,)``
    :param estimator: a fitted :class:`~sklearn.ensemble.IsolationForest`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: label tensor name, or tuple ``(label, scores)``
    """
    assert isinstance(
        estimator, IsolationForest
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    ml_opset = g.get_opset("ai.onnx.ml")
    n_estimators = estimator.n_estimators
    estimators = estimator.estimators_
    estimators_features = estimator.estimators_features_
    max_samples = int(estimator.max_samples_)
    offset = float(estimator.offset_)

    # Normalization constant: c(max_samples)
    normalization = float(_average_path_length([max_samples])[0])

    avg_depth_raw = g.unique_name(f"{name}_avg_depth_raw")

    if ml_opset >= 5:
        # v5 path: TreeEnsemble (SUM aggregation, weights pre-divided by n_estimators)
        # The output dtype matches the input dtype for TreeEnsemble v5.
        attrs = _extract_iforest_attributes_v5(
            estimators, estimators_features, n_estimators, itype
        )
        g.make_node(
            "TreeEnsemble",
            [X],
            outputs=[avg_depth_raw],
            domain="ai.onnx.ml",
            name=f"{name}_tree",
            post_transform=0,  # NONE
            aggregate_function=1,  # SUM
            **attrs,  # type: ignore
        )
        np_dtype = np.float64 if itype == onnx.TensorProto.DOUBLE else np.float32
    else:
        # Legacy path: TreeEnsembleRegressor (AVERAGE aggregation, float32 output)
        attrs = _extract_iforest_attributes_legacy(estimators, estimators_features, n_estimators)
        g.make_node(
            "TreeEnsembleRegressor",
            [X],
            outputs=[avg_depth_raw],
            domain="ai.onnx.ml",
            name=f"{name}_tree",
            n_targets=1,
            aggregate_function="AVERAGE",
            post_transform="NONE",
            **attrs,  # type: ignore
        )
        np_dtype = np.float32

    # Flatten (N, 1) → (N,)
    avg_depth_flat = g.op.Reshape(
        avg_depth_raw,
        np.array([-1], dtype=np.int64),
        name=f"{name}_reshape",
    )

    # Cast to the input dtype when float64 is requested (legacy tree always outputs float32).
    if ml_opset < 5 and itype == onnx.TensorProto.DOUBLE:
        avg_depth = g.op.Cast(
            avg_depth_flat,
            to=onnx.TensorProto.DOUBLE,
            name=f"{name}_cast_f64",
        )
        np_dtype = np.float64
    else:
        avg_depth = avg_depth_flat

    # score = 2^(-avg_depth / normalization) = exp(avg_depth * (-log(2) / normalization))
    scale = np_dtype(-np.log(2.0) / normalization)
    exponent = g.op.Mul(
        avg_depth,
        np.array([scale], dtype=np_dtype),
        name=f"{name}_scale",
    )
    score = g.op.Exp(exponent, name=f"{name}_exp")

    # score_samples = -score
    score_samples = g.op.Neg(score, name=f"{name}_score_samples")

    # decision_function = score_samples - offset_
    offset_arr = np.array([offset], dtype=np_dtype)
    decision = g.op.Sub(
        score_samples,
        offset_arr,
        name=f"{name}_decision",
    )

    # label = 1 if decision_function >= 0 else -1
    zero_arr = np.array([np_dtype(0)], dtype=np_dtype)
    is_inlier = g.op.GreaterOrEqual(decision, zero_arr, name=f"{name}_ge")
    label = g.op.Where(
        is_inlier,
        np.array([1], dtype=np.int64),
        np.array([-1], dtype=np.int64),
        name=f"{name}_label",
        outputs=outputs[:1],
    )
    assert isinstance(label, str)

    emit_scores = len(outputs) > 1
    if emit_scores:
        scores_out = g.op.Identity(decision, name=f"{name}_scores", outputs=outputs[1:])
        assert isinstance(scores_out, str)
        return label, scores_out

    return label
