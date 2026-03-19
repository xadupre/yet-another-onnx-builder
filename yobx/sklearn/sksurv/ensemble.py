from typing import Dict, List

import numpy as np
import onnx
import onnx.helper as oh
from sksurv.ensemble import RandomSurvivalForest

from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from ..register import register_sklearn_converter
from ..tree.decision_tree import _LEAF, _NODE_MODE_LEQ


def _rsf_leaf_value(tree_value: np.ndarray, node_id: int, is_event_time: np.ndarray) -> float:
    """Returns the scalar risk-score contribution for a leaf node in a :class:`SurvivalTree`.

    The raw leaf value tensor has shape ``(n_unique_times, 2)`` where
    ``[:, 0]`` holds the cumulative hazard function (CHF) values and
    ``[:, 1]`` holds the survival-function values.  The risk-score
    prediction for that leaf equals the sum of CHF values at the observed
    event times::

        leaf_score = tree_.value[node_id, :, 0][is_event_time].sum()

    :param tree_value: ``tree_.value`` array, shape ``(n_nodes, n_unique_times, 2)``
    :param node_id: index of the leaf node
    :param is_event_time: boolean mask of length ``n_unique_times``; ``True``
        where a unique time corresponds to an observed event
    :return: scalar leaf prediction
    """
    return float(tree_value[node_id, :, 0][is_event_time].sum())


def _extract_rsf_attributes_legacy(
    estimators: list, n_estimators: int, is_event_time: np.ndarray
) -> dict:
    """Extracts ``TreeEnsembleRegressor`` attributes for a :class:`RandomSurvivalForest`.

    Mirrors :func:`~yobx.sklearn.ensemble.random_forest._extract_forest_attributes_legacy`
    but computes each leaf weight as the sum of the cumulative hazard function
    (CHF) values at the observed event times — the exact quantity returned by
    :meth:`sksurv.tree.SurvivalTree.predict`.

    Leaf weights are stored as-is; the ``TreeEnsembleRegressor`` node is
    configured with ``aggregate_function="AVERAGE"`` so the ONNX output
    matches :meth:`RandomSurvivalForest.predict`.

    :param estimators: list of fitted :class:`sksurv.tree.SurvivalTree` objects
        (``estimator.estimators_``)
    :param n_estimators: total number of trees (``estimator.n_estimators``)
    :param is_event_time: boolean mask of shape ``(n_unique_times,)`` indicating
        observed event times (``estimator.estimators_[0].is_event_time_``)
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
        value = tree.value  # shape: (n_nodes, n_unique_times, 2)

        for node_id in range(n_nodes):
            all_nodeids.append(node_id)
            all_treeids.append(tree_id)
            all_hitrates.append(1.0)
            all_mvt.append(0)

            left = children_left[node_id]
            right = children_right[node_id]

            if left == _LEAF:
                # Leaf node — weight is the CHF-sum prediction for this leaf.
                all_featureids.append(0)
                all_values.append(0.0)
                all_modes.append("LEAF")
                all_truenodeids.append(0)
                all_falsenodeids.append(0)

                all_target_nodeids.append(node_id)
                all_target_treeids.append(tree_id)
                all_target_ids.append(0)
                all_target_weights.append(_rsf_leaf_value(value, node_id, is_event_time))
            else:
                # Internal node.
                all_featureids.append(int(feature[node_id]))
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


def _extract_rsf_attributes_v5(
    estimators: list, n_estimators: int, is_event_time: np.ndarray, itype: int
) -> dict:
    """Extracts ``TreeEnsemble`` (opset 5) attributes for a :class:`RandomSurvivalForest`.

    Mirrors :func:`~yobx.sklearn.ensemble.random_forest._extract_forest_attributes_v5`
    for the regressor case, but derives each leaf weight from the cumulative
    hazard function stored in the survival tree.

    Leaf weights are pre-divided by ``n_estimators`` so that the ``SUM``
    aggregate function of ``TreeEnsemble`` yields the same average as
    :meth:`RandomSurvivalForest.predict`.

    :param estimators: list of fitted :class:`sksurv.tree.SurvivalTree` objects
    :param n_estimators: total number of trees
    :param is_event_time: boolean mask indicating observed event times
    :param itype: ONNX element type of the input tensor (e.g. ``TensorProto.FLOAT``)
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

    cumulative_internal_offset = 0
    cumulative_leaf_offset = 0

    for base_estimator in estimators:
        tree = base_estimator.tree_
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold.astype(dtype)
        value = tree.value  # shape: (n_nodes, n_unique_times, 2)
        n_nodes = tree.node_count

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

        # One virtual tree per base estimator (regression, n_targets=1).
        all_tree_roots.append(cumulative_internal_offset)

        # ------------------------------------------------------------------ #
        # nodes_* arrays                                                       #
        # ------------------------------------------------------------------ #
        if not internal_nodes:
            # Degenerate tree: one dummy internal node pointing to the single leaf.
            this_leaf_offset = cumulative_leaf_offset
            all_nodes_featureids.append(0)
            all_nodes_splits.append(0.0)
            all_nodes_modes.append(int(_NODE_MODE_LEQ))
            all_nodes_truenodeids.append(this_leaf_offset)
            all_nodes_trueleafs.append(1)
            all_nodes_falsenodeids.append(this_leaf_offset)
            all_nodes_falseleafs.append(1)
        else:
            node_offset = cumulative_internal_offset
            leaf_offset = cumulative_leaf_offset

            for nid in internal_nodes:
                left = int(children_left[nid])
                right = int(children_right[nid])

                all_nodes_featureids.append(int(feature[nid]))
                all_nodes_splits.append(float(threshold[nid]))
                all_nodes_modes.append(int(_NODE_MODE_LEQ))

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

        # ------------------------------------------------------------------ #
        # leaf_* arrays                                                        #
        # ------------------------------------------------------------------ #
        if not leaf_nodes:
            # Degenerate tree: single leaf from root node.
            leaf_w = _rsf_leaf_value(value, 0, is_event_time) / n_estimators
            all_leaf_targetids.append(0)
            all_leaf_weights.append(leaf_w)
        else:
            for nid in leaf_nodes:
                leaf_w = _rsf_leaf_value(value, nid, is_event_time) / n_estimators
                all_leaf_targetids.append(0)
                all_leaf_weights.append(leaf_w)

        # ------------------------------------------------------------------ #
        # Advance cumulative offsets.                                          #
        # ------------------------------------------------------------------ #
        cumulative_internal_offset += max(n_internal, 1)
        cumulative_leaf_offset += max(n_leaves, 1)

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


@register_sklearn_converter((RandomSurvivalForest,))
def sklearn_random_survival_forest(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: RandomSurvivalForest,
    X: str,
    name: str = "random_survival_forest",
) -> str:
    """
    Converts a :class:`sksurv.ensemble.RandomSurvivalForest` into ONNX.

    **Algorithm overview**

    :class:`~sksurv.ensemble.RandomSurvivalForest` is an ensemble of
    :class:`~sksurv.tree.SurvivalTree` estimators.  The risk-score
    prediction for a sample is the average of the individual tree
    predictions::

        predict(x) = mean_t [ sum_{j: is_event_time[j]} CHF_t(T_j | x) ]

    where ``CHF_t(T_j | x)`` is the cumulative hazard function value
    stored in the leaf reached by sample *x* in tree *t* at the *j*-th
    unique training time.

    Because each tree's contribution reduces to a scalar per leaf, the
    forest is equivalent to a standard ``TreeEnsembleRegressor`` once the
    leaf weights are pre-computed as the CHF sum over observed event times.

    Graph structure:

    .. code-block:: text

        X ──TreeEnsemble[Regressor]──► risk_scores (N, 1)

    When ``ai.onnx.ml`` opset 5 (or later) is available, the unified
    ``TreeEnsemble`` operator is used (leaf weights pre-divided by
    ``n_estimators``, ``aggregate_function=SUM``); otherwise the legacy
    ``TreeEnsembleRegressor`` operator is emitted with
    ``aggregate_function="AVERAGE"``.

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names
    :param estimator: a fitted :class:`~sksurv.ensemble.RandomSurvivalForest`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: output tensor name
    """
    assert isinstance(
        estimator, RandomSurvivalForest
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    ml_opset = g.get_opset("ai.onnx.ml")
    itype = g.get_type(X)
    n_estimators = estimator.n_estimators
    estimators = estimator.estimators_
    # All trees share the same unique_times_ / is_event_time_ from the forest.
    is_event_time: np.ndarray = estimators[0].is_event_time_

    if ml_opset >= 5:
        attrs = _extract_rsf_attributes_v5(estimators, n_estimators, is_event_time, itype)
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
        return outputs[0]
    # TreeEnsembleRegressor always outputs float32; cast back when needed.
    tree_outputs = [f"{outputs[0]}_tree_out"]

    attrs = _extract_rsf_attributes_legacy(estimators, n_estimators, is_event_time)
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

    g.make_node("Cast", [tree_result], outputs=outputs[:1], name=f"{name}_cast", to=itype)
    return outputs[0]
