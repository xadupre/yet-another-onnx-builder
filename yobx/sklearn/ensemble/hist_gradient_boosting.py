"""
Converter for :class:`sklearn.ensemble.HistGradientBoostingClassifier` and
:class:`sklearn.ensemble.HistGradientBoostingRegressor`.

The ONNX graph mirrors the model's prediction pipeline::

    raw_prediction = sum(tree_values for all trees) + baseline_prediction
    # regression:        raw_prediction  →  output (N, 1)
    # binary cls:        Sigmoid(raw)    →  [1-p, p],  ArgMax → label
    # multiclass:        Softmax(raw)    →  proba,     ArgMax → label

Two encoding paths are supported:

* **Legacy** (``ai.onnx.ml`` opset ≤ 4): ``TreeEnsembleRegressor``
  with ``aggregate_function="SUM"`` and ``base_values``.
* **Modern** (``ai.onnx.ml`` opset 5): ``TreeEnsemble``
  with ``aggregate_function=1`` (SUM) and ``base_values_as_tensor``.

Both paths raise :class:`NotImplementedError` when the model contains
categorical splits (``is_categorical == 1`` in any tree node), as the
ONNX ML operator set does not support bitset-based categorical splits.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import onnx
import onnx.helper as oh
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ..tree.decision_tree import _NODE_MODE_LEQ

_HGB_TYPES = (HistGradientBoostingClassifier, HistGradientBoostingRegressor)


def _check_no_categorical(trees) -> None:
    """Raise :class:`NotImplementedError` if any tree node uses a categorical split."""
    for tree in trees:
        if np.any(tree.nodes["is_categorical"]):
            raise NotImplementedError(
                "Categorical feature splits in HistGradientBoosting are not yet "
                "supported by this ONNX converter. Only numerical features are "
                "currently handled."
            )


def _extract_hgb_attributes_legacy(
    all_trees: List, n_targets: int, target_ids_per_tree: List[int]
) -> Dict:
    """
    Extract ``TreeEnsembleRegressor`` attributes from a flat list of
    :class:`sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor`
    objects.

    :param all_trees: flat list of ``TreePredictor`` objects ordered so that
        ``all_trees[i]`` is assigned ``tree_id = i`` in the ONNX node.
    :param n_targets: number of output targets (``n_trees_per_iteration_``).
    :param target_ids_per_tree: list of length ``len(all_trees)`` giving the
        target output index for every tree.
    :return: dict of ONNX attribute lists/values.
    """
    all_nodeids: List[int] = []
    all_treeids: List[int] = []
    all_modes: List[str] = []
    all_featureids: List[int] = []
    all_values: List[float] = []
    all_truenodeids: List[int] = []
    all_falsenodeids: List[int] = []
    all_hitrates: List[float] = []
    all_mvt: List[int] = []

    all_target_nodeids: List[int] = []
    all_target_treeids: List[int] = []
    all_target_ids: List[int] = []
    all_target_weights: List[float] = []

    for tree_id, (tree, target_id) in enumerate(zip(all_trees, target_ids_per_tree)):
        nodes = tree.nodes
        for node_idx in range(len(nodes)):
            node = nodes[node_idx]
            all_nodeids.append(node_idx)
            all_treeids.append(tree_id)
            all_hitrates.append(1.0)

            if node["is_leaf"]:
                all_modes.append("LEAF")
                all_featureids.append(0)
                all_values.append(0.0)
                all_truenodeids.append(0)
                all_falsenodeids.append(0)
                all_mvt.append(0)

                all_target_nodeids.append(node_idx)
                all_target_treeids.append(tree_id)
                all_target_ids.append(target_id)
                all_target_weights.append(float(node["value"]))
            else:
                all_modes.append("BRANCH_LEQ")
                all_featureids.append(int(node["feature_idx"]))
                all_values.append(float(node["num_threshold"]))
                # In HistGB: go left when feature <= threshold → truenodeid = left
                all_truenodeids.append(int(node["left"]))
                all_falsenodeids.append(int(node["right"]))
                all_mvt.append(int(node["missing_go_to_left"]))

    return dict(
        n_targets=n_targets,
        nodes_nodeids=all_nodeids,
        nodes_treeids=all_treeids,
        nodes_modes=all_modes,
        nodes_featureids=all_featureids,
        nodes_values=all_values,
        nodes_truenodeids=all_truenodeids,
        nodes_falsenodeids=all_falsenodeids,
        nodes_hitrates=all_hitrates,
        nodes_missing_value_tracks_true=all_mvt,
        target_nodeids=all_target_nodeids,
        target_treeids=all_target_treeids,
        target_ids=all_target_ids,
        target_weights=all_target_weights,
    )


def _extract_hgb_attributes_v5(
    all_trees: List, n_targets: int, target_ids_per_tree: List[int], itype: int
) -> Dict:
    """
    Extract ``TreeEnsemble`` (``ai.onnx.ml`` opset 5) attributes from a flat
    list of ``TreePredictor`` objects.

    The v5 operator stores internal nodes and leaf nodes separately.  For each
    tree, this function partitions nodes into internal and leaf arrays,
    re-maps absolute node indices to positions in those arrays, and builds the
    flat ``nodes_*`` / ``leaf_*`` attribute tensors.

    :param all_trees: flat list of ``TreePredictor`` objects.
    :param n_targets: number of output targets.
    :param target_ids_per_tree: target output index for every tree.
    :param dtype: numpy float dtype (``np.float32`` or ``np.float64``).
    :return: dict of ONNX attributes.
    """
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
    dtype = tensor_dtype_to_np_dtype(itype)

    for tree, target_id in zip(all_trees, target_ids_per_tree):
        nodes = tree.nodes
        n_nodes = len(nodes)

        # Partition into internal and leaf nodes.
        internal_indices: List[int] = []
        leaf_indices: List[int] = []
        for nid in range(n_nodes):
            if nodes[nid]["is_leaf"]:
                leaf_indices.append(nid)
            else:
                internal_indices.append(nid)

        # Maps from original node index to position within internal/leaf arrays.
        internal_idx_map = {nid: i for i, nid in enumerate(internal_indices)}
        leaf_idx_map = {nid: i for i, nid in enumerate(leaf_indices)}

        n_internal = len(internal_indices)
        n_leaves = len(leaf_indices)

        # Root is always node 0; map it to its position in the internal array
        # (it must be internal, because a fully degenerate tree is a single leaf).
        if n_internal == 0:
            # Degenerate tree: single leaf node, create a dummy internal node.
            all_tree_roots.append(cumulative_internal_offset)
            leaf_offset = cumulative_leaf_offset
            all_nodes_featureids.append(0)
            all_nodes_splits.append(0.0)
            all_nodes_modes.append(int(_NODE_MODE_LEQ))
            all_nodes_truenodeids.append(leaf_offset)
            all_nodes_trueleafs.append(1)
            all_nodes_falsenodeids.append(leaf_offset)
            all_nodes_falseleafs.append(1)
            all_leaf_targetids.append(target_id)
            all_leaf_weights.append(float(nodes[0]["value"]))
            cumulative_internal_offset += 1
            cumulative_leaf_offset += 1
        else:
            # Root is internal_indices[0] = 0 (HistGB root is always node 0).
            all_tree_roots.append(cumulative_internal_offset + internal_idx_map[0])

            node_offset = cumulative_internal_offset
            leaf_offset = cumulative_leaf_offset

            for nid in internal_indices:
                node = nodes[nid]
                left = int(node["left"])
                right = int(node["right"])

                all_nodes_featureids.append(int(node["feature_idx"]))
                all_nodes_splits.append(float(node["num_threshold"]))
                all_nodes_modes.append(int(_NODE_MODE_LEQ))

                left_is_leaf = bool(nodes[left]["is_leaf"])
                if left_is_leaf:
                    all_nodes_truenodeids.append(leaf_offset + leaf_idx_map[left])
                    all_nodes_trueleafs.append(1)
                else:
                    all_nodes_truenodeids.append(node_offset + internal_idx_map[left])
                    all_nodes_trueleafs.append(0)

                right_is_leaf = bool(nodes[right]["is_leaf"])
                if right_is_leaf:
                    all_nodes_falsenodeids.append(leaf_offset + leaf_idx_map[right])
                    all_nodes_falseleafs.append(1)
                else:
                    all_nodes_falsenodeids.append(node_offset + internal_idx_map[right])
                    all_nodes_falseleafs.append(0)

            for nid in leaf_indices:
                all_leaf_targetids.append(target_id)
                all_leaf_weights.append(float(nodes[nid]["value"]))

            cumulative_internal_offset += n_internal
            cumulative_leaf_offset += n_leaves

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
        n_targets=n_targets,
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


def _flatten_hgb_trees(
    estimator: Union[HistGradientBoostingClassifier, HistGradientBoostingRegressor],
) -> Tuple[List, List[int], int]:
    """
    Flatten the nested ``_predictors`` list into a single list of trees.

    Returns:
        ``(all_trees, target_ids_per_tree, n_targets)`` where:

        * ``all_trees`` — flat list of ``TreePredictor`` objects.
        * ``target_ids_per_tree`` — output target index for each tree (0 for
          regression / binary, class index for multiclass).
        * ``n_targets`` — total number of output targets.
    """
    n_trees_per_iteration = estimator.n_trees_per_iteration_
    all_trees: List = []
    target_ids: List[int] = []
    for iteration_trees in estimator._predictors:
        for k, tree in enumerate(iteration_trees):
            all_trees.append(tree)
            target_ids.append(k)
    return all_trees, target_ids, n_trees_per_iteration


def _build_hgb_raw_output_legacy(
    g: GraphBuilderExtendedProtocol,
    X: str,
    name: str,
    all_trees: List,
    target_ids_per_tree: List[int],
    n_targets: int,
    baseline: np.ndarray,
    raw_outputs: List[str],
) -> str:
    """
    Emit a ``TreeEnsembleRegressor`` node (legacy path) and return its output name.

    The ``base_values`` attribute encodes the ``_baseline_prediction`` from the
    model so the ONNX node computes ``raw_prediction`` in a single pass.
    """
    attrs = _extract_hgb_attributes_legacy(all_trees, n_targets, target_ids_per_tree)
    # base_values: add baseline to the sum of all tree contributions.
    attrs["base_values"] = baseline.flatten().astype(np.float32).tolist()
    attrs["aggregate_function"] = "SUM"
    attrs["post_transform"] = "NONE"

    result = g.make_node(
        "TreeEnsembleRegressor",
        [X],
        outputs=raw_outputs,
        domain="ai.onnx.ml",
        name=f"{name}_ter",
        **attrs,  # type: ignore
    )
    return result if isinstance(result, str) else result[0]


def _build_hgb_raw_output_v5(
    g: GraphBuilderExtendedProtocol,
    X: str,
    name: str,
    all_trees: List,
    target_ids_per_tree: List[int],
    n_targets: int,
    baseline: np.ndarray,
    raw_outputs: List[str],
    itype: int,
) -> str:
    """
    Emit a ``TreeEnsemble`` node (opset-5 path) followed by an ``Add`` for the
    baseline, and return the output name.

    The ``TreeEnsemble`` operator (``ai.onnx.ml`` opset 5) does not have a
    ``base_values`` attribute, so the baseline prediction is applied via a
    constant ``Add`` node.
    """
    attrs = _extract_hgb_attributes_v5(all_trees, n_targets, target_ids_per_tree, itype)

    te_out = g.unique_name(f"{name}_te_out")
    result = g.make_node(
        "TreeEnsemble",
        [X],
        outputs=[te_out],
        domain="ai.onnx.ml",
        name=f"{name}_te",
        post_transform=0,  # NONE
        aggregate_function=1,  # SUM
        **attrs,  # type: ignore
    )
    te_out_name = result if isinstance(result, str) else result[0]

    # Add the baseline prediction as a constant.
    # Use the same dtype as the tree node output (which matches leaf_weights).
    bv = baseline.flatten().astype(tensor_dtype_to_np_dtype(itype))
    bv_expanded = bv.reshape(1, n_targets)  # broadcast over batch dimension
    add_result = g.op.Add(
        te_out_name, bv_expanded, name=f"{name}_add_baseline", outputs=raw_outputs
    )
    return add_result if isinstance(add_result, str) else add_result[0]


@register_sklearn_converter((HistGradientBoostingRegressor,))
def sklearn_hgb_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: HistGradientBoostingRegressor,
    X: str,
    name: str = "hgb_regressor",
) -> str:
    """
    Converts a :class:`sklearn.ensemble.HistGradientBoostingRegressor` to ONNX.

    When ``ai.onnx.ml`` opset 5 (or later) is active the unified
    ``TreeEnsemble`` operator is used; otherwise the legacy
    ``TreeEnsembleRegressor`` is emitted.

    The prediction formula is::

        raw = sum(tree.predict(X) for tree in _predictors) + _baseline_prediction
        output = raw          # shape (N, 1)

    When the input is ``float64`` the output is cast back to ``float64``
    (both ONNX ML tree operators always output ``float32``).

    :param g: graph builder
    :param sts: shapes provided by scikit-learn
    :param outputs: desired output names
    :param estimator: fitted ``HistGradientBoostingRegressor``
    :param X: input tensor name
    :param name: node-name prefix
    :return: output tensor name  (shape ``[N, 1]``)
    :raises NotImplementedError: if the model contains categorical splits
    """
    assert isinstance(estimator, HistGradientBoostingRegressor)

    all_trees, target_ids, n_targets = _flatten_hgb_trees(estimator)
    _check_no_categorical(all_trees)

    baseline = estimator._baseline_prediction  # shape (1, 1)

    itype = g.get_type(X) if g.has_type(X) else onnx.TensorProto.FLOAT
    tree_outputs = [f"{outputs[0]}_tree_out"]

    ml_opset = g.get_opset("ai.onnx.ml")
    if ml_opset >= 5:
        raw = _build_hgb_raw_output_v5(
            g,
            X,
            name,
            all_trees,
            target_ids,
            n_targets,
            baseline,
            tree_outputs,
            itype=g.get_type(X),
        )
    else:
        raw = _build_hgb_raw_output_legacy(
            g, X, name, all_trees, target_ids, n_targets, baseline, tree_outputs
        )

    cast_result = g.make_node("Cast", [raw], outputs=outputs, name=f"{name}_cast_f64", to=itype)
    return cast_result if isinstance(cast_result, str) else cast_result[0]


@register_sklearn_converter((HistGradientBoostingClassifier,))
def sklearn_hgb_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: HistGradientBoostingClassifier,
    X: str,
    name: str = "hgb_classifier",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.ensemble.HistGradientBoostingClassifier` to ONNX.

    When ``ai.onnx.ml`` opset 5 (or later) is active the unified
    ``TreeEnsemble`` operator is used; otherwise the legacy
    ``TreeEnsembleRegressor`` is emitted.

    **Binary classification** — the raw sum (one logit per sample) passes
    through a ``Sigmoid``; the resulting probability ``p`` for class 1 is
    concatenated as ``[1-p, p]`` to match ``predict_proba``.

    **Multiclass** — the raw sums (one logit per class) pass through a
    ``Softmax`` along axis 1.

    In both cases the predicted label is derived via ``ArgMax`` and a
    ``Gather`` into the ``classes_`` array.

    :param g: graph builder
    :param sts: shapes provided by scikit-learn
    :param outputs: desired output names  (label, probabilities)
    :param estimator: fitted ``HistGradientBoostingClassifier``
    :param X: input tensor name
    :param name: node-name prefix
    :return: tuple ``(label_name, proba_name)``
    :raises NotImplementedError: if the model contains categorical splits
    """
    assert isinstance(estimator, HistGradientBoostingClassifier)

    classes = estimator.classes_
    is_binary = estimator.n_trees_per_iteration_ == 1

    all_trees, target_ids, n_targets = _flatten_hgb_trees(estimator)
    _check_no_categorical(all_trees)

    baseline = estimator._baseline_prediction  # shape (1, n_trees_per_iteration_)

    ml_opset = g.get_opset("ai.onnx.ml")
    itype = g.get_type(X)
    raw_name = g.unique_name(f"{name}_raw")

    if ml_opset >= 5:
        raw = _build_hgb_raw_output_v5(
            g, X, name, all_trees, target_ids, n_targets, baseline, [raw_name], itype=itype
        )
    else:
        raw = _build_hgb_raw_output_legacy(
            g, X, name, all_trees, target_ids, n_targets, baseline, [raw_name]
        )

    itype = g.get_type(X)
    raw_f32 = g.op.Cast(raw, to=itype, name=name)

    if is_binary:
        # raw_f32: (N, 1) → sigmoid → p1, complement → p0, concat → proba (N, 2)
        p1 = g.op.Sigmoid(raw_f32, name=f"{name}_sigmoid")
        one_cst = np.ones((1, 1), dtype=tensor_dtype_to_np_dtype(itype))
        p0 = g.op.Sub(one_cst, p1, name=f"{name}_p0")
        proba_raw = g.op.Concat(p0, p1, axis=1, name=f"{name}_concat")
    else:
        # raw_f32: (N, n_classes) → softmax → proba (N, n_classes)
        proba_raw = g.op.Softmax(raw_f32, axis=1, name=f"{name}_softmax")

    # Rename to desired output name.
    proba = g.op.Identity(proba_raw, name=f"{name}_proba", outputs=outputs[1:])

    # ------------------------------------------------------------------ #
    # Predicted label: ArgMax over probabilities.                         #
    # ------------------------------------------------------------------ #
    label_idx = g.op.ArgMax(proba_raw, axis=1, keepdims=0, name=f"{name}_argmax")
    label_idx_i64 = g.op.Cast(label_idx, to=onnx.TensorProto.INT64, name=f"{name}_cast_idx")

    if np.issubdtype(classes.dtype, np.integer):  # type: ignore
        classes_arr = classes.astype(np.int64)
        label = g.op.Gather(
            classes_arr, label_idx_i64, axis=0, name=f"{name}_label", outputs=outputs[:1]
        )
        g.set_type(label, onnx.TensorProto.INT64)
    else:
        classes_arr = np.array(classes.astype(str))
        label = g.op.Gather(
            classes_arr, label_idx_i64, axis=0, name=f"{name}_label_str", outputs=outputs[:1]
        )
        g.set_type(label, onnx.TensorProto.STRING)

    return label, proba
