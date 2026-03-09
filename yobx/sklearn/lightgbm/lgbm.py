"""
ONNX converters for :class:`lightgbm.LGBMClassifier` and
:class:`lightgbm.LGBMRegressor`.

The trees are extracted from the fitted booster via
``booster_.dump_model()`` and encoded using the ONNX ML
``TreeEnsembleClassifier`` / ``TreeEnsembleRegressor`` operators (legacy
``ai.onnx.ml`` opset ≤ 4) or the unified ``TreeEnsemble`` operator
(``ai.onnx.ml`` opset ≥ 5).

* **Binary classification** — the raw per-sample margin is passed through
  a sigmoid function and assembled into a ``[N, 2]`` probability matrix.
* **Multi-class classification** — per-class margins are passed through
  softmax to produce a ``[N, n_classes]`` probability matrix.
* **Regression** — raw margin output with an objective-dependent output
  transform:

  * Identity objectives (``regression``, ``regression_l1``, ``huber``,
    ``quantile``, ``mape``, …): no transform; raw == prediction.
  * Exp objectives (``poisson``, ``tweedie``): ``exp(margin)``; prediction
    is in positive-real space.

LightGBM's tree-branching condition *"go to left child when
x ≤ split_condition"* maps to:

* ``BRANCH_LEQ`` for both ``ai.onnx.ml`` opset ≤ 4 and opset ≥ 5 — exact
  match.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import onnx
import onnx.helper as oh
from lightgbm import LGBMRegressor, LGBMClassifier
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from ..register import register_sklearn_converter

# Mode encoding for ai.onnx.ml opset-5 TreeEnsemble.
# 0 = BRANCH_LEQ (x <= threshold) — exact match for LightGBM semantics.
_NODE_MODE_LEQ = np.uint8(0)

# ---------------------------------------------------------------------------
# Supported regression objectives, grouped by their output transform.
# ---------------------------------------------------------------------------

#: Regression objectives that use the identity link (no output transform).
_REG_IDENTITY_OBJECTIVES = frozenset(
    {
        "regression",
        "regression_l1",
        "huber",
        "quantile",
        "mape",
        "cross_entropy",
        "cross_entropy_lambda",
    }
)

#: Regression objectives that apply ``exp`` as the output transform
#: (log-link objectives).
_REG_EXP_OBJECTIVES = frozenset(
    {
        "poisson",
        "tweedie",
    }
)


def _get_reg_output_transform(objective: str) -> Optional[str]:
    """Return the ONNX output-transform type for an :class:`~lightgbm.LGBMRegressor` objective.

    :param objective: LightGBM objective string (e.g. ``"regression"``)
    :return: ``"exp"`` when a transform is needed, ``None`` for identity
        objectives.
    :raises NotImplementedError: when *objective* is not supported by this
        converter.  Callers should catch this and report a meaningful error
        rather than silently producing wrong outputs.
    """
    # Strip any extra parameters (e.g. "tweedie tweedie_variance_power:1.5")
    base = objective.split()[0].split(":")[0]
    if base in _REG_IDENTITY_OBJECTIVES:
        return None
    if base in _REG_EXP_OBJECTIVES:
        return "exp"
    raise NotImplementedError(
        f"LGBMRegressor ONNX converter: objective {objective!r} is not yet supported. "
        f"Supported objectives — identity: {sorted(_REG_IDENTITY_OBJECTIVES)}, "
        f"exp: {sorted(_REG_EXP_OBJECTIVES)}."
    )


# ---------------------------------------------------------------------------
# Tree flattening helpers
# ---------------------------------------------------------------------------


def _collect_lgbm_nodes(
    node: dict,
    n_internal: int,
    internal_nodes: List[dict],
    leaf_nodes: List[dict],
) -> None:
    """Recursively collect all nodes from a LightGBM dump_model tree.

    Internal nodes (split nodes) are appended to *internal_nodes* using
    their ``split_index`` as the identifier.  Leaf nodes are appended to
    *leaf_nodes* using their ``leaf_index``.

    :param node: root node dict from ``booster_.dump_model()['tree_info'][i]['tree_structure']``
    :param n_internal: total number of internal nodes in the tree (used to
        offset leaf IDs)
    :param internal_nodes: output list that collects internal node dicts
    :param leaf_nodes: output list that collects leaf node dicts
    """
    if "leaf_index" in node:
        leaf_nodes.append(node)
    else:
        internal_nodes.append(node)
        _collect_lgbm_nodes(node["left_child"], n_internal, internal_nodes, leaf_nodes)
        _collect_lgbm_nodes(node["right_child"], n_internal, internal_nodes, leaf_nodes)


def _child_node_id(child: dict, n_internal: int) -> int:
    """Return the flat node ID for a child node.

    Internal nodes are numbered ``0..n_internal-1`` (by ``split_index``).
    Leaf nodes are numbered ``n_internal..n_internal+n_leaves-1``
    (by ``n_internal + leaf_index``).

    :param child: child node dict from dump_model
    :param n_internal: total number of internal nodes in the tree
    :return: flat node ID
    """
    if "leaf_index" in child:
        return n_internal + child["leaf_index"]
    return child["split_index"]


def _build_lgbm_tree_attrs_legacy(
    trees: List[dict],
    n_targets: int,
    is_classifier: bool = False,
) -> dict:
    """Build legacy ``TreeEnsembleRegressor`` / ``TreeEnsembleClassifier`` attribute arrays.

    All trees are encoded into a single flat set of arrays suitable for the
    ``ai.onnx.ml ≤ 4`` ``TreeEnsembleRegressor`` or ``TreeEnsembleClassifier``
    operators.

    The branching direction maps LightGBM's *left child* (``x ≤ threshold``)
    to the ONNX ``BRANCH_LEQ`` *true* branch, matching LightGBM's exact
    semantics.

    :param trees: list of ``tree_info`` dicts from ``booster_.dump_model()``
    :param n_targets: number of output targets (1 for binary/regression,
        ``n_classes`` for multi-class)
    :param is_classifier: when ``True`` the returned dict uses ``class_*``
        attribute keys (for ``TreeEnsembleClassifier``) instead of the
        ``target_*`` keys used by ``TreeEnsembleRegressor``
    :return: flat attribute dict for the tree ensemble operator
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

    for tree_idx, tree_info in enumerate(trees):
        ts = tree_info["tree_structure"]
        num_leaves = tree_info["num_leaves"]
        n_internal = num_leaves - 1  # binary tree: n_internal = n_leaves - 1

        # Collect all internal/leaf nodes from the recursive structure.
        internal_nodes: List[dict] = []
        leaf_nodes: List[dict] = []
        _collect_lgbm_nodes(ts, n_internal, internal_nodes, leaf_nodes)

        target_id = tree_idx % n_targets

        # Internal nodes: node_id = split_index
        for node in sorted(internal_nodes, key=lambda n: n["split_index"]):
            node_id = node["split_index"]
            all_nodeids.append(node_id)
            all_treeids.append(tree_idx)
            all_hitrates.append(1.0)

            all_featureids.append(node["split_feature"])
            all_values.append(float(node["threshold"]))
            # LightGBM: left child taken when x <= threshold (BRANCH_LEQ true branch).
            all_modes.append("BRANCH_LEQ")
            # True branch = left child, False branch = right child.
            all_truenodeids.append(_child_node_id(node["left_child"], n_internal))
            all_falsenodeids.append(_child_node_id(node["right_child"], n_internal))
            # missing_value_tracks_true = 1 if NaN goes to left (true) branch.
            all_mvt.append(1 if node["default_left"] else 0)

        # Leaf nodes: node_id = n_internal + leaf_index
        for node in sorted(leaf_nodes, key=lambda n: n["leaf_index"]):
            node_id = n_internal + node["leaf_index"]
            all_nodeids.append(node_id)
            all_treeids.append(tree_idx)
            all_hitrates.append(1.0)

            all_featureids.append(0)
            all_values.append(0.0)
            all_modes.append("LEAF")
            all_truenodeids.append(0)
            all_falsenodeids.append(0)
            all_mvt.append(0)

            all_target_nodeids.append(node_id)
            all_target_treeids.append(tree_idx)
            all_target_ids.append(target_id)
            all_target_weights.append(float(node["leaf_value"]))

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


def _build_lgbm_tree_attrs_v5(
    trees: List[dict],
    n_targets: int,
    itype: int,
) -> dict:
    """Build ``TreeEnsemble`` (``ai.onnx.ml`` opset 5) attribute arrays.

    The opset-5 encoding stores *internal* nodes and *leaf* nodes in separate
    flat arrays indexed by contiguous offsets.  ``BRANCH_LEQ`` (mode 0) is
    used so that LightGBM's ``x ≤ threshold`` condition is matched exactly.

    :param trees: list of ``tree_info`` dicts from ``booster_.dump_model()``
    :param n_targets: number of output targets
    :param itype: onnx float dtype for splits / weights
    :return: attribute dict for ``TreeEnsemble``
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

    for tree_idx, tree_info in enumerate(trees):
        ts = tree_info["tree_structure"]
        num_leaves = tree_info["num_leaves"]
        n_internal = num_leaves - 1  # binary tree: n_internal = n_leaves - 1

        # Collect all internal/leaf nodes from the recursive structure.
        internal_nodes_list: List[dict] = []
        leaf_nodes_list: List[dict] = []
        _collect_lgbm_nodes(ts, n_internal, internal_nodes_list, leaf_nodes_list)

        # Sort by split_index / leaf_index for deterministic ordering.
        internal_nodes_sorted = sorted(internal_nodes_list, key=lambda n: n["split_index"])
        leaf_nodes_sorted = sorted(leaf_nodes_list, key=lambda n: n["leaf_index"])

        # Build index maps: split_index → position in internal_nodes_sorted,
        # leaf_index → position in leaf_nodes_sorted.
        internal_idx = {n["split_index"]: i for i, n in enumerate(internal_nodes_sorted)}
        leaf_idx = {n["leaf_index"]: i for i, n in enumerate(leaf_nodes_sorted)}

        n_internal_actual = len(internal_nodes_sorted)
        n_leaves_actual = len(leaf_nodes_sorted)

        target_id = tree_idx % n_targets

        all_tree_roots.append(cumulative_internal_offset)

        if not internal_nodes_sorted:
            # Degenerate tree (single leaf root): the ONNX TreeEnsemble spec
            # requires at least one internal node.  We emit a dummy split node
            # whose both branches (true and false) point to the single leaf at
            # cumulative_leaf_offset.  The split condition (feature 0, threshold
            # 0.0, BRANCH_LEQ) will always route to one branch, but since both
            # branches yield the same constant leaf value the result is correct.
            all_nodes_featureids.append(0)
            all_nodes_splits.append(0.0)
            all_nodes_modes.append(int(_NODE_MODE_LEQ))
            all_nodes_truenodeids.append(cumulative_leaf_offset)
            all_nodes_trueleafs.append(1)
            all_nodes_falsenodeids.append(cumulative_leaf_offset)
            all_nodes_falseleafs.append(1)
        else:
            for node in internal_nodes_sorted:
                feat_idx = node["split_feature"]
                left_child = node["left_child"]
                right_child = node["right_child"]

                all_nodes_featureids.append(feat_idx)
                all_nodes_splits.append(float(node["threshold"]))
                all_nodes_modes.append(int(_NODE_MODE_LEQ))

                # True branch = left child (x <= threshold).
                left_is_leaf = "leaf_index" in left_child
                if left_is_leaf:
                    all_nodes_truenodeids.append(
                        cumulative_leaf_offset + leaf_idx[left_child["leaf_index"]]
                    )
                    all_nodes_trueleafs.append(1)
                else:
                    all_nodes_truenodeids.append(
                        cumulative_internal_offset + internal_idx[left_child["split_index"]]
                    )
                    all_nodes_trueleafs.append(0)

                # False branch = right child (x > threshold).
                right_is_leaf = "leaf_index" in right_child
                if right_is_leaf:
                    all_nodes_falsenodeids.append(
                        cumulative_leaf_offset + leaf_idx[right_child["leaf_index"]]
                    )
                    all_nodes_falseleafs.append(1)
                else:
                    all_nodes_falsenodeids.append(
                        cumulative_internal_offset + internal_idx[right_child["split_index"]]
                    )
                    all_nodes_falseleafs.append(0)

        # Build leaf arrays.
        if not leaf_nodes_sorted:
            # Degenerate tree: single dummy leaf.
            all_leaf_targetids.append(target_id)
            all_leaf_weights.append(0.0)
        else:
            for node in leaf_nodes_sorted:
                all_leaf_targetids.append(target_id)
                all_leaf_weights.append(float(node["leaf_value"]))

        cumulative_internal_offset += max(n_internal_actual, 1)
        cumulative_leaf_offset += max(n_leaves_actual, 1)

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


def _emit_lgbm_tree_node(
    g: GraphBuilderExtendedProtocol,
    X: str,
    name: str,
    n_targets: int,
    trees: List[dict],
    ml_opset: int,
    intermediate_name: Optional[str] = None,
    is_classifier: bool = False,
    itype: int = 0,
) -> str:
    """Emit a ``TreeEnsembleRegressor`` / ``TreeEnsembleClassifier`` / ``TreeEnsemble`` ONNX node.

    :param g: graph builder
    :param X: input tensor name
    :param name: node name prefix
    :param n_targets: number of output targets
    :param trees: list of ``tree_info`` dicts from ``booster_.dump_model()``
    :param ml_opset: ``ai.onnx.ml`` opset version
    :param intermediate_name: if provided, use this as the output name;
        otherwise let the builder choose
    :param is_classifier: when ``True`` and ``ml_opset < 5``, emits a
        ``TreeEnsembleClassifier`` node and returns the scores (second output)
    :param itype: onnx float dtype for numeric attributes
    :return: output tensor name (shape ``[N, n_targets]``)
    """
    out_arg = [intermediate_name] if intermediate_name else 1

    if ml_opset >= 5:
        attrs = _build_lgbm_tree_attrs_v5(trees, n_targets=n_targets, itype=itype)
        result = g.make_node(
            "TreeEnsemble",
            [X],
            outputs=out_arg,
            domain="ai.onnx.ml",
            name=f"{name}_te",
            post_transform=0,  # NONE
            aggregate_function=1,  # SUM
            **attrs,  # type: ignore
        )
    elif is_classifier:
        attrs = _build_lgbm_tree_attrs_legacy(trees, n_targets=n_targets, is_classifier=True)
        n_labels = max(n_targets, 2)
        classlabels = list(range(n_labels))
        result = g.make_node(
            "TreeEnsembleClassifier",
            [X],
            outputs=2,
            domain="ai.onnx.ml",
            name=f"{name}_tec",
            post_transform="NONE",
            classlabels_int64s=classlabels,
            **attrs,  # type: ignore
        )
        assert len(result) == 2, f"Unexpected output: {result!r}{g.get_debug_msg()}"
        assert g.has_type(result[0]), f"Type is missing for {result[0]}{g.get_debug_msg()}"
        assert g.has_type(result[1]), f"Type is missing for {result[1]}{g.get_debug_msg()}"
        scores = result[1]
        if n_targets == 1:
            # Binary: ONNX binary complement gives [N, 2] where col 1 = raw_tree_sum.
            col1 = np.array([1], dtype=np.int64)
            return g.op.Gather(scores, col1, axis=1, name=f"{name}_tec_col1")
        return scores
    else:
        attrs = _build_lgbm_tree_attrs_legacy(trees, n_targets=n_targets)
        result = g.make_node(
            "TreeEnsembleRegressor",
            [X],
            outputs=out_arg,
            domain="ai.onnx.ml",
            name=f"{name}_ter",
            n_targets=n_targets,
            aggregate_function="SUM",
            post_transform="NONE",
            **attrs,  # type: ignore
        )

    return result if isinstance(result, str) else result[0]


def _gather_labels(
    g: GraphBuilderExtendedProtocol,
    label_idx: str,
    classes,
    name: str,
    out_name: str,
) -> str:
    """Gather class labels from ``classes`` at positions ``label_idx``.

    :param g: graph builder
    :param label_idx: ``[N]`` int64 tensor of class indices
    :param classes: sklearn ``classes_`` array
    :param name: node name prefix
    :param out_name: desired output tensor name
    :return: output tensor name
    """
    if np.issubdtype(classes.dtype, np.integer):
        classes_arr = classes.astype(np.int64)
        label = g.op.Gather(
            classes_arr,
            label_idx,
            axis=0,
            name=f"{name}_label",
            outputs=[out_name],
        )
        if not g.has_type(label):
            g.set_type(label, onnx.TensorProto.INT64)
    else:
        classes_arr = np.array(classes, dtype=str)
        label = g.op.Gather(
            classes_arr,
            label_idx,
            axis=0,
            name=f"{name}_label_str",
            outputs=[out_name],
        )
        if not g.has_type(label):
            g.set_type(label, onnx.TensorProto.STRING)
    return label


# ---------------------------------------------------------------------------
# LGBMClassifier converter
# ---------------------------------------------------------------------------


@register_sklearn_converter(LGBMClassifier)
def sklearn_lgbm_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator,
    X: str,
    name: str = "lgbm_classifier",
) -> Tuple[str, str]:
    """Convert an :class:`lightgbm.LGBMClassifier` to ONNX.

    The converter supports:

    * **Binary classification** (``n_classes_ == 2``) — one tree per
      boosting round; sigmoid post-processing; output shape ``[N, 2]``.
    * **Multi-class classification** (``n_classes_ > 2``) — ``n_classes``
      trees per round; softmax post-processing; output shape
      ``[N, n_classes]``.

    Both ``ai.onnx.ml`` legacy (opset ≤ 4) and modern (opset ≥ 5) encodings
    are emitted based on the active opset in *g*.

    :param g: the graph builder to add nodes to
    :param sts: shapes dict (passed through, not used internally)
    :param outputs: desired output names ``[label, probabilities]``
    :param estimator: a fitted ``LGBMClassifier``
    :param X: input tensor name
    :param name: prefix for node names added to the graph
    :return: tuple ``(label_result_name, proba_result_name)``
    """
    booster = estimator.booster_
    model_dict = booster.dump_model()
    n_classes: int = int(estimator.n_classes_)
    is_binary: bool = n_classes == 2

    ml_opset = g.get_opset("ai.onnx.ml")
    itype = g.get_type(X)

    trees = model_dict["tree_info"]
    n_targets = 1 if is_binary else n_classes

    raw_scores = _emit_lgbm_tree_node(
        g,
        X,
        name,
        n_targets=n_targets,
        trees=trees,
        ml_opset=ml_opset,
        is_classifier=True,
        itype=itype,
    )

    classes = estimator.classes_
    post_dtype = tensor_dtype_to_np_dtype(g.get_type(raw_scores))

    if is_binary:
        # sigmoid → [N, 1] probability of positive class
        p1 = g.op.Sigmoid(raw_scores, name=f"{name}_sigmoid")

        # p0 = 1 - p1 → [N, 1]
        ones = np.array([1.0], dtype=post_dtype)
        p0 = g.op.Sub(ones, p1, name=f"{name}_p0")

        # Concat [p0, p1] → [N, 2]
        proba = g.op.Concat(p0, p1, axis=1, name=f"{name}_concat", outputs=outputs[1:])
        assert isinstance(proba, str)

        # Label via ArgMax → [N]
        label_idx = g.op.ArgMax(proba, axis=1, keepdims=0, name=f"{name}_argmax")
        label_idx_i64 = g.op.Cast(label_idx, to=onnx.TensorProto.INT64, name=f"{name}_cast")
    else:
        # Multi-class: softmax → [N, n_classes]
        proba = g.op.Softmax(raw_scores, axis=1, name=f"{name}_softmax", outputs=outputs[1:])
        assert isinstance(proba, str)

        # Label via ArgMax → [N]
        label_idx = g.op.ArgMax(proba, axis=1, keepdims=0, name=f"{name}_argmax")
        label_idx_i64 = g.op.Cast(label_idx, to=onnx.TensorProto.INT64, name=f"{name}_cast")

    label = _gather_labels(g, label_idx_i64, classes, name, outputs[0])
    return label, proba


# ---------------------------------------------------------------------------
# LGBMRegressor converter
# ---------------------------------------------------------------------------


@register_sklearn_converter(LGBMRegressor)
def sklearn_lgbm_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator,
    X: str,
    name: str = "lgbm_regressor",
) -> str:
    """Convert an :class:`lightgbm.LGBMRegressor` to ONNX.

    The raw margin (sum of all tree leaf values) is computed via a
    ``TreeEnsembleRegressor`` / ``TreeEnsemble`` node, and then an
    objective-dependent output transform is applied to match
    :meth:`~lightgbm.LGBMRegressor.predict`:

    * Identity (``regression``, ``regression_l1``, ``huber``, ``quantile``,
      ``mape``, …): no transform.
    * ``poisson``, ``tweedie``: ``exp(margin)``.

    Unsupported objectives raise :class:`NotImplementedError`.

    :param g: the graph builder to add nodes to
    :param sts: shapes dict (passed through, not used internally)
    :param outputs: desired output names ``[predictions]``
    :param estimator: a fitted ``LGBMRegressor``
    :param X: input tensor name
    :param name: prefix for node names added to the graph
    :return: output tensor name (shape ``[N, 1]``)
    :raises NotImplementedError: if the model's objective is not supported
    """
    booster = estimator.booster_
    model_dict = booster.dump_model()
    objective: str = model_dict["objective"]

    # Validate early so the error message is clear before we build any graph.
    out_transform = _get_reg_output_transform(objective)

    ml_opset = g.get_opset("ai.onnx.ml")
    itype = g.get_type(X)

    trees = model_dict["tree_info"]
    tree_out_name = f"{outputs[0]}_tree_out"

    raw_scores = _emit_lgbm_tree_node(
        g,
        X,
        name,
        n_targets=1,
        trees=trees,
        ml_opset=ml_opset,
        intermediate_name=tree_out_name,
        itype=itype,
    )

    # For opset ≥ 5 with float64 leaf weights, TreeEnsemble natively outputs
    # float64, so no extra Cast is required.  For legacy opset < 5, the ONNX
    # spec declares float32 output, but runtimes may propagate float64 when the
    # input is float64 (reference evaluator) or always output float32 (ORT).
    # We normalise by inserting an explicit Cast(→itype) so that both runtimes
    # behave consistently before any output transform is applied.
    raw_scores = g.make_node("Cast", [raw_scores], outputs=1, name=f"{name}_tree_cast", to=itype)

    # Apply the objective-specific output transform (Exp / identity).
    if out_transform == "exp":
        raw_scores = g.op.Exp(raw_scores, name=f"{name}_exp")
    # out_transform is None → identity, no node needed.

    cast_result = g.make_node(
        "Cast", [raw_scores], outputs=outputs, name=f"{name}_cast", to=itype
    )
    return cast_result
