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

**Numerical splits**: LightGBM's condition *"go to left child when
x ≤ split_condition"* maps to ``BRANCH_LEQ`` for both ``ai.onnx.ml``
opset ≤ 4 and opset ≥ 5 — exact match.

**Categorical splits**: LightGBM encodes categorical splits as
``decision_type == '=='`` with a threshold like ``'0||1||2'``.  ONNX only
supports single-value ``BRANCH_EQ`` comparisons, so each multi-value
categorical node is expanded into a chain of single-value checks by
:func:`_expand_categorical_splits` before flattening.  The memoised DFS in
:func:`_flatten_lgbm_tree` ensures shared subtree references (the ``left``
branch of every chain node) are assigned exactly one flat node ID.
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
# 0 = BRANCH_LEQ (x <= threshold) — exact match for LightGBM numerical split semantics.
_NODE_MODE_LEQ = np.uint8(0)
# 4 = BRANCH_EQ  (x == threshold) — used for LightGBM categorical splits.
_NODE_MODE_EQ = np.uint8(4)

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
# Tree flattening helpers (categorical-aware)
# ---------------------------------------------------------------------------


def _expand_categorical_splits(node: dict) -> dict:
    """Recursively expand LightGBM categorical split nodes into EQ chains.

    LightGBM represents a categorical split as ``decision_type == '=='`` with
    a threshold string like ``'0||1||2'``, meaning *go left if feature value is
    0, 1, or 2*.  ONNX only supports single-value ``BRANCH_EQ`` (``x == v``),
    so multi-value sets must be expanded into a chain of single-value checks:

    .. code-block:: text

        feature IN {0, 1}
          → EQ(feature==0): true→left, false→
              EQ(feature==1): true→left, false→right

    The ``left`` subtree may be shared (same object reference) across multiple
    EQ chain nodes — this is intentional and handled by the memoised DFS in
    :func:`_flatten_lgbm_tree`.

    :param node: node dict from ``booster_.dump_model()['tree_info'][i]['tree_structure']``
    :return: new node dict (leaf nodes are returned unchanged)
    """
    if "leaf_index" in node:
        return node

    left = _expand_categorical_splits(node["left_child"])
    right = _expand_categorical_splits(node["right_child"])

    if node.get("decision_type", "<=") == "==":
        # Categorical split: threshold is e.g. '0||1||2'
        categories = [int(c) for c in str(node["threshold"]).split("||")]
        feat = node["split_feature"]
        default_left = node.get("default_left", False)

        # Build chain from back: check each category value in turn.
        # True branch of every node goes to the original left subtree.
        # False branch of last node goes to the original right subtree.
        result: dict = right
        for cat in reversed(categories):
            result = {
                "split_feature": feat,
                "threshold": float(cat),
                "decision_type": "==",
                "default_left": default_left,
                "left_child": left,  # shared reference — memoised during flatten
                "right_child": result,
            }
        return result

    return {**node, "left_child": left, "right_child": right}


def _flatten_lgbm_tree(tree_structure: dict) -> Tuple[List[dict], List[dict], int]:
    """Flatten a LightGBM tree structure into sorted flat lists of nodes.

    Handles both numerical (``<=``) and categorical (``==``) splits.
    Shared subtree references introduced by :func:`_expand_categorical_splits`
    are handled via memoisation on object identity so that each unique node
    object is assigned exactly one flat ID.

    :param tree_structure: expanded root node (output of
        :func:`_expand_categorical_splits`)
    :return: ``(internal_nodes, leaf_nodes, n_internal)`` where

        * ``internal_nodes`` — list of dicts (one per unique internal node),
          each containing:
          ``id``, ``split_feature``, ``threshold``, ``mode`` (``"BRANCH_LEQ"``
          or ``"BRANCH_EQ"``), ``true_id``, ``true_is_leaf``,
          ``false_id``, ``false_is_leaf``, ``default_left``.
        * ``leaf_nodes`` — list of dicts (one per unique leaf node), each
          containing: ``id``, ``leaf_value``.
        * ``n_internal`` — number of unique internal nodes.

    Internal nodes are assigned IDs ``0..n_internal-1`` in DFS pre-order;
    leaf nodes are assigned IDs ``n_internal..n_internal+n_leaves-1``.
    """
    internal_nodes: List[dict] = []
    leaf_nodes: List[dict] = []
    # memo: id(node_object) → ('internal'|'leaf', assigned_id)
    memo: dict = {}

    internal_counter = [0]
    leaf_counter = [0]

    def _is_leaf(node: dict) -> bool:
        return "leaf_index" in node or "leaf_value" in node

    def _visit(node: dict) -> Tuple[str, int]:
        oid = id(node)
        if oid in memo:
            return memo[oid]

        if _is_leaf(node):
            my_id = leaf_counter[0]
            leaf_counter[0] += 1
            memo[oid] = ("leaf", my_id)
            leaf_nodes.append({"id": my_id, "leaf_value": float(node["leaf_value"])})
            return memo[oid]

        # Pre-assign internal ID before recursing so that back-references
        # (cycles, if any) resolve correctly.
        my_id = internal_counter[0]
        internal_counter[0] += 1
        memo[oid] = ("internal", my_id)

        left_type, left_id = _visit(node["left_child"])
        right_type, right_id = _visit(node["right_child"])

        dt = node.get("decision_type", "<=")
        mode = "BRANCH_EQ" if dt == "==" else "BRANCH_LEQ"

        internal_nodes.append(
            {
                "id": my_id,
                "split_feature": node["split_feature"],
                "threshold": float(node["threshold"]),
                "mode": mode,
                "true_id": left_id,
                "true_is_leaf": left_type == "leaf",
                "false_id": right_id,
                "false_is_leaf": right_type == "leaf",
                "default_left": bool(node.get("default_left", False)),
            }
        )
        return memo[oid]

    if _is_leaf(tree_structure):
        # Degenerate tree: single leaf root (no internal nodes).
        # Use _visit so memoization state remains consistent.
        _visit(tree_structure)
    else:
        _visit(tree_structure)

    n_internal = internal_counter[0]
    return internal_nodes, leaf_nodes, n_internal


def _build_lgbm_tree_attrs_legacy(
    trees: List[dict],
    n_targets: int,
    is_classifier: bool = False,
) -> dict:
    """Build legacy ``TreeEnsembleRegressor`` / ``TreeEnsembleClassifier`` attribute arrays.

    All trees are encoded into a single flat set of arrays suitable for the
    ``ai.onnx.ml ≤ 4`` ``TreeEnsembleRegressor`` or ``TreeEnsembleClassifier``
    operators.

    Numerical splits use ``BRANCH_LEQ`` (``x ≤ threshold``, left branch taken).
    Categorical splits (LightGBM ``decision_type == '=='``) are expanded into
    chains of ``BRANCH_EQ`` single-value checks by
    :func:`_expand_categorical_splits`, then flattened by
    :func:`_flatten_lgbm_tree`.

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
        ts_expanded = _expand_categorical_splits(tree_info["tree_structure"])
        internal_nodes, leaf_nodes, n_internal = _flatten_lgbm_tree(ts_expanded)

        target_id = tree_idx % n_targets

        # Internal nodes
        for node in sorted(internal_nodes, key=lambda n: n["id"]):
            node_id = node["id"]
            true_id = n_internal + node["true_id"] if node["true_is_leaf"] else node["true_id"]
            false_id = (
                n_internal + node["false_id"] if node["false_is_leaf"] else node["false_id"]
            )

            all_nodeids.append(node_id)
            all_treeids.append(tree_idx)
            all_hitrates.append(1.0)
            all_featureids.append(node["split_feature"])
            all_values.append(node["threshold"])
            all_modes.append(node["mode"])
            all_truenodeids.append(true_id)
            all_falsenodeids.append(false_id)
            all_mvt.append(1 if node["default_left"] else 0)

        # Leaf nodes
        for node in sorted(leaf_nodes, key=lambda n: n["id"]):
            node_id = n_internal + node["id"]
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
            all_target_weights.append(node["leaf_value"])

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
    flat arrays indexed by contiguous offsets.  Numerical splits use
    ``BRANCH_LEQ`` (mode 0) and categorical splits use ``BRANCH_EQ`` (mode 4),
    after categorical nodes have been expanded into chains by
    :func:`_expand_categorical_splits`.

    :param trees: list of ``tree_info`` dicts from ``booster_.dump_model()``
    :param n_targets: number of output targets
    :param itype: onnx float dtype for splits / weights
    :return: attribute dict for ``TreeEnsemble``
    """
    _MODE_INT = {"BRANCH_LEQ": int(_NODE_MODE_LEQ), "BRANCH_EQ": int(_NODE_MODE_EQ)}

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

    for tree_info in trees:
        ts_expanded = _expand_categorical_splits(tree_info["tree_structure"])
        internal_nodes, leaf_nodes, n_internal_actual = _flatten_lgbm_tree(ts_expanded)
        n_leaves_actual = len(leaf_nodes)

        all_tree_roots.append(cumulative_internal_offset)

        if not internal_nodes:
            # Degenerate tree (single leaf root): TreeEnsemble requires at least
            # one internal node.  Emit a dummy BRANCH_LEQ node that routes to the
            # single leaf regardless of the split result.
            all_nodes_featureids.append(0)
            all_nodes_splits.append(0.0)
            all_nodes_modes.append(int(_NODE_MODE_LEQ))
            all_nodes_truenodeids.append(cumulative_leaf_offset)
            all_nodes_trueleafs.append(1)
            all_nodes_falsenodeids.append(cumulative_leaf_offset)
            all_nodes_falseleafs.append(1)
        else:
            for node in sorted(internal_nodes, key=lambda n: n["id"]):
                true_id = (
                    cumulative_leaf_offset + node["true_id"]
                    if node["true_is_leaf"]
                    else cumulative_internal_offset + node["true_id"]
                )
                false_id = (
                    cumulative_leaf_offset + node["false_id"]
                    if node["false_is_leaf"]
                    else cumulative_internal_offset + node["false_id"]
                )

                all_nodes_featureids.append(node["split_feature"])
                all_nodes_splits.append(node["threshold"])
                all_nodes_modes.append(_MODE_INT[node["mode"]])
                all_nodes_truenodeids.append(true_id)
                all_nodes_trueleafs.append(1 if node["true_is_leaf"] else 0)
                all_nodes_falsenodeids.append(false_id)
                all_nodes_falseleafs.append(1 if node["false_is_leaf"] else 0)

        # Leaf arrays
        if not leaf_nodes:
            all_leaf_targetids.append(0)
            all_leaf_weights.append(0.0)
        else:
            for node in sorted(leaf_nodes, key=lambda n: n["id"]):
                all_leaf_targetids.append(0)  # overwritten per-tree below
                all_leaf_weights.append(node["leaf_value"])

        # Fix up leaf_targetids for the leaves we just appended
        n_leaves_appended = max(n_leaves_actual, 1)
        tree_idx = len(all_tree_roots) - 1
        real_target_id = tree_idx % n_targets
        for i in range(1, n_leaves_appended + 1):
            all_leaf_targetids[-i] = real_target_id

        cumulative_internal_offset += max(n_internal_actual, 1)
        cumulative_leaf_offset += n_leaves_appended

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
