"""
ONNX converters for :class:`xgboost.XGBClassifier` and
:class:`xgboost.XGBRegressor`.

The trees are extracted from the fitted booster via
``booster.get_dump(dump_format='json')`` and encoded using the ONNX ML
``TreeEnsembleRegressor`` operator (legacy ``ai.onnx.ml`` opset ≤ 4) or the
unified ``TreeEnsemble`` operator (``ai.onnx.ml`` opset ≥ 5).

* **Binary classification** — the raw per-sample margin is passed through
  a sigmoid function and assembled into a ``[N, 2]`` probability matrix.
* **Multi-class classification** — per-class margins are passed through
  softmax to produce a ``[N, n_classes]`` probability matrix.
* **Regression** — raw margin output with the XGBoost ``base_score`` bias
  added as a constant.

The conversion supports XGBoost 2.x and treats the stored ``base_score``
configuration value as the untransformed prediction-space value:

* Binary/logistic objectives: ``margin_bias = logit(base_score)``
  (equals 0 for the default ``base_score = 0.5``).
* Regression objectives: ``bias = base_score`` added directly.
* Multi-class objectives: no bias (base score is zero for each class).

XGBoost's tree-branching condition *"go to yes-child when
x < split_condition"* maps to:

* ``BRANCH_LT`` (mode 1) for ``ai.onnx.ml`` opset ≥ 5 — exact match.
* ``BRANCH_LEQ`` (mode 0 / string ``"BRANCH_LEQ"``) for older opsets —
  differs only at the exact threshold value (rarely relevant for
  floating-point features).
"""

import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import onnx
import onnx.helper as oh
from ...typing import GraphBuilderExtendedProtocol
from ..tree.decision_tree import _get_ml_opset, _get_input_dtype

# Mode encoding for ai.onnx.ml opset-5 TreeEnsemble.
# 0 = BRANCH_LEQ, 1 = BRANCH_LT (exact match for XGBoost's x < threshold).
_NODE_MODE_LT = np.uint8(1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_nodes(node: dict, nodes_dict: dict) -> None:
    """Recursively collect all nodes in an XGBoost JSON tree into a flat dict.

    :param node: root node dict from ``booster.get_dump(dump_format='json')``
    :param nodes_dict: output dict mapping ``nodeid`` → node dict
    """
    nodeid = node["nodeid"]
    nodes_dict[nodeid] = node
    for child in node.get("children", []):
        _collect_nodes(child, nodes_dict)


def _parse_feature_idx(
    split_name: str, feature_name_to_idx: Optional[dict] = None
) -> int:
    """Return the 0-based integer index for an XGBoost feature name.

    When no feature names were supplied to XGBoost during training, the JSON
    dump uses the default ``"f<index>"`` format.  If explicit feature names
    were used, ``feature_name_to_idx`` provides the name-to-index mapping.

    :param split_name: the ``split`` field from the XGBoost JSON tree node
    :param feature_name_to_idx: optional dict mapping feature name → index
    :return: 0-based integer feature index
    :raises ValueError: if the feature name cannot be parsed
    """
    if feature_name_to_idx and split_name in feature_name_to_idx:
        return feature_name_to_idx[split_name]
    if split_name.startswith("f") and split_name[1:].isdigit():
        return int(split_name[1:])
    raise ValueError(
        f"Cannot parse XGBoost feature name {split_name!r}. "
        "Ensure that feature names follow the 'f<index>' convention or "
        "that a feature_name_to_idx mapping is available."
    )


def _get_base_score(booster) -> float:
    """Return the raw (prediction-space) ``base_score`` from an XGBoost booster.

    The value is read from ``booster.save_config()`` which returns the stored
    JSON configuration.  In XGBoost 2.x this is the untransformed
    prediction-space value (e.g. ``0.5`` for the default binary classification
    base score).  Falls back to ``0.5`` if the field cannot be read.

    :param booster: fitted :class:`xgboost.Booster`
    :return: raw base score as a Python float
    """
    try:
        cfg = json.loads(booster.save_config())
        return float(cfg["learner"]["learner_model_param"]["base_score"])
    except Exception:
        return 0.5


def _build_xgb_tree_attrs_legacy(
    trees_json: List[dict],
    n_targets: int,
    feature_name_to_idx: Optional[dict] = None,
) -> dict:
    """Build legacy ``TreeEnsembleRegressor`` attribute arrays from XGBoost trees.

    All trees are encoded into a single flat set of arrays suitable for the
    ``ai.onnx.ml ≤ 4`` ``TreeEnsembleRegressor`` operator.

    The branching direction maps XGBoost's *yes* child (``x < threshold``) to
    the ONNX ``BRANCH_LEQ`` *true* branch (``x ≤ threshold``).  For
    floating-point features this is equivalent except at the exact threshold
    value.

    Each tree is assigned ``target_id = tree_idx % n_targets`` so that for
    multi-class models (``n_targets = n_classes``) tree contributions are
    routed to the correct class channel; for binary/regression models
    ``n_targets = 1`` and all trees contribute to target 0.

    :param trees_json: list of root node dicts from
        ``booster.get_dump(dump_format='json')``
    :param n_targets: number of output targets (1 for binary/regression,
        ``n_classes`` for multi-class)
    :param feature_name_to_idx: optional feature-name → index mapping
    :return: flat attribute dict for ``TreeEnsembleRegressor``
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

    for tree_idx, tree_root in enumerate(trees_json):
        nodes: dict = {}
        _collect_nodes(tree_root, nodes)
        target_id = tree_idx % n_targets

        for node_id in sorted(nodes.keys()):
            node = nodes[node_id]
            all_nodeids.append(node_id)
            all_treeids.append(tree_idx)
            all_hitrates.append(1.0)

            if "leaf" in node:
                # Leaf node
                all_featureids.append(0)
                all_values.append(0.0)
                all_modes.append("LEAF")
                all_truenodeids.append(0)
                all_falsenodeids.append(0)
                all_mvt.append(0)

                all_target_nodeids.append(node_id)
                all_target_treeids.append(tree_idx)
                all_target_ids.append(target_id)
                all_target_weights.append(float(node["leaf"]))
            else:
                # Internal node
                feat_idx = _parse_feature_idx(node["split"], feature_name_to_idx)
                yes_id = int(node["yes"])
                no_id = int(node["no"])
                missing_id = int(node.get("missing", node["yes"]))

                all_featureids.append(feat_idx)
                all_values.append(float(node["split_condition"]))
                # XGBoost: yes = x < threshold (BRANCH_LT semantics).
                # Encoded as BRANCH_LEQ for legacy opset compatibility.
                all_modes.append("BRANCH_LEQ")
                all_truenodeids.append(yes_id)
                all_falsenodeids.append(no_id)
                # 1 when NaN routes to the yes (true) branch, else 0.
                all_mvt.append(1 if missing_id == yes_id else 0)

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


def _build_xgb_tree_attrs_v5(
    trees_json: List[dict],
    n_targets: int,
    dtype=None,
    feature_name_to_idx: Optional[dict] = None,
) -> dict:
    """Build ``TreeEnsemble`` (``ai.onnx.ml`` opset 5) attribute arrays.

    The opset-5 encoding stores *internal* nodes and *leaf* nodes in separate
    flat arrays indexed by contiguous offsets.  ``BRANCH_LT`` (mode 1) is
    used so that XGBoost's ``x < threshold`` condition is matched exactly.

    :param trees_json: list of root node dicts from
        ``booster.get_dump(dump_format='json')``
    :param n_targets: number of output targets
    :param dtype: numpy float dtype for splits / weights; defaults to
        ``np.float32``
    :param feature_name_to_idx: optional feature-name → index mapping
    :return: attribute dict for ``TreeEnsemble``
    """
    if dtype is None:
        dtype = np.float32
    onnx_float_type = (
        onnx.TensorProto.DOUBLE if dtype == np.float64 else onnx.TensorProto.FLOAT
    )

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

    for tree_idx, tree_root in enumerate(trees_json):
        nodes: dict = {}
        _collect_nodes(tree_root, nodes)
        target_id = tree_idx % n_targets

        # Partition nodes into internal (has children) and leaf (no children).
        internal_nodes = sorted(nid for nid, n in nodes.items() if "leaf" not in n)
        leaf_nodes = sorted(nid for nid, n in nodes.items() if "leaf" in n)

        internal_idx = {nid: i for i, nid in enumerate(internal_nodes)}
        leaf_idx = {nid: i for i, nid in enumerate(leaf_nodes)}

        n_internal = len(internal_nodes)
        n_leaves = len(leaf_nodes)

        all_tree_roots.append(cumulative_internal_offset)

        # Build nodes_* arrays.
        if not internal_nodes:
            # Degenerate tree (single leaf root): emit one dummy internal node.
            all_nodes_featureids.append(0)
            all_nodes_splits.append(0.0)
            all_nodes_modes.append(int(_NODE_MODE_LT))
            all_nodes_truenodeids.append(cumulative_leaf_offset)
            all_nodes_trueleafs.append(1)
            all_nodes_falsenodeids.append(cumulative_leaf_offset)
            all_nodes_falseleafs.append(1)
        else:
            for nid in internal_nodes:
                node = nodes[nid]
                feat_idx = _parse_feature_idx(node["split"], feature_name_to_idx)
                yes_id = int(node["yes"])
                no_id = int(node["no"])

                all_nodes_featureids.append(feat_idx)
                all_nodes_splits.append(float(node["split_condition"]))
                all_nodes_modes.append(int(_NODE_MODE_LT))

                # True branch → yes child (x < threshold)
                yes_is_leaf = "leaf" in nodes[yes_id]
                if yes_is_leaf:
                    all_nodes_truenodeids.append(
                        cumulative_leaf_offset + leaf_idx[yes_id]
                    )
                    all_nodes_trueleafs.append(1)
                else:
                    all_nodes_truenodeids.append(
                        cumulative_internal_offset + internal_idx[yes_id]
                    )
                    all_nodes_trueleafs.append(0)

                # False branch → no child (x >= threshold)
                no_is_leaf = "leaf" in nodes[no_id]
                if no_is_leaf:
                    all_nodes_falsenodeids.append(
                        cumulative_leaf_offset + leaf_idx[no_id]
                    )
                    all_nodes_falseleafs.append(1)
                else:
                    all_nodes_falsenodeids.append(
                        cumulative_internal_offset + internal_idx[no_id]
                    )
                    all_nodes_falseleafs.append(0)

        # Build leaf_* arrays.
        if not leaf_nodes:
            # Degenerate tree: single dummy leaf.
            all_leaf_targetids.append(target_id)
            all_leaf_weights.append(0.0)
        else:
            for nid in leaf_nodes:
                all_leaf_targetids.append(target_id)
                all_leaf_weights.append(float(nodes[nid]["leaf"]))

        cumulative_internal_offset += max(n_internal, 1)
        cumulative_leaf_offset += max(n_leaves, 1)

    nodes_splits_tensor = oh.make_tensor(
        "nodes_splits",
        onnx_float_type,
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
        onnx_float_type,
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


def _compute_margin_bias(base_score: float, objective: str) -> float:
    """Compute the margin-space bias from the raw ``base_score``.

    XGBoost 2.x stores ``base_score`` in the prediction space.  The value
    must be transformed to margin (log-odds) space before adding to the raw
    tree output:

    * Binary logistic: ``logit(base_score)`` — zero for the default 0.5.
    * Regression (``reg:*`` / ``count:*`` / etc.): use ``base_score`` directly.
    * Multi-class: return 0 (no single bias applies across classes).

    :param base_score: untransformed base score from the model config
    :param objective: XGBoost objective string (e.g. ``"binary:logistic"``)
    :return: margin-space bias to add to tree output
    """
    if "binary" in objective or "logistic" in objective:
        # Transform probability → log-odds; clamp to avoid log(0).
        p = float(np.clip(base_score, 1e-7, 1.0 - 1e-7))
        return float(np.log(p / (1.0 - p)))
    if "softmax" in objective or "softprob" in objective or "multi" in objective:
        return 0.0
    # Regression objectives: base_score is in prediction space, added directly.
    return float(base_score)


def _emit_tree_node(
    g: GraphBuilderExtendedProtocol,
    X: str,
    name: str,
    n_targets: int,
    trees_json: List[dict],
    feature_name_to_idx: Optional[dict],
    ml_opset: int,
    dtype,
    intermediate_name: Optional[str] = None,
) -> str:
    """Emit a ``TreeEnsembleRegressor`` / ``TreeEnsemble`` ONNX node.

    :param g: graph builder
    :param X: input tensor name
    :param name: node name prefix
    :param n_targets: number of output targets
    :param trees_json: parsed XGBoost tree dicts
    :param feature_name_to_idx: optional feature-name → index mapping
    :param ml_opset: ``ai.onnx.ml`` opset version
    :param dtype: numpy float dtype for numeric attributes
    :param intermediate_name: if provided, use this as the output name;
        otherwise let the builder choose
    :return: output tensor name (shape ``[N, n_targets]``)
    """
    out_arg = [intermediate_name] if intermediate_name else 1

    if ml_opset >= 5:
        attrs = _build_xgb_tree_attrs_v5(
            trees_json,
            n_targets=n_targets,
            dtype=dtype,
            feature_name_to_idx=feature_name_to_idx,
        )
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
    else:
        attrs = _build_xgb_tree_attrs_legacy(
            trees_json,
            n_targets=n_targets,
            feature_name_to_idx=feature_name_to_idx,
        )
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
# XGBClassifier converter
# ---------------------------------------------------------------------------


def sklearn_xgb_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator,
    X: str,
    name: str = "xgb_classifier",
) -> Tuple[str, str]:
    """Convert an :class:`xgboost.XGBClassifier` to ONNX.

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
    :param estimator: a fitted ``XGBClassifier``
    :param X: input tensor name
    :param name: prefix for node names added to the graph
    :return: tuple ``(label_result_name, proba_result_name)``
    """
    booster = estimator.get_booster()
    n_classes: int = int(estimator.n_classes_)
    is_binary: bool = n_classes == 2
    objective: str = estimator.objective or "binary:logistic"

    ml_opset = _get_ml_opset(g)
    dtype = _get_input_dtype(g, X)

    trees_json = [json.loads(t) for t in booster.get_dump(dump_format="json")]
    feature_names = booster.feature_names
    feature_name_to_idx = (
        {fn: i for i, fn in enumerate(feature_names)} if feature_names else None
    )

    n_targets = 1 if is_binary else n_classes

    raw_scores = _emit_tree_node(
        g,
        X,
        name,
        n_targets=n_targets,
        trees_json=trees_json,
        feature_name_to_idx=feature_name_to_idx,
        ml_opset=ml_opset,
        dtype=dtype,
    )

    classes = estimator.classes_

    if is_binary:
        # Add margin-space bias from base_score (zero for default 0.5).
        bias = _compute_margin_bias(_get_base_score(booster), objective)
        if abs(bias) > 1e-8:
            bias_arr = np.array([bias], dtype=np.float32)
            raw_scores = g.op.Add(raw_scores, bias_arr, name=f"{name}_bias")

        # sigmoid → [N, 1] probability of positive class
        p1 = g.op.Sigmoid(raw_scores, name=f"{name}_sigmoid")

        # p0 = 1 - p1 → [N, 1]
        ones = np.array([1.0], dtype=np.float32)
        p0 = g.op.Sub(ones, p1, name=f"{name}_p0")

        # Concat [p0, p1] → [N, 2]
        proba = g.op.Concat(p0, p1, axis=1, name=f"{name}_concat", outputs=outputs[1:])
        assert isinstance(proba, str)

        # Label via ArgMax → [N]
        label_idx = g.op.ArgMax(proba, axis=1, keepdims=0, name=f"{name}_argmax")
        label_idx_i64 = g.op.Cast(
            label_idx, to=onnx.TensorProto.INT64, name=f"{name}_cast"
        )
    else:
        # Multi-class: softmax → [N, n_classes]
        proba = g.op.Softmax(
            raw_scores, axis=1, name=f"{name}_softmax", outputs=outputs[1:]
        )
        assert isinstance(proba, str)

        # Label via ArgMax → [N]
        label_idx = g.op.ArgMax(proba, axis=1, keepdims=0, name=f"{name}_argmax")
        label_idx_i64 = g.op.Cast(
            label_idx, to=onnx.TensorProto.INT64, name=f"{name}_cast"
        )

    label = _gather_labels(g, label_idx_i64, classes, name, outputs[0])
    return label, proba


# ---------------------------------------------------------------------------
# XGBRegressor converter
# ---------------------------------------------------------------------------


def sklearn_xgb_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator,
    X: str,
    name: str = "xgb_regressor",
) -> str:
    """Convert an :class:`xgboost.XGBRegressor` to ONNX.

    The raw margin (sum of all tree leaf values) is computed via a
    ``TreeEnsembleRegressor`` / ``TreeEnsemble`` node and then the XGBoost
    ``base_score`` is added as a constant bias.

    :param g: the graph builder to add nodes to
    :param sts: shapes dict (passed through, not used internally)
    :param outputs: desired output names ``[predictions]``
    :param estimator: a fitted ``XGBRegressor``
    :param X: input tensor name
    :param name: prefix for node names added to the graph
    :return: output tensor name (shape ``[N, 1]``)
    """
    booster = estimator.get_booster()
    objective: str = estimator.objective or "reg:squarederror"

    ml_opset = _get_ml_opset(g)
    dtype = _get_input_dtype(g, X)

    trees_json = [json.loads(t) for t in booster.get_dump(dump_format="json")]
    feature_names = booster.feature_names
    feature_name_to_idx = (
        {fn: i for i, fn in enumerate(feature_names)} if feature_names else None
    )

    # Detect float64 input: TreeEnsembleRegressor/TreeEnsemble always outputs
    # float32 per the ONNX ML spec, so we may need to cast back to float64.
    itype = g.get_type(X) if g.has_type(X) else onnx.TensorProto.FLOAT
    need_cast = itype == onnx.TensorProto.DOUBLE
    tree_out_name = f"{outputs[0]}_tree_out" if need_cast else None

    raw_scores = _emit_tree_node(
        g,
        X,
        name,
        n_targets=1,
        trees_json=trees_json,
        feature_name_to_idx=feature_name_to_idx,
        ml_opset=ml_opset,
        dtype=dtype,
        intermediate_name=tree_out_name,
    )

    # Add base_score bias (regression: base_score is in prediction space).
    base_score = _get_base_score(booster)
    bias = _compute_margin_bias(base_score, objective)
    if abs(bias) > 1e-8:
        bias_arr = np.array([bias], dtype=np.float32)
        raw_scores = g.op.Add(raw_scores, bias_arr, name=f"{name}_bias")

    if not need_cast:
        result = g.op.Identity(raw_scores, name=f"{name}_out", outputs=outputs)
        return result if isinstance(result, str) else result[0]

    # Correct the inferred float64 type before the Cast so the CastPattern
    # optimiser does not eliminate it.
    g._known_types[raw_scores] = onnx.TensorProto.FLOAT  # type: ignore[attr-defined]
    cast_result = g.make_node(
        "Cast",
        [raw_scores],
        outputs=outputs,
        name=f"{name}_cast_f64",
        to=onnx.TensorProto.DOUBLE,
    )
    return cast_result if isinstance(cast_result, str) else cast_result[0]
