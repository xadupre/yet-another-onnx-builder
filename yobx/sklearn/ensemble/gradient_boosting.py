"""
Converter for :class:`sklearn.ensemble.GradientBoostingClassifier` and
:class:`sklearn.ensemble.GradientBoostingRegressor`.

The ONNX graph mirrors the model's prediction pipeline::

    raw_prediction = baseline + learning_rate * sum(tree_values for all trees)
    # regression:        raw_prediction  →  output (N, 1)
    # binary cls:        Sigmoid(raw)    →  [1-p, p],  ArgMax → label
    # multiclass:        Softmax(raw)    →  proba,     ArgMax → label

where ``baseline`` is the initial raw score from ``init_`` (a constant for
the default :class:`~sklearn.dummy.DummyRegressor` /
:class:`~sklearn.dummy.DummyClassifier` init, or zero when ``init='zero'``).

Two encoding paths are supported:

* **Legacy** (``ai.onnx.ml`` opset ≤ 4): ``TreeEnsembleRegressor``
  with ``aggregate_function="SUM"`` and ``base_values``.
* **Modern** (``ai.onnx.ml`` opset 5): ``TreeEnsemble``
  with ``aggregate_function=1`` (SUM) and a constant ``Add`` for the baseline.

Custom init estimators (other than ``DummyRegressor`` / ``DummyClassifier``
or ``'zero'``) are not supported and raise :class:`NotImplementedError`.
"""

from typing import Dict, List, Tuple
import numpy as np
import onnx
import onnx.helper as oh
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ..tree.decision_tree import _LEAF, _NODE_MODE_LEQ


def _get_gb_baseline(estimator) -> np.ndarray:
    """
    Return the constant initial raw score as a ``float32`` array of shape
    ``(1, K)`` where ``K`` is ``estimators_.shape[1]``.

    Supports ``init='zero'`` (or any value that maps to zero initialisation)
    and the default :class:`~sklearn.dummy.DummyRegressor` /
    :class:`~sklearn.dummy.DummyClassifier` init.  Custom init estimators
    raise :class:`NotImplementedError`.

    :param estimator: fitted ``GradientBoostingRegressor`` or
        ``GradientBoostingClassifier``.
    :return: numpy array of shape ``(1, K)``.
    :raises NotImplementedError: if a custom (non-Dummy) init estimator is used.
    """
    K = estimator.estimators_.shape[1]

    init = estimator.init_
    if isinstance(init, str) and init == "zero":
        return np.zeros((1, K), dtype=np.float32)

    if not isinstance(init, (DummyRegressor, DummyClassifier)):
        raise NotImplementedError(
            f"Custom init estimator of type {type(init).__name__!r} is not "
            "supported by this converter.  Only the default "
            "DummyRegressor / DummyClassifier (init=None) or init='zero' "
            "produce X-independent baselines that can be embedded as ONNX "
            "constants."
        )

    # For DummyRegressor / DummyClassifier the raw init prediction is
    # X-independent.  We compute it once on a single zero row.
    n_features = estimator.n_features_in_
    X_dummy = np.zeros((1, n_features), dtype=np.float32)
    baseline = estimator._raw_predict_init(X_dummy)  # shape (1, K)
    return baseline.astype(np.float32)


# ---------------------------------------------------------------------------
# Legacy attribute extraction (ai.onnx.ml opset ≤ 4)
# ---------------------------------------------------------------------------


def _extract_gb_attributes_legacy(
    estimators: np.ndarray,
    learning_rate: float,
    n_targets: int,
) -> Dict:
    """
    Extract ``TreeEnsembleRegressor`` attributes from the 2-D ``estimators_``
    array of a fitted :class:`~sklearn.ensemble.GradientBoostingRegressor` or
    :class:`~sklearn.ensemble.GradientBoostingClassifier`.

    Each ``estimators_[i, k]`` is a :class:`~sklearn.tree.DecisionTreeRegressor`
    whose leaf values contribute to output target ``k``.  Leaf weights are
    pre-multiplied by ``learning_rate`` so that the ``SUM`` aggregation in the
    ``TreeEnsembleRegressor`` node directly yields the boosted raw score
    (excluding the baseline).

    :param estimators: ``estimator.estimators_`` — shape
        ``(n_estimators, n_targets)``.
    :param learning_rate: ``estimator.learning_rate``.
    :param n_targets: number of output targets (``estimators.shape[1]``).
    :return: dict of ONNX attribute lists ready to pass to ``make_node``.
    """
    n_rounds, K = estimators.shape

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

    tree_id = 0
    for i in range(n_rounds):
        for k in range(K):
            tree = estimators[i, k].tree_
            n_nodes = tree.node_count
            feature = tree.feature
            threshold = tree.threshold.astype(np.float32)
            children_left = tree.children_left
            children_right = tree.children_right
            value = tree.value  # shape (n_nodes, 1, 1)

            for node_id in range(n_nodes):
                all_nodeids.append(node_id)
                all_treeids.append(tree_id)
                all_hitrates.append(1.0)
                all_mvt.append(0)

                left = children_left[node_id]
                right = children_right[node_id]

                if left == _LEAF:
                    all_featureids.append(0)
                    all_values.append(0.0)
                    all_modes.append("LEAF")
                    all_truenodeids.append(0)
                    all_falsenodeids.append(0)

                    leaf_val = float(value[node_id, 0, 0]) * learning_rate
                    all_target_nodeids.append(node_id)
                    all_target_treeids.append(tree_id)
                    all_target_ids.append(k)
                    all_target_weights.append(leaf_val)
                else:
                    all_featureids.append(int(feature[node_id]))
                    all_values.append(float(threshold[node_id]))
                    all_modes.append("BRANCH_LEQ")
                    all_truenodeids.append(int(left))
                    all_falsenodeids.append(int(right))

            tree_id += 1

    return dict(
        n_targets=n_targets,
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


# ---------------------------------------------------------------------------
# Modern attribute extraction (ai.onnx.ml opset 5)
# ---------------------------------------------------------------------------


def _extract_gb_attributes_v5(
    estimators: np.ndarray,
    learning_rate: float,
    n_targets: int,
    itype: int,
) -> Dict:
    """
    Extract ``TreeEnsemble`` (``ai.onnx.ml`` opset 5) attributes from the
    2-D ``estimators_`` array of a fitted GradientBoosting estimator.

    The encoding mirrors
    :func:`~yobx.sklearn.ensemble.hist_gradient_boosting._extract_hgb_attributes_v5`
    but reads from :class:`~sklearn.tree.DecisionTreeRegressor` objects
    (``tree_`` attribute) instead of ``TreePredictor`` objects.

    :param estimators: ``estimator.estimators_`` — shape
        ``(n_estimators, n_targets)``.
    :param learning_rate: ``estimator.learning_rate``.
    :param n_targets: number of output targets (``estimators.shape[1]``).
    :param itype: ONNX tensor type for split thresholds and leaf weights.
    :return: dict of ONNX attributes (including tensor attributes).
    """
    dtype = tensor_dtype_to_np_dtype(itype)
    n_rounds, K = estimators.shape

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

    for i in range(n_rounds):
        for k in range(K):
            sk_tree = estimators[i, k].tree_
            n_nodes = sk_tree.node_count
            feature = sk_tree.feature
            threshold = sk_tree.threshold.astype(dtype)
            children_left = sk_tree.children_left
            children_right = sk_tree.children_right
            value = sk_tree.value  # shape (n_nodes, 1, 1)

            # Partition nodes into internal and leaf sets.
            internal_nodes: List[int] = []
            leaf_nodes: List[int] = []
            for nid in range(n_nodes):
                if children_left[nid] == _LEAF:
                    leaf_nodes.append(nid)
                else:
                    internal_nodes.append(nid)

            internal_idx_map = {nid: idx for idx, nid in enumerate(internal_nodes)}
            leaf_idx_map = {nid: idx for idx, nid in enumerate(leaf_nodes)}

            n_internal = len(internal_nodes)
            n_leaves = len(leaf_nodes)

            if n_internal == 0:
                # Degenerate single-leaf tree: emit a dummy internal node.
                all_tree_roots.append(cumulative_internal_offset)
                leaf_pos = cumulative_leaf_offset
                all_nodes_featureids.append(0)
                all_nodes_splits.append(0.0)
                all_nodes_modes.append(int(_NODE_MODE_LEQ))
                all_nodes_truenodeids.append(leaf_pos)
                all_nodes_trueleafs.append(1)
                all_nodes_falsenodeids.append(leaf_pos)
                all_nodes_falseleafs.append(1)
                all_leaf_targetids.append(k)
                all_leaf_weights.append(float(value[0, 0, 0]) * learning_rate)
                cumulative_internal_offset += 1
                cumulative_leaf_offset += 1
            else:
                # Root is always node 0 in sklearn decision trees.
                all_tree_roots.append(cumulative_internal_offset + internal_idx_map[0])
                node_offset = cumulative_internal_offset
                leaf_offset = cumulative_leaf_offset

                for nid in internal_nodes:
                    left = int(children_left[nid])
                    right = int(children_right[nid])

                    all_nodes_featureids.append(int(feature[nid]))
                    all_nodes_splits.append(float(threshold[nid]))
                    all_nodes_modes.append(int(_NODE_MODE_LEQ))

                    # True branch → left child (sklearn: feature <= threshold)
                    left_is_leaf = children_left[left] == _LEAF
                    if left_is_leaf:
                        all_nodes_truenodeids.append(leaf_offset + leaf_idx_map[left])
                        all_nodes_trueleafs.append(1)
                    else:
                        all_nodes_truenodeids.append(node_offset + internal_idx_map[left])
                        all_nodes_trueleafs.append(0)

                    # False branch → right child (sklearn: feature > threshold)
                    right_is_leaf = children_left[right] == _LEAF
                    if right_is_leaf:
                        all_nodes_falsenodeids.append(leaf_offset + leaf_idx_map[right])
                        all_nodes_falseleafs.append(1)
                    else:
                        all_nodes_falsenodeids.append(node_offset + internal_idx_map[right])
                        all_nodes_falseleafs.append(0)

                for nid in leaf_nodes:
                    all_leaf_targetids.append(k)
                    all_leaf_weights.append(float(value[nid, 0, 0]) * learning_rate)

                cumulative_internal_offset += n_internal
                cumulative_leaf_offset += n_leaves

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


# ---------------------------------------------------------------------------
# Raw-output builder helpers
# ---------------------------------------------------------------------------


def _build_gb_raw_output_legacy(
    g: GraphBuilderExtendedProtocol,
    X: str,
    name: str,
    estimators: np.ndarray,
    learning_rate: float,
    n_targets: int,
    baseline: np.ndarray,
    raw_outputs: List[str],
) -> str:
    """
    Emit a ``TreeEnsembleRegressor`` node (legacy path, ``ai.onnx.ml`` opset ≤
    4) for a GradientBoosting estimator and return the output tensor name.

    ``base_values`` encodes the constant initial raw score so that the node
    computes ``baseline + learning_rate * sum(tree_values)`` in a single pass.

    :param g: graph builder.
    :param X: input tensor name.
    :param name: node-name prefix.
    :param estimators: ``estimator.estimators_``.
    :param learning_rate: ``estimator.learning_rate``.
    :param n_targets: number of output targets.
    :param baseline: constant initial raw score, shape ``(1, n_targets)``.
    :param raw_outputs: desired output tensor name(s).
    :return: output tensor name.
    """
    attrs = _extract_gb_attributes_legacy(estimators, learning_rate, n_targets)
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


def _build_gb_raw_output_v5(
    g: GraphBuilderExtendedProtocol,
    X: str,
    name: str,
    estimators: np.ndarray,
    learning_rate: float,
    n_targets: int,
    baseline: np.ndarray,
    raw_outputs: List[str],
    itype: int,
) -> str:
    """
    Emit a ``TreeEnsemble`` node (``ai.onnx.ml`` opset 5) followed by an
    ``Add`` for the baseline, and return the output tensor name.

    :param g: graph builder.
    :param X: input tensor name.
    :param name: node-name prefix.
    :param estimators: ``estimator.estimators_``.
    :param learning_rate: ``estimator.learning_rate``.
    :param n_targets: number of output targets.
    :param baseline: constant initial raw score, shape ``(1, n_targets)``.
    :param raw_outputs: desired output tensor name(s).
    :param itype: ONNX tensor type for the tree node.
    :return: output tensor name.
    """
    attrs = _extract_gb_attributes_v5(estimators, learning_rate, n_targets, itype)

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

    # Add the baseline as a constant (broadcast over the batch dimension).
    bv = baseline.flatten().astype(tensor_dtype_to_np_dtype(itype))
    bv_expanded = bv.reshape(1, n_targets)
    add_result = g.op.Add(
        te_out_name,
        bv_expanded,
        name=f"{name}_add_baseline",
        outputs=raw_outputs,
    )
    return add_result if isinstance(add_result, str) else add_result[0]


# ---------------------------------------------------------------------------
# Public converters
# ---------------------------------------------------------------------------


@register_sklearn_converter((GradientBoostingRegressor,))
def sklearn_gradient_boosting_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: GradientBoostingRegressor,
    X: str,
    name: str = "gradient_boosting_regressor",
) -> str:
    """
    Converts a :class:`sklearn.ensemble.GradientBoostingRegressor` into ONNX.

    The prediction is::

        output = baseline + learning_rate * sum(tree.predict(X) for tree in estimators_)

    where ``baseline`` is the constant initial raw score from the ``init_``
    estimator (``DummyRegressor`` by default, or zero when ``init='zero'``).

    When ``ai.onnx.ml`` opset 5 (or later) is active the unified
    ``TreeEnsemble`` operator is used; otherwise the legacy
    ``TreeEnsembleRegressor`` is emitted with ``aggregate_function="SUM"``
    and ``base_values`` carrying the baseline.

    When the input is ``float64`` the output is cast back to ``float64``
    (ONNX ML tree operators always output ``float32``).

    :param g: graph builder.
    :param sts: shapes provided by scikit-learn.
    :param outputs: desired output names.
    :param estimator: fitted ``GradientBoostingRegressor``.
    :param X: input tensor name.
    :param name: node-name prefix.
    :return: output tensor name (shape ``[N, 1]``).
    :raises NotImplementedError: if a custom init estimator is used.
    """
    assert isinstance(estimator, GradientBoostingRegressor)

    baseline = _get_gb_baseline(estimator)  # shape (1, 1)
    learning_rate = estimator.learning_rate
    estimators = estimator.estimators_
    n_targets = 1

    itype = g.get_type(X) if g.has_type(X) else onnx.TensorProto.FLOAT
    tree_outputs = [f"{outputs[0]}_tree_out"]

    ml_opset = g.get_opset("ai.onnx.ml")
    if ml_opset >= 5:
        raw = _build_gb_raw_output_v5(
            g,
            X,
            name,
            estimators,
            learning_rate,
            n_targets,
            baseline,
            tree_outputs,
            itype=itype,
        )
    else:
        raw = _build_gb_raw_output_legacy(
            g,
            X,
            name,
            estimators,
            learning_rate,
            n_targets,
            baseline,
            tree_outputs,
        )

    cast_result = g.make_node(
        "Cast",
        [raw],
        outputs=outputs,
        name=f"{name}_cast_f64",
        to=itype,
    )
    return cast_result if isinstance(cast_result, str) else cast_result[0]


@register_sklearn_converter((GradientBoostingClassifier,))
def sklearn_gradient_boosting_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: GradientBoostingClassifier,
    X: str,
    name: str = "gradient_boosting_classifier",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.ensemble.GradientBoostingClassifier` into ONNX.

    The raw score (logit) per class is::

        raw[:,k] = baseline[k] + learning_rate * sum(trees_k.predict(X))

    **Binary classification** — the raw score (one logit per sample) passes
    through a ``Sigmoid``; the resulting probability ``p`` for class 1 is
    concatenated as ``[1-p, p]`` to match ``predict_proba``.

    **Multiclass** — the raw scores (one logit per class) pass through a
    ``Softmax`` along axis 1.

    In both cases the predicted label is derived via ``ArgMax`` and a
    ``Gather`` into the ``classes_`` array.

    When ``ai.onnx.ml`` opset 5 (or later) is active the unified
    ``TreeEnsemble`` operator is used; otherwise the legacy
    ``TreeEnsembleRegressor`` is emitted.

    :param g: graph builder.
    :param sts: shapes provided by scikit-learn.
    :param outputs: desired output names (label, probabilities).
    :param estimator: fitted ``GradientBoostingClassifier``.
    :param X: input tensor name.
    :param name: node-name prefix.
    :return: tuple ``(label_name, proba_name)``.
    :raises NotImplementedError: if a custom init estimator is used.
    """
    assert isinstance(estimator, GradientBoostingClassifier)

    classes = estimator.classes_
    n_classes = estimator.n_classes_
    is_binary = n_classes == 2  # binary: one tree per iteration (K=1)

    baseline = _get_gb_baseline(estimator)  # shape (1, K)
    learning_rate = estimator.learning_rate
    estimators = estimator.estimators_
    n_targets = estimators.shape[1]  # K: 1 for binary, n_classes for multiclass

    ml_opset = g.get_opset("ai.onnx.ml")
    itype = g.get_type(X)
    raw_name = g.unique_name(f"{name}_raw")

    if ml_opset >= 5:
        raw = _build_gb_raw_output_v5(
            g,
            X,
            name,
            estimators,
            learning_rate,
            n_targets,
            baseline,
            [raw_name],
            itype=itype,
        )
    else:
        raw = _build_gb_raw_output_legacy(
            g,
            X,
            name,
            estimators,
            learning_rate,
            n_targets,
            baseline,
            [raw_name],
        )

    raw_typed = g.op.Cast(raw, to=itype, name=f"{name}_cast_raw")

    if is_binary:
        # raw_typed: (N, 1) → sigmoid → p1, complement → p0, concat → (N, 2)
        p1 = g.op.Sigmoid(raw_typed, name=f"{name}_sigmoid")
        one_cst = np.ones((1, 1), dtype=tensor_dtype_to_np_dtype(itype))
        p0 = g.op.Sub(one_cst, p1, name=f"{name}_p0")
        proba_raw = g.op.Concat(p0, p1, axis=1, name=f"{name}_concat")
    else:
        # raw_typed: (N, n_classes) → softmax → (N, n_classes)
        proba_raw = g.op.Softmax(raw_typed, axis=1, name=f"{name}_softmax")

    proba = g.op.Identity(proba_raw, name=f"{name}_proba", outputs=outputs[1:])
    assert isinstance(proba, str)

    # Predicted label: ArgMax over probabilities.
    label_idx = g.op.ArgMax(proba_raw, axis=1, keepdims=0, name=f"{name}_argmax")
    label_idx_i64 = g.op.Cast(label_idx, to=onnx.TensorProto.INT64, name=f"{name}_cast_idx")

    if np.issubdtype(classes.dtype, np.integer):  # type: ignore
        classes_arr = classes.astype(np.int64)
        label = g.op.Gather(
            classes_arr,
            label_idx_i64,
            axis=0,
            name=f"{name}_label",
            outputs=outputs[:1],
        )
        if not sts:
            g.set_type(label, onnx.TensorProto.INT64)
    else:
        classes_arr = np.array(classes.astype(str))
        label = g.op.Gather(
            classes_arr,
            label_idx_i64,
            axis=0,
            name=f"{name}_label_str",
            outputs=outputs[:1],
        )
        if not sts:
            g.set_type(label, onnx.TensorProto.STRING)

    assert isinstance(label, str)
    return label, proba
