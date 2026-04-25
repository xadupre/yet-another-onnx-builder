"""
.. _l-plot-tree-statistics:

Tree-Ensemble Statistics
========================

:func:`~yobx.helpers.stats_tree_ensemble` computes per-tree and per-feature
statistics for ``TreeEnsembleClassifier`` and ``TreeEnsembleRegressor`` ONNX
nodes (from the ``ai.onnx.ml`` domain).

The function returns a :class:`~yobx.helpers.NodeStatistics` instance
containing:

* global counts — number of trees, features, outputs, split modes, …
* per-feature threshold distributions (:class:`~yobx.helpers.HistTreeStatistics`)
* per-tree structure summaries (:class:`~yobx.helpers.TreeStatistics`)

:func:`~yobx.helpers.enumerate_stats_nodes` walks a full
:class:`~onnx.ModelProto` and yields statistics for every matching node
in one call, which is convenient for exploring real-world models built from
scikit-learn ``RandomForestClassifier`` or similar estimators.
"""

# %%
# 1. Build a small TreeEnsembleClassifier ONNX node
# --------------------------------------------------
#
# We manually construct a two-tree ensemble with two input features and two
# output classes.  Each tree has a root split node and two leaf nodes.
#
# **Tree 0** splits on feature 0 at threshold 0.5.
# **Tree 1** splits on feature 1 at threshold -0.3.

import onnx
import onnx.helper as oh

TFLOAT = onnx.TensorProto.FLOAT
TINT64 = onnx.TensorProto.INT64

# fmt: off
tree_node = oh.make_node(
    "TreeEnsembleClassifier",
    inputs=["X"],
    outputs=["label", "probabilities"],
    domain="ai.onnx.ml",
    # --- structural attributes (per node in all trees) ---
    nodes_nodeids=[0, 1, 2,   0, 1, 2],
    nodes_treeids=[0, 0, 0,   1, 1, 1],
    nodes_featureids=[0, 0, 0,   1, 1, 1],
    nodes_modes=["BRANCH_LEQ", "LEAF", "LEAF",
                 "BRANCH_LEQ", "LEAF", "LEAF"],
    nodes_values=[0.5, 0.0, 0.0,   -0.3, 0.0, 0.0],
    nodes_truenodeids=[1, 0, 0,   1, 0, 0],
    nodes_falsenodeids=[2, 0, 0,   2, 0, 0],
    nodes_hitrates=[1.0, 1.0, 1.0,   1.0, 1.0, 1.0],
    nodes_missing_value_tracks_true=[0, 0, 0,   0, 0, 0],
    # --- class-weight attributes ---
    class_ids=[0, 1,   0, 1],
    class_nodeids=[1, 2,   1, 2],
    class_treeids=[0, 0,   1, 1],
    class_weights=[1.0, 1.0,   1.0, 1.0],
    classlabels_int64s=[0, 1],
    post_transform="NONE",
)
# fmt: on

# Wrap the node in a minimal ONNX model so we can run it and pass it to
# enumerate_stats_nodes.
X_vi = oh.make_tensor_value_info("X", TFLOAT, [None, 2])
label_vi = oh.make_tensor_value_info("label", TINT64, [None])
proba_vi = oh.make_tensor_value_info("probabilities", TFLOAT, [None, 2])

graph = oh.make_graph([tree_node], "trees", [X_vi], [label_vi, proba_vi])
model = oh.make_model(
    graph,
    opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("ai.onnx.ml", 3)],
    ir_version=10,
)

print("Number of nodes in graph:", len(model.graph.node))

# %%
# 2. Compute statistics for a single node
# ----------------------------------------
#
# :func:`~yobx.helpers.stats_tree_ensemble` analyses the tree structure
# directly from the node's ONNX attributes and returns a
# :class:`~yobx.helpers.NodeStatistics` object.

from yobx.helpers import stats_tree_ensemble  # noqa: E402

stats = stats_tree_ensemble(model.graph, tree_node)

print("kind      :", stats["kind"])
print("n_trees   :", stats["n_trees"])
print("n_outputs :", stats["n_outputs"])
print("n_features:", stats["n_features"])
print("n_rules   :", stats["n_rules"])
print("rules     :", stats["rules"])
print("hist_rules:", stats["hist_rules"])

# %%
# 3. Per-tree breakdown
# ---------------------
#
# The ``"trees"`` key holds a :class:`~yobx.helpers.TreeStatistics` object
# for each tree in the ensemble.

print(f"\nPer-tree statistics ({stats['n_trees']} trees):")
for tr in stats["trees"]:
    row = tr.dict_values
    print(
        f"  tree {tr.tree_id}:"
        f" n_nodes={row['n_nodes']}"
        f" n_leaves={row['n_leaves']}"
        f" n_features={row['n_features']}"
    )

# %%
# 4. Per-feature threshold distribution
# ---------------------------------------
#
# For each input feature that appears as a split condition,
# :class:`~yobx.helpers.HistTreeStatistics` stores the distribution of
# threshold values across all trees.

print(f"\nPer-feature threshold statistics ({len(stats['features'])} features):")
for feat in stats["features"]:
    row = feat.dict_values
    print(
        f"  feature {feat.featureid}:"
        f" min={row['min']:.3f}"
        f" max={row['max']:.3f}"
        f" mean={row['mean']:.3f}"
        f" n_distinct={row['n_distinct']}"
    )

# %%
# 5. Flat dictionary for DataFrame integration
# ---------------------------------------------
#
# :meth:`~yobx.helpers.NodeStatistics.dict_values` flattens all scalar
# statistics into a single dict suitable for creating a pandas DataFrame row.

row = stats.dict_values
print("\nFlat stats dict:")
for k, v in sorted(row.items()):
    print(f"  {k}: {v}")

# %%
# 6. Walk a full model with enumerate_stats_nodes
# ------------------------------------------------
#
# :func:`~yobx.helpers.enumerate_stats_nodes` yields statistics for every
# ``TreeEnsembleClassifier`` / ``TreeEnsembleRegressor`` node it encounters
# while walking the model graph.

from yobx.helpers import enumerate_stats_nodes  # noqa: E402

rows = []
for path, _parent, node_stats in enumerate_stats_nodes(model):
    row = node_stats.dict_values
    row["path"] = "/".join(path)
    rows.append(row)

print(f"\nNodes with tree-ensemble statistics: {len(rows)}")
for r in rows:
    print(f"  path={r['path']}  n_trees={r['n_trees']}  kind={r['kind']}")

# %%
# 7. Visualize the ONNX graph
# ----------------------------
#
# Finally, we render the graph so you can inspect nodes and tensor shapes.

from yobx.doc import plot_dot  # noqa: E402

plot_dot(model)
