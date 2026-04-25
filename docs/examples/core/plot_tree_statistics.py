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
# 1. Train a scikit-learn RandomForestClassifier on dummy data
# -------------------------------------------------------------
#
# We generate a small binary-classification dataset with four features and
# train a five-tree random forest with depth ≤ 4.

import numpy as np
from sklearn.ensemble import RandomForestClassifier

rng = np.random.default_rng(0)
X_train = rng.standard_normal((200, 4)).astype(np.float32)
y_train = (X_train[:, 0] + X_train[:, 2] > 0).astype(int)

clf = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0)
clf.fit(X_train, y_train)
print(
    f"Trained RandomForestClassifier: {clf.n_estimators} trees, "
    f"{clf.n_features_in_} features, classes={list(clf.classes_)}"
)

# %%
# 2. Convert to ONNX with skl2onnx
# ---------------------------------
#
# :func:`skl2onnx.convert_sklearn` produces a ``TreeEnsembleClassifier`` node
# in the ``ai.onnx.ml`` domain — the operator supported by
# :func:`~yobx.helpers.stats_tree_ensemble`.

from skl2onnx import convert_sklearn  # noqa: E402
from skl2onnx.common.data_types import FloatTensorType  # noqa: E402

model = convert_sklearn(clf, "random_forest", [("X", FloatTensorType([None, X_train.shape[1]]))])

print("Graph nodes:", [(n.op_type, n.domain) for n in model.graph.node])

# %%
# 3. Compute statistics for the tree-ensemble node
# -------------------------------------------------
#
# :func:`~yobx.helpers.enumerate_stats_nodes` walks the model graph and
# returns a :class:`~yobx.helpers.NodeStatistics` for every
# ``TreeEnsembleClassifier`` / ``TreeEnsembleRegressor`` it encounters.

from yobx.helpers import enumerate_stats_nodes  # noqa: E402

# Collect results from the full model walk
all_stats = list(enumerate_stats_nodes(model))
print(f"\nNumber of tree-ensemble nodes found: {len(all_stats)}")

# For the single classifier node, inspect the statistics directly
_path, _parent, stats = all_stats[0]
print("kind      :", stats["kind"])
print("n_trees   :", stats["n_trees"])
print("n_outputs :", stats["n_outputs"])
print("n_features:", stats["n_features"])
print("n_rules   :", stats["n_rules"])
print("rules     :", stats["rules"])
print("hist_rules:", stats["hist_rules"])

# %%
# 4. Per-tree breakdown
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
# 5. Per-feature threshold distribution
# ---------------------------------------
#
# For each input feature that appears as a split condition,
# :class:`~yobx.helpers.HistTreeStatistics` stores the distribution of
# threshold values used across all trees.

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
# 6. Flat dictionary for DataFrame integration
# ---------------------------------------------
#
# :meth:`~yobx.helpers.NodeStatistics.dict_values` flattens all scalar
# statistics into a single dict suitable for creating a pandas DataFrame row.

row = stats.dict_values
print("\nFlat stats dict:")
for k, v in sorted(row.items()):
    print(f"  {k}: {v}")

# %%
# 7. Visualize the ONNX graph
# ----------------------------
#
# Finally, we render the ONNX graph so you can inspect nodes and tensor shapes.

from yobx.doc import plot_dot  # noqa: E402

plot_dot(model)
