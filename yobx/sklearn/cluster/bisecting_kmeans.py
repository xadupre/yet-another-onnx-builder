from itertools import count
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from sklearn.cluster import BisectingKMeans

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


def _collect_leaf_paths(node, X_mean: np.ndarray, path_conditions: list) -> list:
    """
    Recursively collect ``(label, conditions)`` tuples for every leaf of the
    bisection tree.

    Each condition is a triple ``(left_center, right_center, go_left)`` where:

    * ``left_center`` / ``right_center`` are the effective (X_mean-adjusted)
      centres of the two children of an internal node.
    * ``go_left`` is ``True`` if a sample should go to the left child at that
      node (i.e. it is closer to ``left_center`` than to ``right_center``).
    """
    if node.left is None:
        return [(node.label, path_conditions)]
    left_c = (node.left.center + X_mean).copy()
    right_c = (node.right.center + X_mean).copy()
    condition = (left_c, right_c)
    return [
        *_collect_leaf_paths(node.left, X_mean, [*path_conditions, (condition, True)]),
        *_collect_leaf_paths(node.right, X_mean, [*path_conditions, (condition, False)]),
    ]


@register_sklearn_converter(BisectingKMeans)
def sklearn_bisecting_kmeans(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: BisectingKMeans,
    X: str,
    name: str = "bisecting_kmeans",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.cluster.BisectingKMeans` into ONNX.

    The converter produces two outputs:

    * **labels** — predicted cluster index for each sample, replicating
      :meth:`~sklearn.cluster.BisectingKMeans.predict` by traversing the
      internal bisection tree.
    * **distances** — Euclidean distances from each sample to every cluster
      centre, replicating
      :meth:`~sklearn.cluster.BisectingKMeans.transform`.

    **Label assignment via tree traversal**

    :class:`~sklearn.cluster.BisectingKMeans` assigns labels by walking down
    the hierarchical bisection tree rather than finding the globally nearest
    centre.  At each internal node the sample is routed to the child whose
    (X_mean-adjusted) centre is closest.  Reached at a leaf, the leaf's label
    is assigned.

    Each internal node defines a linear decision boundary:

    .. code-block:: text

        go_left  iff  X · (right_eff - left_eff) ≤ (||right_eff||² - ||left_eff||²) / 2

    which requires only one ``MatMul`` per internal node.  A sample reaches
    a leaf when all conditions on the root-to-leaf path are satisfied.  Since
    leaves are mutually exclusive and exhaustive, the final label is computed
    as:

    .. code-block:: text

        label = Σ_leaf  leaf_label × Cast(leaf_mask, int64)

    **Distance computation** (transform output):

    When the ``com.microsoft`` opset is available the ``CDist`` operator is
    used directly (``metric="euclidean"``).  Otherwise the distances are
    computed manually:

    .. code-block:: text

        ||x - c||² = ||x||² - 2·x·cᵀ + ||c||²  →  Sqrt

    Full graph structure (three-cluster example):

    .. code-block:: text

        X (N,F)
          │
          ├──Mul──ReduceSum(axis=1,keepdims=1)──────────────────────► x_sq (N,1)
          │                                                                │
          └──MatMul(centersᵀ)──────────────────────────────────────► cross (N,K)
                                                                           │
        c_sq (1,K) ──────────────── Add(x_sq) ── Sub(2·cross) ──► sq_dists (N,K)
                                                                           │
                                            Sqrt ──────────────────► distances (N,K)
          │
          ├──MatMul(dir_0)──LessOrEqual(thr_0) ──► go_left_0 (N,)
          │     …
          └── label accumulation via And / Cast / Mul / Add ──► labels (N,)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``BisectingKMeans``
    :param outputs: desired output names; ``outputs[0]`` receives the cluster
        labels and ``outputs[1]`` receives the distances matrix
    :param X: input tensor name
    :param name: prefix names for the added nodes
    :return: tuple ``(labels, distances)``
    """
    assert isinstance(
        estimator, BisectingKMeans
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    X_mean = (
        estimator._X_mean.astype(dtype)
        if hasattr(estimator, "_X_mean")
        else np.zeros(estimator.cluster_centers_.shape[1], dtype=dtype)
    )

    # ------------------------------------------------------------------
    # Distances output (transform):  Euclidean distances to each center.
    # Use com.microsoft CDist when available; fall back to the manual
    # ||x - c||² = ||x||² - 2·x·cᵀ + ||c||² path otherwise.
    # ------------------------------------------------------------------
    centers = estimator.cluster_centers_.astype(dtype)  # (K, F)
    zero = np.array([0], dtype=dtype)

    if g.has_opset("com.microsoft"):
        centers_name = g.make_initializer(f"{name}_centers", centers)
        cdist_out = g.make_node(
            "CDist",
            [X, centers_name],
            domain="com.microsoft",
            metric="euclidean",
            name=f"{name}_cdist",
        )
        eucl_dists = g.op.Max(cdist_out, zero, name=f"{name}_clip")
    else:
        centers_T = centers.T  # (F, K)
        x_sq = g.op.Mul(X, X, name=f"{name}_x_sq")
        x_sq_sum = g.op.ReduceSum(
            x_sq, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_x_sq_sum"
        )  # (N, 1)
        c_sq = (np.sum(centers**2, axis=1, keepdims=True).T).astype(dtype)  # (1, K)
        cross = g.op.MatMul(X, centers_T, name=f"{name}_cross")  # (N, K)
        two = np.array([2], dtype=dtype)
        two_cross = g.op.Mul(two, cross, name=f"{name}_two_cross")
        sq_plus = g.op.Add(x_sq_sum, c_sq, name=f"{name}_sq_plus")
        sq_dists = g.op.Sub(sq_plus, two_cross, name=f"{name}_sq_dists")
        sq_dists_clipped = g.op.Max(sq_dists, zero, name=f"{name}_clip")
        eucl_dists = g.op.Sqrt(sq_dists_clipped, name=f"{name}_sqrt")

    # Distances: Euclidean distance from each sample to every cluster centre → (N, K).
    distances = g.op.Identity(eucl_dists, name=f"{name}_distances", outputs=outputs[1:2])
    g.set_type(distances, itype)

    # ------------------------------------------------------------------
    # Labels output: bisection-tree traversal
    # ------------------------------------------------------------------
    # Collect all leaf paths from the tree.
    leaf_paths = _collect_leaf_paths(estimator._bisecting_tree, X_mean, [])

    # Cache go_left masks for each (left_center, right_center) pair so that
    # shared internal nodes are only computed once.
    node_go_left_cache: Dict[bytes, str] = {}

    def _go_left_mask(left_c: np.ndarray, right_c: np.ndarray, node_idx: int) -> str:
        """
        Returns the name of a boolean tensor of shape (N,) that is True for
        samples closer to ``left_c`` than to ``right_c``.

        The decision is:
            X · direction ≤ threshold
        where
            direction = right_c - left_c
            threshold = (||right_c||² - ||left_c||²) / 2
        """
        cache_key = left_c.tobytes() + right_c.tobytes()
        if cache_key in node_go_left_cache:
            return node_go_left_cache[cache_key]

        direction = (right_c - left_c).reshape(-1, 1).astype(dtype)  # (F, 1)
        threshold = np.array(
            [(float(np.dot(right_c, right_c)) - float(np.dot(left_c, left_c))) / 2.0], dtype=dtype
        )  # (1,)

        dot = g.op.MatMul(X, direction, name=f"{name}_node{node_idx}_dot")  # (N, 1)
        dot_1d = g.op.Squeeze(
            dot, np.array([1], dtype=np.int64), name=f"{name}_node{node_idx}_1d"
        )  # (N,)
        mask = g.op.LessOrEqual(dot_1d, threshold, name=f"{name}_node{node_idx}_mask")  # (N,)

        node_go_left_cache[cache_key] = mask  # type: ignore
        return mask

    # Build label tensor as the sum of (leaf_mask * leaf_label) over all leaves.
    # Leaves are mutually exclusive and exhaustive, so this equals the correct label.
    label_sum: Optional[str] = None
    node_counter = count(1)  # unique node-name counter

    for leaf_label, conditions in leaf_paths:
        # Compute the leaf mask = AND of all path conditions.
        leaf_mask: Optional[str] = None
        for (left_c, right_c), go_left in conditions:
            node_idx = next(node_counter)
            go_left_mask = _go_left_mask(left_c, right_c, node_idx)
            cond_mask = (
                go_left_mask if go_left else g.op.Not(go_left_mask, name=f"{name}_not{node_idx}")
            )
            if leaf_mask is None:
                leaf_mask = cond_mask
            else:
                and_idx = next(node_counter)
                leaf_mask = g.op.And(leaf_mask, cond_mask, name=f"{name}_and{and_idx}")

        if leaf_mask is None:
            # Degenerate case: single-leaf tree (n_clusters=1).  All samples
            # get the same label.
            leaf_label_val = np.array([leaf_label], dtype=np.int64)
            label_sum = g.op.Expand(
                leaf_label_val,
                g.op.Shape(X, name=f"{name}_shape")[:1],
                name=f"{name}_expand_leaf",
            )
            break

        # Cast boolean mask to int64 and scale by the leaf label.
        leaf_mask_int = g.op.Cast(
            leaf_mask, to=onnx.TensorProto.INT64, name=f"{name}_leafmask{leaf_label}_int"
        )
        if leaf_label == 0:
            # Contribution is zero regardless; skip the Mul.
            contrib: Optional[str] = None
        else:
            leaf_label_val = np.array([leaf_label], dtype=np.int64)
            contrib = g.op.Mul(leaf_mask_int, leaf_label_val, name=f"{name}_contrib{leaf_label}")

        if label_sum is None:
            label_sum = contrib  # may be None if leaf_label == 0
        elif contrib is None:
            pass  # nothing to add
        else:
            label_sum = g.op.Add(label_sum, contrib, name=f"{name}_labelacc{leaf_label}")

    # If label_sum is still None (all leaf labels were 0), create a zero tensor.
    if label_sum is None:
        shape_x = g.op.Shape(X, name=f"{name}_shape_x")
        n_dim = g.op.Gather(shape_x, np.array(0, dtype=np.int64), name=f"{name}_n_dim")
        shape_1d = g.op.Unsqueeze(n_dim, np.array([0], dtype=np.int64), name=f"{name}_shape_1d")
        label_sum = g.op.Expand(np.array([0], dtype=np.int64), shape_1d, name=f"{name}_zeros")

    labels = g.op.Identity(label_sum, name=f"{name}_labels", outputs=outputs[:1])
    g.set_type(labels, onnx.TensorProto.INT64)

    return labels, distances
