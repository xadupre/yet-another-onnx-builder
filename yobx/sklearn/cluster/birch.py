from typing import Dict, List, Tuple, Union

import numpy as np
import onnx
from sklearn.cluster import Birch

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(Birch)
def sklearn_birch(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: Birch,
    X: str,
    name: str = "birch",
) -> Union[str, Tuple[str, str]]:
    """
    Converts a :class:`sklearn.cluster.Birch` into ONNX.

    The converter produces two outputs: the predicted cluster labels
    (equivalent to :meth:`~sklearn.cluster.Birch.predict`) and the
    Euclidean distances from each sample to every subcluster centre
    (equivalent to :meth:`~sklearn.cluster.Birch.transform`).

    After fitting, :class:`~sklearn.cluster.Birch` exposes
    ``subcluster_centers_`` (shape ``(K, F)``), which are the centroids
    used to assign new samples.  Prediction is nearest-centroid assignment
    based on Euclidean distance.

    **CDist path** (``com.microsoft`` domain available):

    When the ``com.microsoft`` opset is registered in the graph builder,
    the pairwise Euclidean distances are computed by a single
    ``com.microsoft.CDist`` node, which ONNX Runtime executes via a fused
    C++ kernel.

    .. code-block:: text

        X (N,F)  centers (K,F)
              `---- CDist(metric="euclidean") --► distances (N,K)
                                                       │
                              ArgMin(axis=1) ──────────► subcluster_idx (N,)
                                                       │
                              Gather(subcluster_labels_) ► labels (N,)

    **Standard ONNX path** (fallback):

    When the ``com.microsoft`` domain is absent the distances are computed
    via the squared-distance identity:

    .. code-block:: text

        ||x - c||² = ||x||² - 2·x·cᵀ + ||c||²

    .. code-block:: text

        X (N,F)
          │
          ├──Mul──ReduceSum(axis=1, keepdims=1)──────────────────────────────► x_sq (N,1)
          │                                                                         │
          └──MatMul(centersᵀ)────────────────────────────────────────────────► cross (N,K)
                                                                                    │
        c_sq (1,K) ─────────────────────── Add(x_sq) ─── Sub(Mul(2,cross)) ──► sq_dists (N,K)
                                                                                    │
                                                   Sqrt ──────────────────────► distances (N,K)
                                                                                    │
                          ArgMin(axis=1) ──────────────────────────────────► subcluster_idx (N,)
                                                                                    │
                          Gather(subcluster_labels_) ──────────────────────► labels (N,)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``Birch``
    :param outputs: desired output names; ``outputs[0]`` receives the cluster
        labels and ``outputs[1]`` (if present) receives the distances matrix
    :param X: input tensor name
    :param name: prefix names for the added nodes
    :return: tuple ``(labels, distances)`` when two outputs are requested,
        otherwise just ``labels``
    """
    assert isinstance(estimator, Birch), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)
    zero = np.array([0], dtype=dtype)

    centers = estimator.subcluster_centers_.astype(dtype)  # (K, F)

    # ------------------------------------------------------------------ CDist
    if g.has_opset("com.microsoft"):
        centers_name = g.make_initializer(f"{name}_centers", centers)
        distances_raw = g.make_node(
            "CDist",
            [X, centers_name],
            domain="com.microsoft",
            metric="euclidean",
            name=f"{name}_cdist",
        )
        distances_clipped = g.op.Max(distances_raw, zero, name=f"{name}_clip")

    # ------------------------------------------------ Standard ONNX fallback
    else:
        centers_T = centers.T  # (F, K)

        # ||x||² — sum of squares over the feature axis → (N, 1)
        x_sq = g.op.Mul(X, X, name=f"{name}_x_sq")
        x_sq_sum = g.op.ReduceSum(
            x_sq,
            np.array([1], dtype=np.int64),
            keepdims=1,
            name=f"{name}_x_sq_sum",
        )  # (N, 1)

        # ||c||² — precomputed constant for each centre → (1, K)
        c_sq = np.sum(centers**2, axis=1, keepdims=True).T.astype(dtype)  # (1, K)

        # Cross term: X @ centersᵀ → (N, K)
        cross = g.op.MatMul(X, centers_T, name=f"{name}_cross")  # (N, K)

        # Squared distances: x_sq + c_sq - 2 * cross → (N, K)
        two = np.array([2], dtype=dtype)
        two_cross = g.op.Mul(two, cross, name=f"{name}_two_cross")
        sq_plus = g.op.Add(x_sq_sum, c_sq, name=f"{name}_sq_plus")
        sq_dists = g.op.Sub(sq_plus, two_cross, name=f"{name}_sq_dists")

        # Clip negative values to zero before sqrt (numerical safety).
        sq_dists_clipped = g.op.Max(sq_dists, zero, name=f"{name}_sq_clip")
        distances_clipped = g.op.Sqrt(sq_dists_clipped, name=f"{name}_sqrt_tmp")

    n_outputs = len(outputs)

    # Distances output (optional second output).
    if n_outputs >= 2:
        distances = g.op.Identity(
            distances_clipped,
            name=f"{name}_distances",
            outputs=outputs[1:2],
        )
        assert isinstance(distances, str)
        if not sts:
            g.set_type(distances, itype)
    else:
        distances = distances_clipped

    # Nearest subcluster index → (N,)
    subcluster_idx = g.op.ArgMin(
        distances_clipped,
        axis=1,
        keepdims=0,
        name=f"{name}_argmin",
    )

    # Map subcluster indices to final cluster labels via subcluster_labels_.
    # subcluster_labels_ shape: (K,) — a lookup table of int64 values.
    subcluster_labels = estimator.subcluster_labels_.astype(np.int64)
    label_idx = g.op.Gather(
        subcluster_labels,
        subcluster_idx,
        axis=0,
        name=f"{name}_gather",
    )
    labels = g.op.Cast(
        label_idx,
        to=onnx.TensorProto.INT64,
        name=f"{name}_cast",
        outputs=outputs[:1],
    )
    assert isinstance(labels, str)
    if not sts:
        g.set_type(labels, onnx.TensorProto.INT64)

    if n_outputs >= 2:
        return labels, distances
    return labels
