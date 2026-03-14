from typing import Dict, List, Tuple, Union

import numpy as np
import onnx
from sklearn.cluster import MeanShift

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(MeanShift)
def sklearn_mean_shift(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: MeanShift,
    X: str,
    name: str = "mean_shift",
) -> Union[str, Tuple[str, str]]:
    """
    Converts a :class:`sklearn.cluster.MeanShift` into ONNX.

    The converter produces two outputs: the predicted cluster labels
    (equivalent to :meth:`~sklearn.cluster.MeanShift.predict`) and the
    Euclidean distances from each sample to every cluster centre
    (equivalent to the distance matrix used internally by ``predict``).

    The ``predict`` method of :class:`~sklearn.cluster.MeanShift` assigns
    each sample to the nearest cluster centre using Euclidean distance,
    which is reproduced here via the identity:

    .. code-block:: text

        ||x - c||² = ||x||² - 2·x·cᵀ + ||c||²

    Two computation paths are used depending on the available opsets:

    **With** ``com.microsoft`` **opset** (CDist path):

    .. code-block:: text

        X (N,F)  centers (K,F)
              │       │
         com.microsoft.CDist(metric="euclidean") ──► distances (N,K)
                                                          │
                                               ArgMin(axis=1) ──► labels (N,)

    **Without** ``com.microsoft`` **opset** (standard ONNX path):

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
                                               ArgMin(axis=1) ─────────────────► labels (N,)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``MeanShift``
    :param outputs: desired output names; ``outputs[0]`` receives the cluster
        labels and ``outputs[1]`` (if present) receives the distances matrix
    :param X: input tensor name
    :param name: prefix names for the added nodes
    :return: tuple ``(labels, distances)`` when two outputs are requested,
        otherwise just ``labels``
    """
    assert isinstance(
        estimator, MeanShift
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    centers = estimator.cluster_centers_.astype(dtype)  # (K, F)
    zero = np.array([0], dtype=dtype)

    # ------------------------------------------------------------------
    # Euclidean distances: use com.microsoft.CDist when available,
    # fall back to the manual ||x-c||² = ||x||² - 2·x·cᵀ + ||c||² path.
    # ------------------------------------------------------------------
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

        # ||x||² - sum of squares over the feature axis for every sample → (N, 1)
        x_sq = g.op.Mul(X, X, name=f"{name}_x_sq")
        x_sq_sum = g.op.ReduceSum(
            x_sq,
            np.array([1], dtype=np.int64),
            keepdims=1,
            name=f"{name}_x_sq_sum",
        )  # (N, 1)

        # ||c||² - precomputed constant for each centre → (1, K)
        c_sq = np.sum(centers**2, axis=1, keepdims=True).T.astype(dtype)  # (1, K)

        # Cross term: X @ centersᵀ → (N, K)
        cross = g.op.MatMul(X, centers_T, name=f"{name}_cross")  # (N, K)

        # Squared distances: x_sq + c_sq - 2 * cross → (N, K)
        two = np.array([2], dtype=dtype)
        two_cross = g.op.Mul(two, cross, name=f"{name}_two_cross")
        sq_plus = g.op.Add(x_sq_sum, c_sq, name=f"{name}_sq_plus")
        sq_dists = g.op.Sub(sq_plus, two_cross, name=f"{name}_sq_dists")

        # Clip negative values to zero before sqrt (numerical safety).
        sq_dists_clipped = g.op.Max(sq_dists, zero, name=f"{name}_clip")
        eucl_dists = g.op.Sqrt(sq_dists_clipped, name=f"{name}_sqrt")

    n_outputs = len(outputs)

    # Distances output (optional second output).
    if n_outputs >= 2:
        distances = g.op.Identity(
            eucl_dists,
            name=f"{name}_distances",
            outputs=outputs[1:2],
        )
        assert isinstance(distances, str)
        if not sts:
            g.set_type(distances, itype)
    else:
        distances = eucl_dists
        assert isinstance(distances, str)

    # Labels: nearest centre index → (N,)
    label_idx = g.op.ArgMin(
        eucl_dists,
        axis=1,
        keepdims=0,
        name=f"{name}_argmin",
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
