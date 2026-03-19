from typing import Dict, List

import numpy as np
from sklearn.neighbors import RadiusNeighborsTransformer

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from .kneighbors import _compute_pairwise_distances


@register_sklearn_converter(RadiusNeighborsTransformer)
def sklearn_radius_neighbors_transformer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: RadiusNeighborsTransformer,
    X: str,
    name: str = "rnn_transform",
) -> str:
    """
    Converts a :class:`sklearn.neighbors.RadiusNeighborsTransformer` into ONNX.

    The converter produces a **dense** ``(N, M)`` output tensor where ``N`` is
    the number of query samples and ``M`` is the number of training samples.

    * ``mode='connectivity'`` — entry ``(i, j)`` is ``1.0`` when training
      sample ``j`` is within the radius of query point ``i``, and ``0.0``
      otherwise.
    * ``mode='distance'`` — entry ``(i, j)`` is the distance from query point
      ``i`` to training sample ``j`` when ``j`` is within the radius, and
      ``0.0`` otherwise.

    .. note::

        :meth:`sklearn.neighbors.RadiusNeighborsTransformer.transform` returns
        a **sparse** CSR matrix.  The ONNX graph returns the equivalent
        **dense** matrix (i.e. what you would obtain by calling ``.toarray()``
        on the sparse result).

    Supported metrics: ``"sqeuclidean"``, ``"euclidean"``, ``"cosine"``,
    ``"manhattan"`` (aliases: ``"cityblock"``, ``"l1"``), ``"chebyshev"``,
    ``"minkowski"``.  The ``"euclidean"`` and ``"sqeuclidean"`` metrics use
    ``com.microsoft.CDist`` when that domain is registered; all other metrics
    use the standard-ONNX path.

    Full graph structure (standard-ONNX path):

    .. code-block:: text

        X (N, F)
          │
          └─── pairwise distances ─────────────────────────────────────► dists (N, M)
                                                                               │
                                  in_radius = (dists <= radius) ──► mask (N, M) bool
                                                                               │
        mode='connectivity':  Cast(float) ──────────────────────► output (N, M)
        mode='distance':      Where(mask, dists, 0.0) ──────────► output (N, M)

    :param g: graph builder
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a fitted ``RadiusNeighborsTransformer``
    :param X: input tensor name
    :param name: prefix for node names
    :return: output tensor name — dense ``(N, M)`` matrix
    :raises NotImplementedError: if opset < 13 or the metric is not supported
    """
    assert isinstance(estimator, RadiusNeighborsTransformer)
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    opset = g.get_opset("")
    if opset < 13:
        raise NotImplementedError(
            f"RadiusNeighborsTransformer converter requires opset >= 13 "
            f"(ReduceSum with axes as input was added in opset 13), "
            f"but the graph builder has opset {opset}."
        )

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)
    radius = float(estimator.radius)
    mode = estimator.mode
    training_data = estimator._fit_X.astype(dtype)  # (M, F)
    metric = estimator.effective_metric_
    metric_params = estimator.effective_metric_params_

    # 1. Pairwise distances: (N, M)
    dists = _compute_pairwise_distances(
        g, X, training_data, itype, metric, f"{name}_dist", **metric_params
    )

    # 2. Radius mask: (N, M) bool — True where dist <= radius
    radius_const = np.array([radius], dtype=dtype)
    not_in_radius = g.op.Greater(dists, radius_const, name=f"{name}_gt")
    in_radius = g.op.Not(not_in_radius, name=f"{name}_mask")  # (N, M) bool

    # 3. Build output based on mode
    if mode == "connectivity":
        # Cast bool to float: 1.0 where in radius, 0.0 otherwise
        output = g.op.Cast(in_radius, to=itype, name=f"{name}_cast", outputs=outputs[:1])
    else:
        # mode == "distance": scatter actual distances where in radius, 0.0 elsewhere
        zeros = np.array([0.0], dtype=dtype)
        output = g.op.Where(in_radius, dists, zeros, name=f"{name}_where", outputs=outputs[:1])

    assert isinstance(output, str)
    g.set_type(output, itype)

    return output
