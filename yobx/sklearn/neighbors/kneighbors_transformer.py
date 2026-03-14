from typing import Dict, List

import numpy as np
from sklearn.neighbors import KNeighborsTransformer

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from .kneighbors import _compute_pairwise_distances


@register_sklearn_converter(KNeighborsTransformer)
def sklearn_kneighbors_transformer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: KNeighborsTransformer,
    X: str,
    name: str = "knn_transform",
) -> str:
    """
    Converts a :class:`sklearn.neighbors.KNeighborsTransformer` into ONNX.

    The converter produces a **dense** ``(N, M)`` output tensor where ``N`` is
    the number of query samples and ``M`` is the number of training samples.

    * ``mode='connectivity'`` — entry ``(i, j)`` is ``1.0`` when training
      sample ``j`` is among the ``n_neighbors`` nearest neighbours of query
      point ``i``, and ``0.0`` otherwise.
    * ``mode='distance'`` — entry ``(i, j)`` is the distance from query point
      ``i`` to training sample ``j`` when ``j`` is among the ``n_neighbors``
      nearest neighbours, and ``0.0`` otherwise.

    .. note::

        :meth:`sklearn.neighbors.KNeighborsTransformer.transform` returns a
        **sparse** CSR matrix.  The ONNX graph returns the equivalent **dense**
        matrix (i.e. what you would obtain by calling ``.toarray()`` on the
        sparse result).

    .. note::

        sklearn's ``transform()`` uses ``n_neighbors + 1`` neighbours
        internally for ``mode='distance'`` to account for self-connections
        when transforming the training set.  This converter always uses
        exactly ``n_neighbors`` neighbours for both modes.  The output
        matches ``sklearn.neighbors.kneighbors(X, n_neighbors)`` applied
        to the query points.  For the training set with ``mode='distance'``,
        one of the ``n_neighbors`` slots may be the query point itself
        (distance ``0.0`` scattered at the diagonal), which is
        indistinguishable from a non-neighbour entry.

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
                                TopK(k, axis=1, largest=0) ──► values (N, k),  indices (N, k)
                                                                               │
                     zeros (1, M)  ──► Expand(N, M) ──► zeros_NM (N, M)        │
                                                              │                │
                          ScatterElements(axis=1) ─────────────────────► output (N, M)

    :param g: graph builder
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a fitted ``KNeighborsTransformer``
    :param X: input tensor name
    :param name: prefix for node names
    :return: output tensor name — dense ``(N, M)`` matrix
    :raises NotImplementedError: if opset < 13 or the metric is not supported
    """
    assert isinstance(estimator, KNeighborsTransformer)
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    opset = g.get_opset("")
    if opset < 13:
        raise NotImplementedError(
            f"KNeighborsTransformer converter requires opset >= 13 "
            f"(ReduceSum with axes as input was added in opset 13), "
            f"but the graph builder has opset {opset}."
        )

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)
    k = estimator.n_neighbors
    mode = estimator.mode
    training_data = estimator._fit_X.astype(dtype)  # (M, F)
    M = training_data.shape[0]
    metric = estimator.effective_metric_
    metric_params = estimator.effective_metric_params_

    # 1. Pairwise distances: (N, M)
    dists = _compute_pairwise_distances(
        g, X, training_data, itype, metric, f"{name}_dist", **metric_params
    )

    # 2. k nearest neighbour distances and indices: (N, k)
    topk_values, topk_indices = g.op.TopK(
        dists,
        np.array([k], dtype=np.int64),
        axis=1,
        largest=0,
        sorted=1,
        name=f"{name}_topk",
    )

    # 3. Build a zero matrix of shape (N, M).
    #    N is dynamic (query batch size); M is fixed (number of training samples).
    #    We use Expand on a static (1, M) zero constant to avoid ConstantOfShape.
    zeros_1M = np.zeros((1, M), dtype=dtype)  # (1, M) constant

    x_shape = g.op.Shape(X, name=f"{name}_x_shape")  # [N, F] as int64 tensor [2]
    N_shape = g.op.Slice(
        x_shape,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        name=f"{name}_N_shape",
    )  # [1] int64 tensor with value N
    M_arr = np.array([M], dtype=np.int64)
    out_shape = g.op.Concat(
        N_shape, M_arr, axis=0, name=f"{name}_out_shape"
    )  # [2] int64 tensor: [N, M]
    zeros = g.op.Expand(zeros_1M, out_shape, name=f"{name}_zeros")  # (N, M)

    # 4. Values to scatter at the k-NN positions.
    if mode == "connectivity":
        # Scatter 1.0 at all k-NN positions.
        zeros_k = g.op.Mul(
            topk_values,
            np.array([0.0], dtype=dtype),
            name=f"{name}_zeros_k",
        )  # (N, k) — all zeros, same shape as topk_values
        scatter_vals = g.op.Add(
            zeros_k,
            np.array([1.0], dtype=dtype),
            name=f"{name}_ones_k",
        )  # (N, k) — all ones
    else:
        # mode == "distance": scatter the actual distances.
        scatter_vals = topk_values

    # 5. Scatter values into the zero matrix at the k-NN column indices.
    output = g.op.ScatterElements(
        zeros,
        topk_indices,
        scatter_vals,
        axis=1,
        name=f"{name}_scatter",
        outputs=outputs[:1],
    )

    assert isinstance(output, str)
    if not sts:
        g.set_type(output, itype)

    return output
