from typing import Dict, List

import numpy as np
import onnx
from sklearn.cluster import FeatureAgglomeration

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(FeatureAgglomeration)
def sklearn_feature_agglomeration(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: FeatureAgglomeration,
    X: str,
    name: str = "feature_agglomeration",
) -> str:
    """
    Converts a :class:`sklearn.cluster.FeatureAgglomeration` into ONNX.

    The converter replicates
    :meth:`~sklearn.cluster.FeatureAgglomeration.transform`, which pools
    features that belong to the same cluster using ``pooling_func``
    (default: ``numpy.mean``).  The fitted ``labels_`` attribute assigns
    each input feature to a cluster index in ``[0, n_clusters)``.

    Supported pooling functions:

    * ``numpy.mean`` — implemented as a single ``MatMul`` with a precomputed
      weight matrix ``W`` of shape ``(n_features, n_clusters)`` where
      ``W[i, c] = 1 / count_c`` when ``labels_[i] == c`` and ``0``
      otherwise.  This replicates the fast ``bincount``-based path in
      scikit-learn.
    * ``numpy.max`` — implemented as per-cluster ``Gather`` +
      ``ReduceMax(axis=1)`` followed by a ``Concat``.
    * ``numpy.min`` — implemented as per-cluster ``Gather`` +
      ``ReduceMin(axis=1)`` followed by a ``Concat``.

    .. code-block:: text

        **numpy.mean path**

        X (N, F)
          │
          └──MatMul(W)──► transform_output (N, n_clusters)

        where W (F, n_clusters): W[i, c] = 1/count_c if labels_[i]==c else 0

        **numpy.max / numpy.min path**

        X (N, F)
          │
          ├──Gather(cols_0, axis=1)──ReduceMax/Min(axis=1)──► cluster_0 (N,1)
          ├──Gather(cols_1, axis=1)──ReduceMax/Min(axis=1)──► cluster_1 (N,1)
          │  …
          └──Concat(axis=1)─────────────────────────────────► transform_output (N, C)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``FeatureAgglomeration``
    :param outputs: desired output names; ``outputs[0]`` receives the
        transformed feature matrix
    :param X: input tensor name
    :param name: prefix for added node names
    :return: name of the output tensor of shape ``(N, n_clusters)``
    """
    assert isinstance(
        estimator, FeatureAgglomeration
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    labels = estimator.labels_  # shape (n_features,)
    n_clusters = estimator.n_clusters_
    pooling_func = estimator.pooling_func

    # ------------------------------------------------------------------ mean
    if pooling_func is np.mean:
        # Build weight matrix W of shape (n_features, n_clusters) where
        # W[i, c] = 1 / count_c when labels_[i] == c, else 0.
        n_features = len(labels)
        W = np.zeros((n_features, n_clusters), dtype=dtype)
        for c in range(n_clusters):
            col_mask = labels == c
            count_c = col_mask.sum()
            if count_c > 0:
                W[col_mask, c] = dtype.type(1.0 / count_c)

        out = g.op.MatMul(X, W, name=f"{name}_matmul", outputs=outputs[:1])

    # ------------------------------------------------------------------ max / min
    elif pooling_func is np.max or pooling_func is np.min:
        reduce_op = "ReduceMax" if pooling_func is np.max else "ReduceMin"
        cluster_tensors = []
        axis_arr = np.array([1], dtype=np.int64)
        for c in np.unique(labels):
            col_indices = np.where(labels == c)[0].astype(np.int64)
            gathered = g.op.Gather(
                X, col_indices, axis=1, name=f"{name}_gather_{c}"
            )  # (N, count_c)
            reduced = getattr(g.op, reduce_op)(
                gathered, axis_arr, keepdims=1, name=f"{name}_{reduce_op.lower()}_{c}"
            )  # (N, 1)
            cluster_tensors.append(reduced)

        out = g.op.Concat(
            *cluster_tensors, axis=1, name=f"{name}_concat", outputs=outputs[:1]
        )

    else:
        raise NotImplementedError(
            f"pooling_func={pooling_func!r} is not supported. "
            "Supported functions are numpy.mean, numpy.max, and numpy.min."
        )

    assert isinstance(out, str)
    if not sts:
        g.set_type(out, itype)
    return out
