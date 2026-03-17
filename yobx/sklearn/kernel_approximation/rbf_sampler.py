import numpy as np
from typing import Dict, List

from sklearn.kernel_approximation import RBFSampler

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(RBFSampler)
def sklearn_rbf_sampler(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: RBFSampler,
    X: str,
    name: str = "rbf_sampler",
) -> str:
    """
    Converts a :class:`sklearn.kernel_approximation.RBFSampler` into ONNX.

    The transform replicates
    :meth:`sklearn.kernel_approximation.RBFSampler.transform` using random
    Fourier features (Random Kitchen Sinks) to approximate the RBF (Gaussian)
    kernel:

    .. code-block:: text

        X_new = sqrt(2 / n_components) * cos(X @ random_weights_ + random_offset_)

    where ``random_weights_`` has shape ``(n_features, n_components)`` and
    ``random_offset_`` has shape ``(n_components,)``.

    **Graph structure**:

    .. code-block:: text

        X ──MatMul(random_weights_)──Add(random_offset_)──Cos──Mul(scale)──► X_new

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``RBFSampler``
    :param outputs: desired output names (transformed inputs)
    :param X: input tensor name (shape ``(N, F)``)
    :param name: prefix name for the added nodes
    :return: output tensor name (shape ``(N, n_components)``)
    """
    assert isinstance(estimator, RBFSampler), (
        f"Unexpected type {type(estimator)} for estimator."
    )
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    random_weights = estimator.random_weights_.astype(dtype)  # (n_features, n_components)
    random_offset = estimator.random_offset_.astype(dtype)    # (n_components,)
    n_components = int(random_weights.shape[1])

    scale = np.array([np.sqrt(2.0 / n_components)], dtype=dtype)

    # X @ random_weights_   →  (N, n_components)
    projection = g.op.MatMul(X, random_weights, name=f"{name}_proj")

    # + random_offset_
    shifted = g.op.Add(projection, random_offset, name=f"{name}_shift")

    # cos(...)
    cosined = g.op.Cos(shifted, name=f"{name}_cos")

    # * sqrt(2 / n_components)
    res = g.op.Mul(cosined, scale, name=name, outputs=outputs)

    assert isinstance(res, str)  # type happiness
    if not sts:
        g.set_type(res, itype)
        if g.has_shape(X):
            batch_dim = g.get_shape(X)[0]
            g.set_shape(res, (batch_dim, n_components))
    return res
