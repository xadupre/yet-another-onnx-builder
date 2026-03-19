from typing import Dict, List

from sklearn.cross_decomposition import CCA

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(CCA)
def sklearn_cca(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: CCA,
    X: str,
    name: str = "cca",
) -> str:
    """
    Converts a :class:`sklearn.cross_decomposition.CCA` into ONNX.

    The conversion mirrors :meth:`CCA.transform` with a single input ``X``:

    .. code-block:: text

        X ──Sub(_x_mean)──► centered ── Div(_x_std)──► scaled ──MatMul(x_rotations_) ──► output

    The input is centred by subtracting ``_x_mean``, normalized by dividing
    by ``_x_std``, then projected using ``x_rotations_`` to obtain the
    X canonical variates.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``CCA``
    :param outputs: desired output names
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output tensor name
    """
    assert isinstance(estimator, CCA), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # Center X
    x_mean = estimator._x_mean.astype(dtype)
    centered = g.op.Sub(X, x_mean, name=name)

    # Scale X
    x_std = estimator._x_std.astype(dtype)
    scaled = g.op.Div(centered, x_std, name=name)

    # Apply rotation: x_scores = scaled @ x_rotations_
    x_rotations = estimator.x_rotations_.astype(dtype)
    res = g.op.MatMul(scaled, x_rotations, name=name, outputs=outputs)

    g.set_type(res, itype)
    if g.has_shape(X):
        batch_dim = g.get_shape(X)[0]
        n_components = estimator.x_rotations_.shape[1]
        g.set_shape(res, (batch_dim, n_components))
    elif g.has_rank(X):
        g.set_rank(res, 2)
    return res
