from typing import Dict, List
from sklearn.preprocessing import MinMaxScaler
from ..register import register_sklearn_converter
from ...xbuilder import GraphBuilder


@register_sklearn_converter(MinMaxScaler)
def sklearn_min_max_scaler(
    g: GraphBuilder,
    sts: Dict,
    outputs: List[str],
    estimator: MinMaxScaler,
    X: str,
    name: str = "scaler",
) -> str:
    """
    Converts a :class:`sklearn.preprocessing.MinMaxScaler` into ONNX.

    The transformation applied by a fitted ``MinMaxScaler`` is the linear map:

    .. code-block:: text

        X_scaled = X * scale_ + min_

    where ``scale_`` and ``min_`` are the per-feature coefficients stored on the
    fitted estimator.  This is emitted as a ``Mul`` followed by an ``Add`` node:

    .. code-block:: text

        X  ──Mul(scale_)──►  scaled  ──Add(min_)──►  output

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``MinMaxScaler``
    :param outputs: desired output names
    :param X: input name
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(
        estimator, MinMaxScaler
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = g.onnx_dtype_to_np_dtype(itype)

    scale = estimator.scale_.astype(dtype)
    min_ = estimator.min_.astype(dtype)

    scaled = g.op.Mul(X, scale, name=name)
    res = g.op.Add(scaled, min_, name=name, outputs=outputs)
    assert isinstance(res, str)
    if not sts:
        g.set_type_shape_unary_op(res, X)
    return res
