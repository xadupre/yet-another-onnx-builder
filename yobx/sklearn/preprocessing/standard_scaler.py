from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
from ..register import register_sklearn_converter
from ...xbuilder import GraphBuilder


@register_sklearn_converter(StandardScaler)
def sklearn_standard_scaler(
    g: GraphBuilder,
    sts: Dict,
    outputs: List[str],
    estimator: StandardScaler,
    X: str,
    name: str = "scaler",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.preprocessing.StandardScaler` into ONNX.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``StandardScaler``
    :param outputs: desired names (scaled inputs)
    :param X: inputs
    :param name: prefix name for the added nodes
    :return: output
    """
    assert isinstance(
        estimator, StandardScaler
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = g.onnx_dtype_to_np_dtype(itype)

    mean = estimator.mean_.astype(dtype)
    scale = estimator.scale_.astype(dtype)
    centered = g.op.Sub(X, mean, name=name)
    res = g.op.Div(centered, scale, name=name, outputs=outputs)
    if not sts:
        g.set_type(res, g.get_type(X))
        g.set_shape(res, g.get_shape(X))
        g.set_device(res, g.get_device(X))
    return res
