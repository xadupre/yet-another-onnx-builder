from typing import Dict, List
from sklearn.preprocessing import MaxAbsScaler
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(MaxAbsScaler)
def sklearn_max_abs_scaler(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: MaxAbsScaler,
    X: str,
    name: str = "scaler",
) -> str:
    """
    Converts a :class:`sklearn.preprocessing.MaxAbsScaler` into ONNX.

    The transformation divides each feature by its maximum absolute value:

    .. code-block:: text

        X  ──Div(max_abs_)──►  output

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``MaxAbsScaler``
    :param outputs: desired output names
    :param X: input name
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(
        estimator, MaxAbsScaler
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    max_abs = estimator.max_abs_.astype(dtype)

    res = g.op.Div(X, max_abs, name=name, outputs=outputs)
    assert isinstance(res, str)  # type happiness
    g.set_type_shape_unary_op(res, X)
    return res
