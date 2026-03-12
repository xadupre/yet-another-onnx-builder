from typing import Dict, List
from sklearn.preprocessing import RobustScaler
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(RobustScaler)
def sklearn_robust_scaler(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: RobustScaler,
    X: str,
    name: str = "scaler",
) -> str:
    """
    Converts a :class:`sklearn.preprocessing.RobustScaler` into ONNX.

    The implementation respects the ``with_centering`` and ``with_scaling`` flags:

    .. code-block:: text

        X  ──Sub(center_)──►  centered  ──Div(scale_)──►  output
             (if with_centering)            (if with_scaling)

    When ``with_centering=False`` the ``Sub`` node is skipped; when
    ``with_scaling=False`` the ``Div`` node is replaced by an ``Identity``.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``RobustScaler``
    :param outputs: desired output names
    :param X: input name
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(
        estimator, RobustScaler
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # Apply centering only if requested.
    if getattr(estimator, "with_centering", True):
        assert estimator.center_ is not None  # type happiness
        center = estimator.center_.astype(dtype)
        centered = g.op.Sub(X, center, name=name)
    else:
        centered = X

    # Apply scaling only if requested.
    if getattr(estimator, "with_scaling", True):
        assert estimator.scale_ is not None  # type happiness
        scale = estimator.scale_.astype(dtype)
        res = g.op.Div(centered, scale, name=name, outputs=outputs)
    else:
        # No scaling: forward the (possibly centered) tensor to the desired outputs.
        res = g.op.Identity(centered, name=name, outputs=outputs)
    assert isinstance(res, str)  # type happiness
    if not sts:
        g.set_type_shape_unary_op(res, X)
    return res
