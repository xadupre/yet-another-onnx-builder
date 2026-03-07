from typing import Dict, List
from sklearn.preprocessing import StandardScaler
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol


@register_sklearn_converter(StandardScaler)
def sklearn_standard_scaler(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: StandardScaler,
    X: str,
    name: str = "scaler",
) -> str:
    """
    Converts a :class:`sklearn.preprocessing.StandardScaler` into ONNX.

    The implementation respects the ``with_mean`` and ``with_std`` flags:

    .. code-block:: text

        X  ──Sub(mean)──►  centered  ──Div(scale)──►  output
             (if with_mean)               (if with_std)

    When ``with_mean=False`` the ``Sub`` node is skipped; when
    ``with_std=False`` the ``Div`` node is replaced by an ``Identity``.

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

    # Apply centering only if requested.
    if getattr(estimator, "with_mean", True):
        assert estimator.mean_ is not None  # type happiness
        mean = estimator.mean_.astype(dtype)
        centered = g.op.Sub(X, mean, name=name)
    else:
        centered = X

    # Apply scaling only if requested.
    if getattr(estimator, "with_std", True):
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
