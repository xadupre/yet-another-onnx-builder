import numpy as np
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

    The transformation is decomposed into two mandatory steps mirroring
    :func:`sklearn_standard_scaler`:

    .. code-block:: text

        X  ──Sub(data_min_)──►  shifted  ──Div(data_range_)──►  normalized

    followed by an optional rescaling to the requested ``feature_range``:

    .. code-block:: text

        normalized  ──Mul(feature_width)──►  scaled  ──Add(feature_min)──►  output
                     (if feature_range != (0, 1))

    When ``feature_range`` is the default ``(0, 1)`` the last two nodes are
    replaced by a single ``Identity``.

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

    data_min = estimator.data_min_.astype(dtype)
    data_range = estimator.data_range_.astype(dtype)

    # Shift: subtract per-feature minimum (analogous to Sub(mean_) in StandardScaler).
    shifted = g.op.Sub(X, data_min, name=name)

    # Normalize to [0, 1] by dividing by the per-feature range (analogous to Div(scale_)).
    normalized = g.op.Div(shifted, data_range, name=name)

    # Rescale to the requested feature_range when it differs from the default (0, 1).
    feature_min, feature_max = estimator.feature_range
    if feature_min == 0.0 and feature_max == 1.0:
        res = g.op.Identity(normalized, name=name, outputs=outputs)
    else:
        feature_width = np.array([feature_max - feature_min], dtype=dtype)
        feature_offset = np.array([feature_min], dtype=dtype)
        scaled = g.op.Mul(normalized, feature_width, name=name)
        res = g.op.Add(scaled, feature_offset, name=name, outputs=outputs)

    assert isinstance(res, str)  # type happiness
    if not sts:
        g.set_type_shape_unary_op(res, X)
    return res
