from typing import Dict, List, Union

import numpy as np
from sklearn.feature_selection import RFE, RFECV

from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter


def _sklearn_rfe_transform(
    g: GraphBuilderExtendedProtocol,
    outputs: List[str],
    estimator: Union[RFE, RFECV],
    X: str,
    name: str,
) -> str:
    """Shared implementation: select fitted feature indices via Gather."""
    indices = estimator.get_support(indices=True).astype(np.int64)
    res = g.op.Gather(X, indices, axis=1, name=name, outputs=outputs)
    return res


@register_sklearn_converter(RFE)
def sklearn_rfe(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: RFE,
    X: str,
    name: str = "rfe",
) -> str:
    """
    Converts a :class:`sklearn.feature_selection.RFE` into ONNX.

    After fitting, the transformer keeps the features selected by recursive
    feature elimination.  The selected column indices are stored in
    ``estimator.get_support(indices=True)`` and are fixed constants at
    conversion time, so the ONNX graph contains a single ``Gather`` node
    that selects those columns from the input matrix:

    .. code-block:: text

        X  ──Gather(axis=1, indices)──►  output

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``RFE``
    :param outputs: desired output names
    :param X: input name (shape ``(N, F)``)
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(estimator, RFE), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"
    return _sklearn_rfe_transform(g, outputs, estimator, X, name)


@register_sklearn_converter(RFECV)
def sklearn_rfecv(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: RFECV,
    X: str,
    name: str = "rfecv",
) -> str:
    """
    Converts a :class:`sklearn.feature_selection.RFECV` into ONNX.

    After fitting, the transformer keeps the features selected by recursive
    feature elimination with cross-validation.  The selected column indices
    are stored in ``estimator.get_support(indices=True)`` and are fixed
    constants at conversion time, so the ONNX graph contains a single
    ``Gather`` node that selects those columns from the input matrix:

    .. code-block:: text

        X  ──Gather(axis=1, indices)──►  output

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``RFECV``
    :param outputs: desired output names
    :param X: input name (shape ``(N, F)``)
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(estimator, RFECV), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"
    return _sklearn_rfe_transform(g, outputs, estimator, X, name)
