from typing import Dict, List

import numpy as np
from sklearn.feature_selection import VarianceThreshold

from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter


@register_sklearn_converter(VarianceThreshold)
def sklearn_variance_threshold(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: VarianceThreshold,
    X: str,
    name: str = "variance_threshold",
) -> str:
    """
    Converts a :class:`sklearn.feature_selection.VarianceThreshold` into ONNX.

    After fitting, the transformer keeps only the features whose variance
    exceeds *threshold*.  The selected column indices are stored in
    ``estimator.get_support(indices=True)`` and are fixed constants at
    conversion time, so the ONNX graph contains a single ``Gather`` node
    that selects those columns from the input matrix:

    .. code-block:: text

        X  ──Gather(axis=1, indices)──►  output

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``VarianceThreshold``
    :param outputs: desired output names
    :param X: input name (shape ``(N, F)``)
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(
        estimator, VarianceThreshold
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    indices = estimator.get_support(indices=True).astype(np.int64)
    res = g.op.Gather(X, indices, axis=1, name=name, outputs=outputs)

    assert isinstance(res, str)
    return res
