from typing import Dict, List

import numpy as np
from sklearn.feature_selection import SelectKBest

from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter


@register_sklearn_converter(SelectKBest)
def sklearn_select_k_best(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: SelectKBest,
    X: str,
    name: str = "select_k_best",
) -> str:
    """
    Converts a :class:`sklearn.feature_selection.SelectKBest` into ONNX.

    After fitting, the transformer keeps the *k* features with the highest
    scores according to the chosen scoring function.  The selected column
    indices are stored in ``estimator.get_support(indices=True)`` and are
    fixed constants at conversion time, so the ONNX graph contains a single
    ``Gather`` node that selects those columns from the input matrix:

    .. code-block:: text

        X  ──Gather(axis=1, indices)──►  output

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``SelectKBest``
    :param outputs: desired output names
    :param X: input name (shape ``(N, F)``)
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(estimator, SelectKBest), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    indices = estimator.get_support(indices=True).astype(np.int64)
    res = g.op.Gather(X, indices, axis=1, name=name, outputs=outputs)

    assert isinstance(res, str)
    return res
