from typing import Dict, List

import numpy as np
from sklearn.feature_selection import SelectFwe

from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter


@register_sklearn_converter(SelectFwe)
def sklearn_select_fwe(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: SelectFwe,
    X: str,
    name: str = "select_fwe",
) -> str:
    """
    Converts a :class:`sklearn.feature_selection.SelectFwe` into ONNX.

    After fitting, the transformer keeps the features whose uncorrected
    p-value is below the family-wise error rate threshold *alpha*.  The
    selected column indices are stored in
    ``estimator.get_support(indices=True)`` and are fixed constants at
    conversion time, so the ONNX graph contains a single ``Gather`` node
    that selects those columns from the input matrix:

    .. code-block:: text

        X  ──Gather(axis=1, indices)──►  output

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``SelectFwe``
    :param outputs: desired output names
    :param X: input name (shape ``(N, F)``)
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(
        estimator, SelectFwe
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    indices = estimator.get_support(indices=True).astype(np.int64)
    res = g.op.Gather(X, indices, axis=1, name=name, outputs=outputs)

    assert isinstance(res, str)
    return res
