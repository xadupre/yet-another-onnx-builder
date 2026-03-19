from typing import Dict, List

import numpy as np
from sklearn.feature_selection import SelectFromModel

from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter


@register_sklearn_converter(SelectFromModel)
def sklearn_select_from_model(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: SelectFromModel,
    X: str,
    name: str = "select_from_model",
) -> str:
    """
    Converts a :class:`sklearn.feature_selection.SelectFromModel` into ONNX.

    After fitting, the meta-transformer selects features whose importance
    (as reported by the wrapped estimator's ``feature_importances_`` or
    ``coef_`` attribute) exceeds the configured threshold.  The selected
    column indices are fixed at conversion time via
    ``estimator.get_support(indices=True)``, so the ONNX graph contains a
    single ``Gather`` node that selects those columns from the input matrix:

    .. code-block:: text

        X  ──Gather(axis=1, indices)──►  output

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``SelectFromModel``
    :param outputs: desired output names
    :param X: input name (shape ``(N, F)``)
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(
        estimator, SelectFromModel
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    indices = estimator.get_support(indices=True).astype(np.int64)
    res = g.op.Gather(X, indices, axis=1, name=name, outputs=outputs)

    return res
