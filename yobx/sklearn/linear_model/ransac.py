from typing import Dict, List

from sklearn.linear_model import RANSACRegressor

from ..register import get_sklearn_converter, register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol


@register_sklearn_converter(RANSACRegressor)
def sklearn_ransac_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: RANSACRegressor,
    X: str,
    name: str = "ransac_regressor",
) -> str:
    """
    Converts a :class:`sklearn.linear_model.RANSACRegressor` into ONNX.

    After fitting, :class:`~sklearn.linear_model.RANSACRegressor` stores
    the best-fitting sub-estimator in ``estimator_``.  Prediction is a
    direct delegation::

        y = estimator_.predict(X)

    This converter therefore simply looks up and calls the ONNX converter
    for the fitted sub-estimator (``estimator.estimator_``).

    Graph structure:

    .. code-block:: text

        X ──[sub-estimator converter]──► predictions

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names (one entry: predictions)
    :param estimator: a fitted :class:`~sklearn.linear_model.RANSACRegressor`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: name of the predictions output tensor
    """
    assert isinstance(
        estimator, RANSACRegressor
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    sub_est = estimator.estimator_
    fct = get_sklearn_converter(type(sub_est))
    result = fct(g, sts, outputs, sub_est, X, name=f"{name}__base")
    assert isinstance(result, str)
    return result
