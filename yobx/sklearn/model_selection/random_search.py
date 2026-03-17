from typing import Dict, List, Tuple, Union

from sklearn.model_selection import RandomizedSearchCV

from ...typing import GraphBuilderExtendedProtocol
from ..register import get_sklearn_converter, register_sklearn_converter


@register_sklearn_converter(RandomizedSearchCV)
def sklearn_randomized_search_cv(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: RandomizedSearchCV,
    X: str,
    name: str = "randomized_search_cv",
) -> Union[str, Tuple[str, ...]]:
    """
    Converts a :class:`sklearn.model_selection.RandomizedSearchCV` into ONNX by
    delegating to the converter registered for ``best_estimator_``.

    After fitting, :class:`~sklearn.model_selection.RandomizedSearchCV` exposes the
    best model found during the randomized cross-validated search via its
    :attr:`~sklearn.model_selection.RandomizedSearchCV.best_estimator_` attribute.
    The ONNX graph produced here is therefore **identical** to converting that
    best estimator directly.

    .. code-block:: text

        X ──[best_estimator_ converter]──► output(s)

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names for the result
    :param estimator: a fitted :class:`~sklearn.model_selection.RandomizedSearchCV`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: name of the output tensor, or a tuple of output tensor names
    :raises AttributeError: if ``estimator`` has not been fitted yet (i.e.
        ``best_estimator_`` does not exist)
    """
    assert isinstance(
        estimator, RandomizedSearchCV
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    best = estimator.best_estimator_
    fct = get_sklearn_converter(type(best))
    result = fct(g, sts, outputs, best, X, name=name)
    return result
