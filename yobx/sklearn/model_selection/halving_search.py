from typing import Dict, List, Tuple, Union

from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

from ...typing import GraphBuilderExtendedProtocol
from ..register import get_sklearn_converter, register_sklearn_converter


@register_sklearn_converter(HalvingGridSearchCV)
def sklearn_halving_grid_search_cv(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: HalvingGridSearchCV,
    X: str,
    name: str = "halving_grid_search_cv",
) -> Union[str, Tuple[str, ...]]:
    """
    Converts a :class:`sklearn.model_selection.HalvingGridSearchCV` into ONNX by
    delegating to the converter registered for ``best_estimator_``.

    After fitting, :class:`~sklearn.model_selection.HalvingGridSearchCV` exposes the
    best model found during successive-halving grid search via its
    :attr:`~sklearn.model_selection.HalvingGridSearchCV.best_estimator_` attribute.
    The ONNX graph produced here is therefore **identical** to converting that
    best estimator directly.

    .. code-block:: text

        X ──[best_estimator_ converter]──► output(s)

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names for the result
    :param estimator: a fitted :class:`~sklearn.model_selection.HalvingGridSearchCV`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: name of the output tensor, or a tuple of output tensor names
    :raises AttributeError: if ``estimator`` has not been fitted yet (i.e.
        ``best_estimator_`` does not exist)
    """
    assert isinstance(
        estimator, HalvingGridSearchCV
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    best = estimator.best_estimator_
    fct = get_sklearn_converter(type(best))
    result = fct(g, sts, outputs, best, X, name=name)
    return result


@register_sklearn_converter(HalvingRandomSearchCV)
def sklearn_halving_random_search_cv(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: HalvingRandomSearchCV,
    X: str,
    name: str = "halving_random_search_cv",
) -> Union[str, Tuple[str, ...]]:
    """
    Converts a :class:`sklearn.model_selection.HalvingRandomSearchCV` into ONNX by
    delegating to the converter registered for ``best_estimator_``.

    After fitting, :class:`~sklearn.model_selection.HalvingRandomSearchCV` exposes the
    best model found during successive-halving randomized search via its
    :attr:`~sklearn.model_selection.HalvingRandomSearchCV.best_estimator_` attribute.
    The ONNX graph produced here is therefore **identical** to converting that
    best estimator directly.

    .. code-block:: text

        X ──[best_estimator_ converter]──► output(s)

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names for the result
    :param estimator: a fitted :class:`~sklearn.model_selection.HalvingRandomSearchCV`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: name of the output tensor, or a tuple of output tensor names
    :raises AttributeError: if ``estimator`` has not been fitted yet (i.e.
        ``best_estimator_`` does not exist)
    """
    assert isinstance(
        estimator, HalvingRandomSearchCV
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    best = estimator.best_estimator_
    fct = get_sklearn_converter(type(best))
    result = fct(g, sts, outputs, best, X, name=name)
    return result
