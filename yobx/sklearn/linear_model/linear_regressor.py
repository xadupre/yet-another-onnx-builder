from typing import Dict, List, Union
import numpy as np
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    HuberRegressor,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    LinearRegression,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
    MultiTaskLasso,
    MultiTaskLassoCV,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    QuantileRegressor,
    Ridge,
    RidgeCV,
    SGDRegressor,
    TheilSenRegressor,
)
from ..register import register_sklearn_converter
from ...typing import GraphBuilderProtocolExtended

# Optional deprecated models
_EXTRA_REGRESSOR_TYPES = []
try:
    from sklearn.linear_model import PassiveAggressiveRegressor  # deprecated in sklearn 1.8

    _EXTRA_REGRESSOR_TYPES.append(PassiveAggressiveRegressor)
except ImportError:
    pass

_LINEAR_REGRESSOR_TYPES = (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    HuberRegressor,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    LinearRegression,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
    MultiTaskLasso,
    MultiTaskLassoCV,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    QuantileRegressor,
    Ridge,
    RidgeCV,
    SGDRegressor,
    TheilSenRegressor,
    *_EXTRA_REGRESSOR_TYPES,
)


@register_sklearn_converter(_LINEAR_REGRESSOR_TYPES)
def sklearn_linear_regressor(
    g: GraphBuilderProtocolExtended,
    sts: Dict,
    outputs: List[str],
    estimator: Union[LinearRegression, Ridge],
    X: str,
    name: str = "linear_regressor",
) -> str:
    """
    Converts any plain sklearn linear regressor into ONNX.

    All supported estimators share the same prediction formula::

        y = X @ coef_.T + intercept_

    The mapping covers :class:`~sklearn.linear_model.LinearRegression`,
    :class:`~sklearn.linear_model.Ridge`, :class:`~sklearn.linear_model.Lasso`,
    :class:`~sklearn.linear_model.ElasticNet` and many more (see
    ``_LINEAR_REGRESSOR_TYPES``).

    Graph structure:

    .. code-block:: text

        X  ──Gemm(coef, intercept, transB=1)──►  predictions

    For single-output models the output shape is ``(N, 1)``; for
    multi-output models it is ``(N, n_targets)``.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a fitted linear regressor
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name
    """
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = g.onnx_dtype_to_np_dtype(itype)

    coef = estimator.coef_.astype(dtype)
    intercept = np.atleast_1d(np.asarray(estimator.intercept_)).astype(dtype)

    # Ensure coef is 2D: (n_targets, n_features) for Gemm with transB=1.
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)

    result = g.op.Gemm(X, coef, intercept, transB=1, name=name, outputs=outputs)
    assert isinstance(result, str)
    if not sts:
        g.set_type(result, itype)
    return result
