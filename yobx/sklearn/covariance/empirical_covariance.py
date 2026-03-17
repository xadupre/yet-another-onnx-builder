from typing import Dict, List

import numpy as np
from sklearn.covariance import (
    EmpiricalCovariance,
    GraphicalLasso,
    GraphicalLassoCV,
    MinCovDet,
    OAS,
    ShrunkCovariance,
)

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype

_EMPIRICAL_COVARIANCE_TYPES = (
    EmpiricalCovariance,
    GraphicalLasso,
    GraphicalLassoCV,
    MinCovDet,
    OAS,
    ShrunkCovariance,
)


def _empirical_covariance_mahalanobis(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: EmpiricalCovariance,
    X: str,
    name: str,
) -> str:
    """
    Shared ONNX implementation of
    :meth:`~sklearn.covariance.EmpiricalCovariance.mahalanobis`.

    All :class:`~sklearn.covariance.EmpiricalCovariance` subclasses share the
    same ``mahalanobis`` computation:

    .. math::

        d^2(x) = (x - \\mu)^\\top \\, \\Sigma^{-1} \\, (x - \\mu)
                = \\sum_j \\bigl[(x - \\mu) \\Sigma^{-1}\\bigr]_j \\cdot (x - \\mu)_j

    where :math:`\\mu` is :attr:`location_` and :math:`\\Sigma^{-1}` is
    :attr:`precision_`.

    Full graph structure:

    .. code-block:: text

        X (N, F)
          │
          └─ Sub(location_) ──► X_centered (N, F)
               │
               ├─ MatMul(precision_) ──► X_prec (N, F)
               │                              │
               └──────────────────── Mul ──► X_prec_centered (N, F)
                                              │
                                   ReduceSum(axis=1) ──► mahal_sq (N,)  [output]

    :param g: graph builder
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a fitted covariance estimator
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name
    """
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    location = estimator.location_.astype(dtype)  # (F,)
    precision = estimator.precision_.astype(dtype)  # (F, F)

    # Center the input: X_centered = X - location_
    x_centered = g.op.Sub(X, location, name=f"{name}_center")  # (N, F)

    # Mahalanobis: X_centered @ precision_  →  (N, F)
    x_prec = g.op.MatMul(x_centered, precision, name=f"{name}_matmul")  # (N, F)

    # Element-wise product, then sum over features → squared Mahalanobis dist
    x_prec_centered = g.op.Mul(x_prec, x_centered, name=f"{name}_ew_mul")  # (N, F)
    mahal_sq = g.op.ReduceSum(
        x_prec_centered,
        np.array([1], dtype=np.int64),
        keepdims=0,
        name=f"{name}_mahal_sq",
        outputs=outputs,
    )  # (N,)
    assert isinstance(mahal_sq, str)
    return mahal_sq


@register_sklearn_converter(EmpiricalCovariance)
def sklearn_empirical_covariance(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: EmpiricalCovariance,
    X: str,
    name: str = "empirical_covariance",
) -> str:
    """
    Converts a :class:`sklearn.covariance.EmpiricalCovariance` into ONNX.

    The converter maps to :meth:`~sklearn.covariance.EmpiricalCovariance.mahalanobis`,
    returning the **squared** Mahalanobis distance from the fitted distribution
    for each observation.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names; ``outputs[0]`` receives the squared
        Mahalanobis distances
    :param estimator: a fitted ``EmpiricalCovariance``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name
    """
    assert isinstance(
        estimator, EmpiricalCovariance
    ), f"Unexpected type {type(estimator)} for estimator."
    return _empirical_covariance_mahalanobis(g, sts, outputs, estimator, X, name)


@register_sklearn_converter(GraphicalLasso)
def sklearn_graphical_lasso(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: GraphicalLasso,
    X: str,
    name: str = "graphical_lasso",
) -> str:
    """
    Converts a :class:`sklearn.covariance.GraphicalLasso` into ONNX.

    The converter maps to :meth:`~sklearn.covariance.GraphicalLasso.mahalanobis`,
    returning the **squared** Mahalanobis distance from the fitted distribution
    for each observation.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names; ``outputs[0]`` receives the squared
        Mahalanobis distances
    :param estimator: a fitted ``GraphicalLasso``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name
    """
    assert isinstance(
        estimator, GraphicalLasso
    ), f"Unexpected type {type(estimator)} for estimator."
    return _empirical_covariance_mahalanobis(g, sts, outputs, estimator, X, name)


@register_sklearn_converter(GraphicalLassoCV)
def sklearn_graphical_lasso_cv(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: GraphicalLassoCV,
    X: str,
    name: str = "graphical_lasso_cv",
) -> str:
    """
    Converts a :class:`sklearn.covariance.GraphicalLassoCV` into ONNX.

    The converter maps to :meth:`~sklearn.covariance.GraphicalLassoCV.mahalanobis`,
    returning the **squared** Mahalanobis distance from the fitted distribution
    for each observation.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names; ``outputs[0]`` receives the squared
        Mahalanobis distances
    :param estimator: a fitted ``GraphicalLassoCV``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name
    """
    assert isinstance(
        estimator, GraphicalLassoCV
    ), f"Unexpected type {type(estimator)} for estimator."
    return _empirical_covariance_mahalanobis(g, sts, outputs, estimator, X, name)


@register_sklearn_converter(MinCovDet)
def sklearn_min_cov_det(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: MinCovDet,
    X: str,
    name: str = "min_cov_det",
) -> str:
    """
    Converts a :class:`sklearn.covariance.MinCovDet` into ONNX.

    The converter maps to :meth:`~sklearn.covariance.MinCovDet.mahalanobis`,
    returning the **squared** Mahalanobis distance from the fitted robust
    distribution for each observation.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names; ``outputs[0]`` receives the squared
        Mahalanobis distances
    :param estimator: a fitted ``MinCovDet``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name
    """
    assert isinstance(
        estimator, MinCovDet
    ), f"Unexpected type {type(estimator)} for estimator."
    return _empirical_covariance_mahalanobis(g, sts, outputs, estimator, X, name)


@register_sklearn_converter(OAS)
def sklearn_oas(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: OAS,
    X: str,
    name: str = "oas",
) -> str:
    """
    Converts a :class:`sklearn.covariance.OAS` into ONNX.

    The converter maps to :meth:`~sklearn.covariance.OAS.mahalanobis`,
    returning the **squared** Mahalanobis distance from the fitted distribution
    for each observation.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names; ``outputs[0]`` receives the squared
        Mahalanobis distances
    :param estimator: a fitted ``OAS``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name
    """
    assert isinstance(estimator, OAS), f"Unexpected type {type(estimator)} for estimator."
    return _empirical_covariance_mahalanobis(g, sts, outputs, estimator, X, name)


@register_sklearn_converter(ShrunkCovariance)
def sklearn_shrunk_covariance(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: ShrunkCovariance,
    X: str,
    name: str = "shrunk_covariance",
) -> str:
    """
    Converts a :class:`sklearn.covariance.ShrunkCovariance` into ONNX.

    The converter maps to :meth:`~sklearn.covariance.ShrunkCovariance.mahalanobis`,
    returning the **squared** Mahalanobis distance from the fitted distribution
    for each observation.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names; ``outputs[0]`` receives the squared
        Mahalanobis distances
    :param estimator: a fitted ``ShrunkCovariance``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name
    """
    assert isinstance(
        estimator, ShrunkCovariance
    ), f"Unexpected type {type(estimator)} for estimator."
    return _empirical_covariance_mahalanobis(g, sts, outputs, estimator, X, name)
