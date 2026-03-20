from typing import Dict, List

import numpy as np
from sklearn.covariance import EmpiricalCovariance

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


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
    for each observation:

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
    return mahal_sq
