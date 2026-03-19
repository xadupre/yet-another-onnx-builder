import warnings
from typing import Dict, List
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from .. import NumericalDiscrepancyWarning, has_sklearn
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(PLSRegression)
def sklearn_pls_regression(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: PLSRegression,
    X: str,
    name: str = "pls_regression",
) -> str:
    """
    Converts a :class:`sklearn.cross_decomposition.PLSRegression` into ONNX.

    The prediction formula mirrors :meth:`PLSRegression.predict`:

    .. code-block:: text

        X  ──Sub(_x_mean)──►  centered  ──Gemm(coef_.T, intercept_)──►  y_pred
                                                                              │
                                                         (if _predict_1d) Squeeze──►  output
                                                                              │
                                                        (if not _predict_1d) └──────►  output

    The input is centred by subtracting ``_x_mean``, then the prediction is
    computed as ``y_pred = centered @ coef_.T + intercept_``.  For single-target
    models (``_predict_1d`` is ``True``) the output is squeezed to shape
    ``(N,)``; for multi-target models the output shape is ``(N, n_targets)``.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``PLSRegression``
    :param outputs: desired output names (predictions)
    :param X: input tensor name
    :param name: prefix name for the added nodes
    :return: output tensor name

    .. note:: discrepancies

        The conversion shows discrepancies for ``scikit-learn==1.4``
        at least in unit tests. It is safe to assume it only works
        for ``scikit-learn>=1.8``.
    """
    assert isinstance(
        estimator, PLSRegression
    ), f"Unexpected type {type(estimator)} for estimator."
    if not has_sklearn("1.8"):
        warnings.warn(
            "Discrepancies were observed for scikit-learn==1.4 but not for 1.8."
            "This is probably because scikit-learn>=1.8 is more consistent with computation "
            "types and does not implicitly switch to float64.",
            NumericalDiscrepancyWarning,
        )
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # Center X by subtracting _x_mean (sklearn does NOT scale X here).
    x_mean = estimator._x_mean.astype(dtype)
    centered = g.op.Sub(X, x_mean, name=name)

    # coef_ has shape (n_targets, n_features); we need (n_features, n_targets).
    coef_T = estimator.coef_.T.astype(dtype)
    intercept = estimator.intercept_.astype(dtype)

    # y_pred = centered @ coef_.T + intercept_
    y_pred = g.op.MatMul(centered, coef_T, name=name)
    y_pred = g.op.Add(y_pred, intercept, name=name)

    if estimator._predict_1d:
        # Single target: squeeze from (N, 1) to (N,)
        res = g.op.Squeeze(y_pred, np.array([1], dtype=np.int64), name=name, outputs=outputs)
    else:
        res = g.op.Identity(y_pred, name=name, outputs=outputs)

    assert isinstance(res, str)
    g.set_type(res, itype)
    if g.has_shape(X):
        batch_dim = g.get_shape(X)[0]
        if estimator._predict_1d:
            g.set_shape(res, (batch_dim,))
        else:
            n_targets = estimator.coef_.shape[0]
            g.set_shape(res, (batch_dim, n_targets))
    elif g.has_rank(X):
        if estimator._predict_1d:
            g.set_rank(res, 1)
        else:
            g.set_rank(res, 2)
    return res
