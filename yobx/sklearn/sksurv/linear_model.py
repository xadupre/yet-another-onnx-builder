from typing import Dict, List

import numpy as np
from sksurv.linear_model import IPCRidge

from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(IPCRidge)
def sklearn_ipc_ridge(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: IPCRidge,
    X: str,
    name: str = "ipc_ridge",
) -> str:
    """
    Converts a :class:`sksurv.linear_model.IPCRidge` into ONNX.

    :class:`~sksurv.linear_model.IPCRidge` fits a Ridge regression on the
    log-transformed survival time using Inverse Probability of Censoring
    Weighting (IPCW).  At inference time the prediction formula is::

        y = exp(X @ coef_ + intercept_)

    that is, a linear combination of the features followed by an
    exponential transformation to map back to the original (non-log)
    time scale.

    Graph structure:

    .. code-block:: text

        X  ──Gemm(coef, intercept, transB=1)──Exp──►  predictions

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names
    :param estimator: a fitted :class:`~sksurv.linear_model.IPCRidge`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: output tensor name
    """
    assert isinstance(estimator, IPCRidge), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    coef = estimator.coef_.astype(dtype)
    intercept = np.atleast_1d(np.asarray(estimator.intercept_)).astype(dtype)

    # Ensure coef is 2D: (1, n_features) for Gemm with transB=1.
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)

    # Linear part: X @ coef.T + intercept  →  (N, 1)
    linear = g.op.Gemm(X, coef, intercept, transB=1, name=f"{name}_gemm")
    assert isinstance(linear, str)

    # Exponential transformation: exp(linear)  →  (N, 1)
    result = g.op.Exp(linear, name=f"{name}_exp", outputs=outputs)
    assert isinstance(result, str)
    if not sts:
        g.set_type(result, itype)
    return result
