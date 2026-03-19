from typing import Dict, List, Union
import numpy as np
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor
from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype

_GLM_TYPES = (TweedieRegressor, PoissonRegressor, GammaRegressor)


@register_sklearn_converter(_GLM_TYPES)
def sklearn_glm_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: Union[TweedieRegressor, PoissonRegressor, GammaRegressor],
    X: str,
    name: str = "glm_regressor",
) -> str:
    """
    Converts a sklearn GLM regressor (:class:`~sklearn.linear_model.TweedieRegressor`,
    :class:`~sklearn.linear_model.PoissonRegressor`,
    :class:`~sklearn.linear_model.GammaRegressor`) into ONNX.

    GLMs apply an inverse link function to the linear predictor:

    .. code-block:: text

        X  ──Gemm(coef, intercept, transB=1)──►  linear_pred
                                                        │
                                          inverse_link(·) ──►  predictions

    Supported link functions:

    * **IdentityLink**: pass-through (no extra node)
    * **LogLink**: ``Exp`` applied to the linear predictor

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a fitted GLM regressor
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name
    :raises NotImplementedError: when an unsupported link function is encountered
    """
    assert isinstance(estimator, _GLM_TYPES), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    coef = estimator.coef_.astype(dtype)
    intercept = np.atleast_1d(np.asarray(estimator.intercept_)).astype(dtype)

    # Ensure coef is 2D for Gemm.
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)

    linear_pred = g.op.Gemm(X, coef, intercept, transB=1, name=f"{name}_linear")

    # Determine the inverse link function from the fitted model.
    link_cls = type(estimator._base_loss.link).__name__
    if "Identity" in link_cls:
        result = g.op.Identity(linear_pred, name=name, outputs=outputs)
    elif "Log" in link_cls:
        result = g.op.Exp(linear_pred, name=name, outputs=outputs)
    else:
        raise NotImplementedError(
            f"Unsupported link function {link_cls!r} for {type(estimator).__name__}. "
            "Only IdentityLink and LogLink are supported."
        )

    assert isinstance(result, str)
    g.set_type(result, itype)
    return result
