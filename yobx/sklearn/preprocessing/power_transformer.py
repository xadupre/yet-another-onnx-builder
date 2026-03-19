import numpy as np
from typing import Dict, List
from sklearn.preprocessing import PowerTransformer
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


def _yeo_johnson_transform(
    g: GraphBuilderExtendedProtocol, X: str, lambdas: np.ndarray, name: str, dtype
) -> str:
    """
    Applies the Yeo-Johnson transformation column-wise using ONNX ops.

    For each feature *j* with fitted lambda ``lam = lambdas[j]``:

    * y >= 0 and lam != 0:  ``((y + 1)^lam - 1) / lam``
    * y >= 0 and lam == 0:  ``log(y + 1)``
    * y < 0  and lam != 2:  ``-((-y + 1)^(2-lam) - 1) / (2-lam)``
    * y < 0  and lam == 2:  ``-log(-y + 1)``

    Division-by-zero is avoided by using *safe* lambda arrays (replacing
    degenerate values with 1) together with ``Where`` nodes that select the
    log branch for those features.
    """
    ones = np.ones_like(lambdas, dtype=dtype)
    zeros = np.zeros_like(lambdas, dtype=dtype)

    # Safe lambdas avoid division-by-zero; Where selects the log branch for
    # the degenerate cases, so the power-branch result is never used there.
    lam_safe_pos = np.where(lambdas == 0, ones, lambdas)
    lam_safe_neg = np.where(lambdas == 2, ones, 2.0 - lambdas)

    lam_is_zero = lambdas == 0  # bool array: selects log branch for positive side
    lam_is_two = lambdas == 2  # bool array: selects log branch for negative side

    # ── Positive branch  (y >= 0) ────────────────────────────────────────────
    x_p1 = g.op.Add(X, ones, name=f"{name}_xp1")
    pos_pow = g.op.Pow(x_p1, lam_safe_pos, name=f"{name}_pos_pow")
    pos_power_result = g.op.Div(
        g.op.Sub(pos_pow, ones, name=f"{name}_pos_sub"), lam_safe_pos, name=f"{name}_pos_div"
    )
    pos_log = g.op.Log(x_p1, name=f"{name}_pos_log")
    pos_transform = g.op.Where(lam_is_zero, pos_log, pos_power_result, name=f"{name}_pos_where")

    # ── Negative branch  (y < 0) ─────────────────────────────────────────────
    neg_x = g.op.Neg(X, name=f"{name}_negx")
    neg_x_p1 = g.op.Add(neg_x, ones, name=f"{name}_nxp1")
    neg_pow = g.op.Pow(neg_x_p1, lam_safe_neg, name=f"{name}_neg_pow")
    neg_power_result = g.op.Neg(
        g.op.Div(
            g.op.Sub(neg_pow, ones, name=f"{name}_neg_sub"), lam_safe_neg, name=f"{name}_neg_div"
        ),
        name=f"{name}_neg_neg",
    )
    neg_log = g.op.Neg(g.op.Log(neg_x_p1, name=f"{name}_neg_log"), name=f"{name}_neg_log_neg")
    neg_transform = g.op.Where(lam_is_two, neg_log, neg_power_result, name=f"{name}_neg_where")

    # ── Select branch based on sign of X ────────────────────────────────────
    mask_pos = g.op.GreaterOrEqual(X, zeros, name=f"{name}_mask")
    return g.op.Where(mask_pos, pos_transform, neg_transform, name=f"{name}_combine")


def _box_cox_transform(
    g: GraphBuilderExtendedProtocol, X: str, lambdas: np.ndarray, name: str, dtype
) -> str:
    """
    Applies the Box-Cox transformation column-wise using ONNX ops.

    For each feature *j* with fitted lambda ``lam = lambdas[j]``:

    * lam != 0:  ``(x^lam - 1) / lam``
    * lam == 0:  ``log(x)``

    The input *X* is assumed to be strictly positive (Box-Cox requirement).
    """
    ones = np.ones_like(lambdas, dtype=dtype)

    lam_safe = np.where(lambdas == 0, ones, lambdas)
    # bool array passed to Where; the graph builder converts it to an ONNX constant
    lam_is_zero = lambdas == 0

    pow_result = g.op.Pow(X, lam_safe, name=f"{name}_pow")
    power_result = g.op.Div(
        g.op.Sub(pow_result, ones, name=f"{name}_sub"), lam_safe, name=f"{name}_div"
    )
    log_result = g.op.Log(X, name=f"{name}_log")
    return g.op.Where(lam_is_zero, log_result, power_result, name=f"{name}_where")


@register_sklearn_converter(PowerTransformer)
def sklearn_power_transformer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: PowerTransformer,
    X: str,
    name: str = "power_transformer",
) -> str:
    """
    Converts a :class:`sklearn.preprocessing.PowerTransformer` into ONNX.

    Both ``method='yeo-johnson'`` (default) and ``method='box-cox'`` are
    supported.  When ``standardize=True`` (the default) the transformer also
    applies a :class:`~sklearn.preprocessing.StandardScaler` to the output;
    this is inlined as ``Sub`` / ``Div`` nodes.

    **Yeo-Johnson** — applied per column:

    .. code-block:: text

        y >= 0, lam != 0 :  ((y + 1)^lam  - 1) / lam
        y >= 0, lam == 0 :  log(y + 1)
        y < 0,  lam != 2 :  -((-y + 1)^(2-lam) - 1) / (2-lam)
        y < 0,  lam == 2 :  -log(-y + 1)

    **Box-Cox** — applied per column (input must be positive):

    .. code-block:: text

        lam != 0 :  (x^lam - 1) / lam
        lam == 0 :  log(x)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names
    :param estimator: a fitted :class:`~sklearn.preprocessing.PowerTransformer`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: name of the output tensor
    """
    assert isinstance(
        estimator, PowerTransformer
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    lambdas = estimator.lambdas_.astype(dtype)

    if estimator.method == "yeo-johnson":
        transformed = _yeo_johnson_transform(g, X, lambdas, name, dtype)
    elif estimator.method == "box-cox":
        transformed = _box_cox_transform(g, X, lambdas, name, dtype)
    else:
        raise NotImplementedError(
            f"PowerTransformer method {estimator.method!r} is not supported."
        )

    if estimator.standardize:
        mean = estimator._scaler.mean_.astype(dtype)
        scale = estimator._scaler.scale_.astype(dtype)
        centered = g.op.Sub(transformed, mean, name=f"{name}_std_sub")
        res = g.op.Div(centered, scale, name=f"{name}_std_div", outputs=outputs)
    else:
        res = g.op.Identity(transformed, name=name, outputs=outputs)

    g.set_type_shape_unary_op(res, X)
    return res
