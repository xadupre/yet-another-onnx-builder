from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from ...typing import GraphBuilderExtendedProtocol
from ..register import register_sklearn_converter


@register_sklearn_converter(TfidfTransformer)
def sklearn_tfidf_transformer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: TfidfTransformer,
    X: str,
    name: str = "tfidf_transformer",
) -> str:
    """
    Converts a :class:`sklearn.feature_extraction.text.TfidfTransformer`
    into ONNX.

    The transformer applies the following steps in order:

    1. **Term-frequency scaling** (``sublinear_tf``): if ``True``, replace
       each non-zero count with ``1 + log(count)``; zero counts stay zero.

    2. **IDF weighting** (``use_idf``): if ``True``, multiply each
       term-frequency value element-wise by the fitted ``idf_`` vector.

    3. **Row normalisation** (``norm``): scale each row to unit ``'l2'``
       or ``'l1'`` norm; ``None`` skips this step.

    **Graph layout (all three options active)**:

    .. code-block:: text

        X  ──Greater(0)──────────────────────────────────┐
           ──Log ──────── Add(1) ──── Where(>0, ·, 0) ───┤
                                                          Mul(idf_) ──ReduceL2──Div── output

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``TfidfTransformer``
    :param outputs: desired output names
    :param X: input tensor name (shape ``(N, F)``, dtype float32 or float64)
    :param name: prefix for added node names
    :return: output tensor name
    """
    assert isinstance(
        estimator, TfidfTransformer
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    tf: str = X

    # Step 1 – sublinear term-frequency scaling: tf = 1 + log(tf) where tf > 0
    if estimator.sublinear_tf:
        zero = np.array(0, dtype=dtype)
        one = np.array(1, dtype=dtype)
        gt_zero = g.op.Greater(tf, zero, name=f"{name}_gt_zero")
        log_tf = g.op.Log(tf, name=f"{name}_log")
        log1p_tf = g.op.Add(log_tf, one, name=f"{name}_log1p")
        tf = g.op.Where(gt_zero, log1p_tf, zero, name=f"{name}_sublinear_tf")

    # Step 2 – IDF weighting: tf = tf * idf_
    if estimator.use_idf:
        idf = estimator.idf_.astype(dtype)
        tf = g.op.Mul(tf, idf, name=f"{name}_idf_mul")

    # Step 3 – row normalisation
    norm: Optional[str] = estimator.norm
    axes = np.array([1], dtype=np.int64)

    if norm in ("l2", "l1"):
        if norm == "l2":
            norms = g.op.ReduceL2(tf, axes, keepdims=1, name=f"{name}_l2norm")
        else:
            norms = g.op.ReduceL1(tf, axes, keepdims=1, name=f"{name}_l1norm")
        zero_n = np.array([0], dtype=dtype)
        one_n = np.array([1], dtype=dtype)
        is_zero = g.op.Equal(norms, zero_n, name=f"{name}_is_zero")
        safe_norms = g.op.Where(is_zero, one_n, norms, name=f"{name}_safe_norm")
        res = g.op.Div(tf, safe_norms, name=name, outputs=outputs)
    elif norm is None:
        # No normalisation – pass through
        res = g.op.Identity(tf, name=name, outputs=outputs)
    else:
        raise ValueError(
            f"Unknown norm={norm!r} for TfidfTransformer, expected 'l1', 'l2', or None."
        )

    assert isinstance(res, str)
    if not sts:
        g.set_type_shape_unary_op(res, X)
    return res
