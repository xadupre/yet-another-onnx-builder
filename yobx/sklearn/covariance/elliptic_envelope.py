from typing import Dict, List, Tuple, Union

import numpy as np
import onnx
from sklearn.covariance import EllipticEnvelope

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(EllipticEnvelope)
def sklearn_elliptic_envelope(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: EllipticEnvelope,
    X: str,
    name: str = "elliptic_envelope",
) -> Union[str, Tuple[str, str]]:
    """
    Converts a :class:`sklearn.covariance.EllipticEnvelope` into ONNX.

    The converter produces two outputs:

    * ``label``: the outlier predictions (``1`` for inlier, ``-1`` for outlier),
      equivalent to :meth:`~sklearn.covariance.EllipticEnvelope.predict`.
    * ``scores``: the decision function values, equivalent to
      :meth:`~sklearn.covariance.EllipticEnvelope.decision_function`.

    The decision function is computed as:

    .. math::

        \\text{decision}(x) = -\\lVert x - \\mu \\rVert^2_{\\Sigma^{-1}} - \\text{offset}

    where :math:`\\mu` is :attr:`location_`, :math:`\\Sigma^{-1}` is
    :attr:`precision_`, and :attr:`offset_` is the threshold learned during
    fitting.  The squared Mahalanobis distance is:

    .. math::

        d^2(x) = (x - \\mu)^\\top \\, \\Sigma^{-1} \\, (x - \\mu)
                = \\sum_j \\bigl[(x - \\mu) \\Sigma^{-1}\\bigr]_j \\cdot (x - \\mu)_j

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
                                   ReduceSum(axis=1) ──► mahal_sq (N,)
                                              │
                                           Neg ──► score_samples (N,)
                                              │
                                   Sub(offset_) ──► decision (N,) [scores output]
                                              │
                             GreaterOrEqual(0) ──► mask (N,)
                                              │
                              Where(1, -1) ──► label (N,) [label output]

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names; ``outputs[0]`` receives the predicted
        labels and ``outputs[1]`` (if present) receives the decision function
        values
    :param estimator: a fitted ``EllipticEnvelope``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: tuple ``(label, scores)``
    """
    assert isinstance(
        estimator, EllipticEnvelope
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    location = estimator.location_.astype(dtype)       # (F,)
    precision = estimator.precision_.astype(dtype)     # (F, F)
    offset = np.array([estimator.offset_], dtype=dtype)  # (1,)

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
    )  # (N,)

    # score_samples = -mahal_sq
    score_samples = g.op.Neg(mahal_sq, name=f"{name}_neg")  # (N,)

    # decision_function = score_samples - offset_
    n_outputs = len(outputs)
    if n_outputs >= 2:
        decision = g.op.Sub(
            score_samples, offset, name=f"{name}_decision", outputs=outputs[1:2]
        )  # (N,)
        assert isinstance(decision, str)
        if not sts:
            g.set_type(decision, itype)
            if g.has_rank(X):
                g.set_rank(decision, 1)
    else:
        decision = g.op.Sub(score_samples, offset, name=f"{name}_decision")  # (N,)
        assert isinstance(decision, str)

    # predict: 1 where decision >= 0, else -1
    zero = np.array([0], dtype=dtype)
    mask = g.op.GreaterOrEqual(decision, zero, name=f"{name}_mask")  # (N,) bool

    one = np.array([1], dtype=np.int64)
    minus_one = np.array([-1], dtype=np.int64)
    label = g.op.Where(
        mask, one, minus_one, name=f"{name}_where", outputs=outputs[:1]
    )  # (N,)
    assert isinstance(label, str)
    if not sts:
        g.set_type(label, onnx.TensorProto.INT64)
        if g.has_rank(X):
            g.set_rank(label, 1)

    if n_outputs >= 2:
        return label, decision
    return label
