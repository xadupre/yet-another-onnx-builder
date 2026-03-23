from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.linear_model import SGDOneClassSVM

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(SGDOneClassSVM)
def sklearn_sgd_one_class_svm(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: SGDOneClassSVM,
    X: str,
    name: str = "sgd_one_class_svm",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.linear_model.SGDOneClassSVM` into ONNX.

    :class:`~sklearn.linear_model.SGDOneClassSVM` is a linear approximation
    of One-Class SVM trained via SGD.  The decision function is::

        decision_function(X) = X @ coef_.T - offset_

    Samples with ``decision_function(X) >= 0`` are predicted as inliers (+1)
    and those with ``decision_function(X) < 0`` as outliers (-1).

    This converter inherits from :class:`~sklearn.base.OutlierMixin`, so it
    returns two outputs: the predicted label and the raw decision scores.

    Graph structure:

    .. code-block:: text

        X  ──Gemm(coef_, offset_)──►  decision (N,)
                    │
            GreaterOrEqual(0) ──►  is_inlier (N,)
                    │
            Where(is_inlier, 1, -1) ──►  label (N,)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names; ``outputs[0]`` receives the predicted
        labels and ``outputs[1]`` the decision function scores
    :param estimator: a fitted :class:`~sklearn.linear_model.SGDOneClassSVM`
    :param X: input tensor name
    :param name: prefix for added node names
    :return: tuple ``(label, scores)``
    """
    assert isinstance(
        estimator, SGDOneClassSVM
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # coef_ has shape (n_features,); reshape to (1, n_features) for Gemm (transB=1).
    coef = estimator.coef_.astype(dtype).reshape(1, -1)
    # offset_ has shape (1,); negate so Gemm bias = -offset_.
    neg_offset = (-estimator.offset_).astype(dtype)

    # decision = X @ coef_.T - offset_  ≡  Gemm(X, coef, -offset, transB=1)
    # Gemm output shape: (N, 1)
    decision_2d = g.op.Gemm(X, coef, neg_offset, transB=1, name=f"{name}_gemm")

    # Flatten (N, 1) → (N,)
    decision = g.op.Reshape(decision_2d, np.array([-1], dtype=np.int64), name=f"{name}_flatten")

    # label = 1 if decision >= 0 else -1
    zero = np.zeros(1, dtype=dtype)
    is_inlier = g.op.GreaterOrEqual(decision, zero, name=f"{name}_ge")
    label = g.op.Where(
        is_inlier,
        np.array([1], dtype=np.int64),
        np.array([-1], dtype=np.int64),
        name=f"{name}_label",
        outputs=outputs[:1],
    )

    scores_out = g.op.Identity(decision, name=f"{name}_scores", outputs=outputs[1:2])
    return label, scores_out
