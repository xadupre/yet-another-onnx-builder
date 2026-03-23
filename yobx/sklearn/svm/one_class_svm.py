from typing import Dict, List, Tuple, Union

import numpy as np
import onnx
from sklearn.svm import OneClassSVM

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from .svm import _get_kernel_params


@register_sklearn_converter(OneClassSVM)
def sklearn_one_class_svm(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: OneClassSVM,
    X: str,
    name: str = "one_class_svm",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.svm.OneClassSVM` into ONNX using the
    ``SVMRegressor`` operator from the ``ai.onnx.ml`` domain.

    **Supported kernels**: ``'linear'``, ``'poly'``, ``'rbf'``, ``'sigmoid'``.
    Callable kernels are not supported.

    The converter produces two outputs:

    * ``label``: the outlier predictions (``1`` for inlier, ``-1`` for outlier),
      equivalent to :meth:`~sklearn.svm.OneClassSVM.predict`.
    * ``scores``: the decision function values, equivalent to
      :meth:`~sklearn.svm.OneClassSVM.decision_function`.

    .. note::

        The ``SVMRegressor`` operator only supports ``float32`` input.
        When the input tensor is ``float64``, it is cast to ``float32`` before
        the SVM node and the resulting scores are cast back to ``float64``
        afterwards.  This may introduce small numerical differences compared to
        running inference with ``float64`` arithmetic.

    Graph structure:

    .. code-block:: text

        X (float64) ──Cast(float32)──► X_f32
                │
        X (float32) ──────────────────────────┐
                                              ▼
                                SVMRegressor(one_class=0) ──► raw_scores (N, 1) [float32]
                                              │
                                         Reshape ──► scores_f32 (N,) [float32]
                                              │
                              [Cast to float64 if input was float64]
                                              │
                                         scores (N,) [input dtype]
                                              │
                                   GreaterOrEqual(0) ──► mask (N,)
                                              │
                                   Where(1, -1) ──► label (N,)

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names; ``outputs[0]`` receives the predicted
        labels and ``outputs[1]`` receives the decision function values
    :param estimator: a fitted ``OneClassSVM``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: tuple ``(label, scores)``
    """
    assert isinstance(estimator, OneClassSVM), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    is_double = itype == onnx.TensorProto.DOUBLE

    kernel_type, kernel_params = _get_kernel_params(estimator)

    # SVMRegressor always works in float32; coefficients and support vectors
    # are stored as float32 regardless of the input dtype.
    coefficients = estimator.dual_coef_.astype(np.float32).flatten().tolist()
    support_vectors = estimator.support_vectors_.astype(np.float32).flatten().tolist()
    n_supports = int(estimator.dual_coef_.shape[1])
    rho = estimator.intercept_.astype(np.float64).tolist()

    # SVMRegressor only supports float32 inputs; cast float64 input first.
    if is_double:
        X_svm = g.op.Cast(X, to=onnx.TensorProto.FLOAT, name=f"{name}_cast_input")
    else:
        X_svm = X

    # Use one_class=0 to get the raw decision function values.
    # The output is always float32 from SVMRegressor.
    raw_scores_name = g.unique_name(f"{name}_raw")
    g.make_node(
        "SVMRegressor",
        [X_svm],
        outputs=[raw_scores_name],
        domain="ai.onnx.ml",
        name=f"{name}_svmr",
        kernel_type=kernel_type,
        kernel_params=[float(v) for v in kernel_params],
        support_vectors=support_vectors,
        coefficients=coefficients,
        n_supports=n_supports,
        rho=[float(v) for v in rho],  # type: ignore
        post_transform="NONE",
        one_class=0,
    )

    # Flatten (N, 1) → (N,) [float32]
    scores_f32 = g.op.Reshape(
        raw_scores_name, np.array([-1], dtype=np.int64), name=f"{name}_reshape"
    )

    # Cast to float64 if the input was float64.
    if is_double:
        scores = g.op.Cast(
            scores_f32,
            to=onnx.TensorProto.DOUBLE,
            name=f"{name}_cast_f64",
            outputs=outputs[1:2],
        )
        zero = np.array([0], dtype=np.float64)
    else:
        scores = g.op.Identity(scores_f32, name=f"{name}_scores", outputs=outputs[1:2])
        zero = np.array([0], dtype=np.float32)

    # label = 1 if decision_function >= 0 else -1
    mask = g.op.GreaterOrEqual(scores, zero, name=f"{name}_mask")
    label = g.op.Where(
        mask,
        np.array([1], dtype=np.int64),
        np.array([-1], dtype=np.int64),
        name=f"{name}_label",
        outputs=outputs[:1],
    )

    return label, scores
