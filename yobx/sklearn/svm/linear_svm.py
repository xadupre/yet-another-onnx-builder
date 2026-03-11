from typing import Dict, List, Tuple, Union
import numpy as np
import onnx
from sklearn.svm import LinearSVC, LinearSVR
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


def _build_label(
    g: GraphBuilderExtendedProtocol,
    classes: np.ndarray,
    label_idx: str,
    name: str,
    outputs: List[str],
    sts: Dict,
) -> str:
    """Gathers the class label from *classes* using *label_idx* as indices."""
    if np.issubdtype(classes.dtype, np.integer):
        classes_arr = classes.astype(np.int64)
        label = g.op.Gather(
            classes_arr,
            label_idx,
            axis=0,
            name=f"{name}_label",
            outputs=outputs[:1],
        )
        assert isinstance(label, str)
        if not sts:
            g.set_type(label, onnx.TensorProto.INT64)
    else:
        classes_arr = np.array(classes.astype(str))
        label = g.op.Gather(
            classes_arr,
            label_idx,
            axis=0,
            name=f"{name}_label_string",
            outputs=outputs[:1],
        )
        assert isinstance(label, str)
        if not sts:
            g.set_type(label, onnx.TensorProto.STRING)
    return label


@register_sklearn_converter(LinearSVC)
def sklearn_linear_svc(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: LinearSVC,
    X: str,
    name: str = "linear_svc",
) -> str:
    """
    Converts a :class:`sklearn.svm.LinearSVC` into ONNX.

    :class:`~sklearn.svm.LinearSVC` does not expose :meth:`predict_proba`, so
    this converter always returns the predicted class label only.

    **Binary classification** (``len(classes_) == 2``):

    .. code-block:: text

        X  ──Gemm(coef, intercept)──►  decision (Nx1)
                                            │
                                        Reshape  ──►  decision_1d (N,)
                                            │
                                        Greater(0) ──Cast(INT64)──Gather(classes) ──►  label

    **Multiclass** (``len(classes_) > 2``):

    .. code-block:: text

        X  ──Gemm(coef, intercept)──►  decision (NxC)
                                            │
                                        ArgMax ──Cast(INT64)──Gather(classes) ──►  label

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names (label only; LinearSVC has no predict_proba)
    :param estimator: a fitted ``LinearSVC``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: label tensor name
    """
    assert isinstance(estimator, LinearSVC), (
        f"Unexpected type {type(estimator)} for estimator."
    )
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    classes = estimator.classes_
    coef = estimator.coef_.astype(dtype)
    intercept = np.atleast_1d(np.asarray(estimator.intercept_)).astype(dtype)

    # Ensure coef is 2D: (n_decision_functions, n_features).
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)

    decision = g.op.Gemm(X, coef, intercept, transB=1, name=f"{name}_decision")

    n_classes = len(classes)
    is_binary = n_classes == 2

    if is_binary:
        decision_1d = g.op.Reshape(
            decision, np.array([-1], dtype=np.int64), name=f"{name}_flatten"
        )
        zero = np.array([0], dtype=dtype)
        gt = g.op.Greater(decision_1d, zero, name=f"{name}_gt")
        label_idx = g.op.Cast(gt, to=onnx.TensorProto.INT64, name=f"{name}_cast")
    else:
        label_idx_raw = g.op.ArgMax(decision, axis=1, keepdims=0, name=f"{name}_argmax")
        label_idx = g.op.Cast(label_idx_raw, to=onnx.TensorProto.INT64, name=f"{name}_cast")

    return _build_label(g, classes, label_idx, name, outputs, sts)


@register_sklearn_converter(LinearSVR)
def sklearn_linear_svr(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: LinearSVR,
    X: str,
    name: str = "linear_svr",
) -> str:
    """
    Converts a :class:`sklearn.svm.LinearSVR` into ONNX.

    The prediction formula is::

        y = X @ coef_.T + intercept_

    Graph structure:

    .. code-block:: text

        X  ──Gemm(coef, intercept, transB=1)──►  predictions

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names
    :param estimator: a fitted ``LinearSVR``
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name
    """
    assert isinstance(estimator, LinearSVR), (
        f"Unexpected type {type(estimator)} for estimator."
    )
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    coef = estimator.coef_.astype(dtype)
    intercept = np.atleast_1d(np.asarray(estimator.intercept_)).astype(dtype)

    # Ensure coef is 2D: (n_targets, n_features) for Gemm with transB=1.
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)

    result = g.op.Gemm(X, coef, intercept, transB=1, name=name, outputs=outputs)
    assert isinstance(result, str)
    if not sts:
        g.set_type(result, itype)
    return result
