from typing import Dict, List, Tuple, Union
import numpy as np
import onnx
from sklearn.linear_model import Perceptron, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype

# Optional deprecated models
_EXTRA_CLASSIFIER_TYPES = []
try:
    from sklearn.linear_model import PassiveAggressiveClassifier  # deprecated in sklearn 1.8

    _EXTRA_CLASSIFIER_TYPES.append(PassiveAggressiveClassifier)
except ImportError:
    pass

_LINEAR_CLASSIFIER_TYPES = (
    Perceptron,
    RidgeClassifier,
    RidgeClassifierCV,
    SGDClassifier,
    *_EXTRA_CLASSIFIER_TYPES,
)


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
            classes_arr, label_idx, axis=0, name=f"{name}_label", outputs=outputs[:1]
        )
        g.set_type(label, onnx.TensorProto.INT64)
    else:
        classes_arr = np.array(classes.astype(str))
        label = g.op.Gather(
            classes_arr, label_idx, axis=0, name=f"{name}_label_string", outputs=outputs[:1]
        )
        g.set_type(label, onnx.TensorProto.STRING)
    return label


@register_sklearn_converter(_LINEAR_CLASSIFIER_TYPES)
def sklearn_linear_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: Union[RidgeClassifier, SGDClassifier, Perceptron],
    X: str,
    name: str = "linear_classifier",
) -> Union[str, Tuple[str, str]]:
    """
    Converts a sklearn linear classifier into ONNX.

    Handles both classifiers that expose :meth:`predict_proba` (e.g.
    :class:`~sklearn.linear_model.SGDClassifier` with ``loss='log_loss'`` or
    ``'modified_huber'``) and those that do not (e.g.
    :class:`~sklearn.linear_model.RidgeClassifier`).

    **Binary classification** (``len(classes_) == 2``):

    .. code-block:: text

        X  ──Gemm(coef, intercept)──►  decision (Nx1)
                                            │
                                        Flatten  ──►  decision_1d (N,)
                                            │
                                  ┌─────────┴──────────┐
                                  │                 (if proba)
                              Greater(0)           Sigmoid ──Sub(1,·)──Concat
                                  │                                        │
                                 Cast(INT64)                            proba (Nx2)
                                  │
                             Gather(classes) ──►  label

    **Multiclass** (``len(classes_) > 2``):

    .. code-block:: text

        X  ──Gemm(coef, intercept)──►  decision (NxC)
                                            │
                                  ┌─────────┴──────────┐
                                  │                 (if proba)
                               ArgMax              Softmax ──►  proba (NxC)
                                  │
                              Cast(INT64)
                                  │
                             Gather(classes) ──►  label

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output names; one name (label only) or two names
        (label + probabilities) depending on :func:`get_output_names`
    :param estimator: a fitted linear classifier
    :param X: input tensor name
    :param name: prefix for added node names
    :return: label tensor name, or tuple ``(label, probabilities)`` when
        the estimator supports :meth:`predict_proba`
    """
    assert isinstance(
        estimator, _LINEAR_CLASSIFIER_TYPES
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    classes = estimator.classes_
    coef = estimator.coef_.astype(dtype)
    intercept = np.atleast_1d(np.asarray(estimator.intercept_)).astype(dtype)

    # Ensure coef is 2D: (n_decision_functions, n_features).
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)

    # Gemm output: (N, 1) for binary (coef shape (1, n_features))
    #              (N, n_classes) for multiclass
    decision = g.op.Gemm(X, coef, intercept, transB=1, name=f"{name}_decision")

    n_classes = len(classes)
    is_binary = n_classes == 2
    emit_proba = len(outputs) > 1

    if is_binary:
        # Flatten (N, 1) → (N,) for scalar comparison
        decision_1d = g.op.Reshape(
            decision, np.array([-1], dtype=np.int64), name=f"{name}_flatten"
        )
        zero = np.array([0], dtype=dtype)
        gt = g.op.Greater(decision_1d, zero, name=f"{name}_gt")
        label_idx = g.op.Cast(gt, to=onnx.TensorProto.INT64, name=f"{name}_cast")

        label = _build_label(g, classes, label_idx, name, outputs, sts)

        if emit_proba:
            # decision has shape (N, 1); Sigmoid gives (N, 1); 1-sigmoid is (N, 1).
            # Concat along axis=1 → (N, 2).
            proba_pos = g.op.Sigmoid(decision, name=f"{name}_sigmoid")
            proba_neg = g.op.Sub(np.array([1], dtype=dtype), proba_pos, name=f"{name}_proba_neg")
            proba = g.op.Concat(
                proba_neg, proba_pos, axis=1, name=f"{name}_concat", outputs=outputs[1:]
            )
            return label, proba
    else:
        label_idx_raw = g.op.ArgMax(decision, axis=1, keepdims=0, name=f"{name}_argmax")
        label_idx = g.op.Cast(label_idx_raw, to=onnx.TensorProto.INT64, name=f"{name}_cast")

        label = _build_label(g, classes, label_idx, name, outputs, sts)

        if emit_proba:
            # SGDClassifier with probabilistic losses uses one-vs-rest (OvR):
            # apply sigmoid to each score, then normalise so rows sum to 1.
            sig = g.op.Sigmoid(decision, name=f"{name}_sigmoid")
            sum_ = g.op.ReduceSum(
                sig, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_sum"
            )
            proba = g.op.Div(sig, sum_, name=f"{name}_normalize", outputs=outputs[1:])
            return label, proba

    return label
