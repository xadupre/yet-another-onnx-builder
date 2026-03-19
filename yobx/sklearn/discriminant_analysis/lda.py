from typing import Dict, List, Tuple
import numpy as np
import onnx
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(LinearDiscriminantAnalysis)
def sklearn_linear_discriminant_analysis(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: LinearDiscriminantAnalysis,
    X: str,
    name: str = "lda",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
    into ONNX.

    The decision function is computed as a linear transformation of the input,
    and probabilities are derived from it:

    **Binary classification** (``coef_.shape[0] == 1``):

    .. code-block:: text

        X  ──Gemm(coef, intercept)──►  decision (Nx1)
                                            │
                                   ┌────────┴────────┐
                                Sigmoid           Sub(1, ·)
                                   │                  │
                                proba_pos          proba_neg
                                   └────────┬────────┘
                                         Concat  ──►  probabilities
                                            │
                                         ArgMax ──Cast──Gather(classes) ──►  label

    **Multiclass** (``coef_.shape[0] > 1``):

    .. code-block:: text

        X  ──Gemm(coef, intercept)──►  decision (NxC)
                                            │
                                        Softmax  ──►  probabilities
                                            │
                                        ArgMax ──Cast──Gather(classes)  ──►  label

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``LinearDiscriminantAnalysis``
    :param outputs: desired names (label, probabilities)
    :param X: input tensor name
    :param name: prefix names for the added nodes
    :return: tuple ``(label_result_name, proba_result_name)``
    """
    assert isinstance(
        estimator, LinearDiscriminantAnalysis
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    coef = estimator.coef_.astype(dtype)
    intercept = estimator.intercept_.astype(dtype)

    classes = estimator.classes_
    is_binary = coef.shape[0] == 1

    decision = g.op.Gemm(X, coef, intercept, transB=1, name=f"{name}_decision")

    if is_binary:
        proba_pos = g.op.Sigmoid(decision, name=name)
        proba_neg = g.op.Sub(np.array([1], dtype=dtype), proba_pos, name=name)
        proba = g.op.Concat(proba_neg, proba_pos, axis=-1, name=name, outputs=outputs[1:])
    else:
        proba = g.op.Softmax(decision, axis=1, name=name, outputs=outputs[1:])

    assert isinstance(proba, str)
    label_idx = g.op.ArgMax(proba, axis=1, keepdims=0, name=name)
    label_idx_cast = g.op.Cast(label_idx, to=onnx.TensorProto.INT64, name=name)

    if np.issubdtype(classes.dtype, np.integer):
        classes_arr = classes.astype(np.int64)
        label = g.op.Gather(
            classes_arr, label_idx_cast, axis=0, name=f"{name}_label", outputs=outputs[:1]
        )
        assert isinstance(label, str)
        g.set_type(label, onnx.TensorProto.INT64)
    else:
        classes_arr = np.array(classes.astype(str))
        label = g.op.Gather(
            classes_arr, label_idx_cast, axis=0, name=f"{name}_label_string", outputs=outputs[:1]
        )
        assert isinstance(label, str)
        g.set_type(label, onnx.TensorProto.STRING)
    return label, proba
