"""
Converts a :class:`sklearn.linear_model.LogisticRegression` into an ONNX graph.
"""
import numpy as np
from onnx import TensorProto
from ..xbuilder import GraphBuilder


def _add_logistic_regression_nodes(
    g: GraphBuilder,
    clf,
    input_name: str,
    output_label_name: str,
    output_proba_name: str,
    prefix: str = "",
) -> tuple:
    """
    Adds LogisticRegression nodes to an existing :class:`GraphBuilder`.

    :param g: the graph builder to add nodes to
    :param clf: a fitted ``LogisticRegression``
    :param input_name: name of the input result
    :param output_label_name: desired name for the label output
    :param output_proba_name: desired name for the probabilities output
    :param prefix: prefix for initializer names (avoids collisions in pipelines)
    :return: tuple ``(label_result_name, proba_result_name)``
    """
    coef = clf.coef_.astype(np.float32)
    intercept = clf.intercept_.astype(np.float32)
    classes = clf.classes_
    n_classes = len(classes)
    is_binary = coef.shape[0] == 1

    coef_name = g.make_initializer(f"{prefix}coef", coef)
    intercept_name = g.make_initializer(f"{prefix}intercept", intercept)

    decision = g.op.Gemm(
        input_name, coef_name, intercept_name,
        transB=1, name=f"{prefix}gemm_decision",
    )

    if is_binary:
        proba_pos = g.op.Sigmoid(decision, name=f"{prefix}sigmoid")
        ones = g.make_initializer(
            f"{prefix}ones_scalar", np.array([[1.0]], dtype=np.float32)
        )
        proba_neg = g.op.Sub(ones, proba_pos, name=f"{prefix}sub_proba")
        proba = g.op.Concat(proba_neg, proba_pos, axis=1, name=f"{prefix}concat_proba")
    else:
        proba = g.op.Softmax(decision, axis=1, name=f"{prefix}softmax")

    label_idx = g.op.ArgMax(proba, axis=1, keepdims=0, name=f"{prefix}argmax")
    label_idx_cast = g.op.Cast(label_idx, to=TensorProto.INT64, name=f"{prefix}cast_idx")

    if np.issubdtype(classes.dtype, np.integer):
        classes_arr = classes.astype(np.int64)
        classes_name = g.make_initializer(f"{prefix}classes", classes_arr)
        label = g.op.Gather(
            classes_name, label_idx_cast, axis=0,
            name=f"{prefix}gather_label", outputs=[output_label_name],
        )
        label_dtype = TensorProto.INT64
    else:
        classes_arr = np.array(classes.astype(str))
        classes_name = g.make_initializer(f"{prefix}classes", classes_arr)
        label = g.op.Gather(
            classes_name, label_idx_cast, axis=0,
            name=f"{prefix}gather_label", outputs=[output_label_name],
        )
        label_dtype = TensorProto.STRING

    proba_out = g.op.Identity(
        proba, name=f"{prefix}identity_proba", outputs=[output_proba_name]
    )
    return label, label_dtype, proba_out, n_classes


def convert_logistic_regression(
    clf,
    input_name: str = "X",
    output_label_name: str = "label",
    output_proba_name: str = "probabilities",
    opset: int = 18,
) -> GraphBuilder:
    """
    Converts a fitted :class:`sklearn.linear_model.LogisticRegression` into
    a :class:`GraphBuilder`.

    :param clf: a fitted ``LogisticRegression``
    :param input_name: name of the input
    :param output_label_name: name of the predicted label output
    :param output_proba_name: name of the probabilities output
    :param opset: ONNX opset version
    :return: :class:`GraphBuilder` ready to be exported with ``to_onnx()``
    """
    n_features = clf.coef_.shape[1]
    g = GraphBuilder(opset, ir_version=9)
    g.make_tensor_input(input_name, TensorProto.FLOAT, (None, n_features))

    label, label_dtype, proba_out, n_classes = _add_logistic_regression_nodes(
        g, clf, input_name, output_label_name, output_proba_name
    )
    g.make_tensor_output(label, label_dtype, (None,), indexed=False)
    g.make_tensor_output(proba_out, TensorProto.FLOAT, (None, n_classes), indexed=False)
    return g
