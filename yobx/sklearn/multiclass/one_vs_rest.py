from typing import Tuple, Dict, List, Union

import numpy as np
import onnx
from sklearn.multiclass import OneVsRestClassifier

from ..register import register_sklearn_converter, get_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ..sklearn_helper import get_n_expected_outputs


@register_sklearn_converter(OneVsRestClassifier)
def sklearn_one_vs_rest_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: OneVsRestClassifier,
    X: str,
    name: str = "one_vs_rest",
) -> Union[str, Tuple[str, str]]:
    """
    Converts a :class:`sklearn.multiclass.OneVsRestClassifier` into ONNX.

    The converter iterates over the fitted binary sub-estimators, calls the
    registered converter for each one to obtain per-class positive-class
    probabilities, stacks them into a score matrix, and then derives the
    final label and (optionally) the probability matrix.

    **Multiclass** (``len(estimators_) > 1``):

    .. code-block:: text

        X ──[sub-est 0 converter]──► proba_0 (Nx2) ──Slice[:, 1]──► pos_0 (Nx1)
        X ──[sub-est 1 converter]──► proba_1 (Nx2) ──Slice[:, 1]──► pos_1 (Nx1)
        ...
                                                 Concat(pos_0, pos_1, ..., axis=1)
                                                             │
                                                          scores (NxC)
                                                             │
                                                    ReduceSum(axis=1) ──►  sum_
                                                             │
                                                        Div(scores, sum_) ──► proba (NxC)
                                                             │
                                               ┌─────────────┘
                                           ArgMax(axis=1) ──Cast──Gather(classes) ──► label

    **Binary** (``len(estimators_) == 1``):

    .. code-block:: text

        X ──[sub-est 0 converter]──► proba_0 (Nx2) ──Slice[:, 1]──► pos (Nx1)
                                                          │
                                            Sub(1, pos) ──┤
                                                  │       │
                                              neg (Nx1)   │
                          Concat(axis=1)──► proba (Nx2) ──┘
                               │
                             ArgMax(axis=1) ──Cast──Gather(classes) ──► label

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names (label, or label + probabilities)
    :param estimator: a fitted :class:`~sklearn.multiclass.OneVsRestClassifier`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: label tensor name, or tuple ``(label, probabilities)``
    :raises NotImplementedError: when ``estimator.multilabel_`` is ``True`` or
        when a sub-estimator does not expose :meth:`predict_proba`
    """
    assert isinstance(
        estimator, OneVsRestClassifier
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    if estimator.multilabel_:
        raise NotImplementedError(
            "OneVsRestClassifier with multilabel_ = True is not yet supported."
        )

    itype = g.get_type(X)
    dtype = g.onnx_dtype_to_np_dtype(itype)
    classes = estimator.classes_
    emit_proba = len(outputs) > 1

    # Collect positive-class probabilities from each binary sub-estimator.
    pos_probs: List[str] = []
    for i, sub_est in enumerate(estimator.estimators_):
        sub_name = f"{name}__est{i}"

        n_sub_outputs = get_n_expected_outputs(sub_est)
        if n_sub_outputs < 2:
            raise NotImplementedError(
                f"Sub-estimator {type(sub_est).__name__} does not expose predict_proba. "
                "Only sub-estimators with predict_proba are supported."
            )

        # Call the registered sub-estimator converter.
        sub_label = g.unique_name(f"{sub_name}_label")
        sub_proba = g.unique_name(f"{sub_name}_proba")
        fct = get_sklearn_converter(type(sub_est))
        fct(g, sts, [sub_label, sub_proba], sub_est, X, name=sub_name)

        # Some converters (e.g. TreeEnsembleClassifier in ai.onnx.ml) do not
        # register the output type in the graph builder because the domain's
        # type inference is not implemented.  Fall back to FLOAT, which is the
        # universal probability dtype for sklearn classifiers.
        if not g.has_type(sub_proba):
            g.set_type(sub_proba, onnx.TensorProto.FLOAT)

        # Extract column 1 (positive-class probability): [n, 2] → [n, 1].
        pos_prob = g.op.Slice(
            sub_proba,
            np.array([1], dtype=np.int64),  # starts
            np.array([2], dtype=np.int64),  # ends
            np.array([1], dtype=np.int64),  # axes
            name=f"{sub_name}_slice",
        )
        pos_probs.append(pos_prob)  # type: ignore[arg-type]

    # Build the score matrix.
    if len(pos_probs) == 1:
        # Binary case (single sub-estimator): build [1 - p, p] matrix.
        pos = pos_probs[0]  # shape [n, 1]
        neg = g.op.Sub(np.array([1], dtype=dtype), pos, name=f"{name}_neg")  # shape [n, 1]
        scores = g.op.Concat(neg, pos, axis=1, name=f"{name}_scores")  # [n, 2]
    else:
        # Multiclass: stack positive-class probabilities, then normalise.
        scores = g.op.Concat(*pos_probs, axis=1, name=f"{name}_scores")  # [n, n_classes]

    # Normalise rows so they sum to 1.
    sum_ = g.op.ReduceSum(
        scores,
        np.array([1], dtype=np.int64),
        keepdims=1,
        name=f"{name}_sum",
    )
    proba_norm = g.op.Div(scores, sum_, name=f"{name}_normalize")

    # Argmax → label index → class label.
    label_idx_raw = g.op.ArgMax(proba_norm, axis=1, keepdims=0, name=f"{name}_argmax")
    label_idx = g.op.Cast(label_idx_raw, to=onnx.TensorProto.INT64, name=f"{name}_cast")

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

    if emit_proba:
        proba = g.op.Identity(proba_norm, name=f"{name}_proba", outputs=outputs[1:])
        assert isinstance(proba, str)
        return label, proba

    return label
