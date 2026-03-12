from typing import Tuple, Dict, List, Union

import numpy as np
import onnx
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from ..register import register_sklearn_converter, get_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ..sklearn_helper import get_n_expected_outputs


@register_sklearn_converter(MultiOutputRegressor)
def sklearn_multi_output_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: MultiOutputRegressor,
    X: str,
    name: str = "multi_output_regressor",
) -> str:
    """
    Converts a :class:`sklearn.multioutput.MultiOutputRegressor` into ONNX.

    The converter iterates over the fitted sub-estimators, calls the registered
    converter for each one, and reshapes every 1-D prediction ``(N,)`` to
    ``(N, 1)`` before concatenating them into the final output of shape
    ``(N, n_targets)``.

    Graph structure:

    .. code-block:: text

        X ──[sub-est 0 converter]──► pred_0 (N,) ──Reshape(N,1)──┐
        X ──[sub-est 1 converter]──► pred_1 (N,) ──Reshape(N,1)──┤
        ...                                                        │
                                              Concat(axis=1) ─────► predictions (N, n_targets)

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names
    :param estimator: a fitted :class:`~sklearn.multioutput.MultiOutputRegressor`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: name of the output tensor of shape ``(N, n_targets)``
    """
    assert isinstance(
        estimator, MultiOutputRegressor
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)

    per_target_preds: List[str] = []
    for i, sub_est in enumerate(estimator.estimators_):
        sub_name = f"{name}__est{i}"
        sub_output = g.unique_name(f"{sub_name}_pred")
        fct = get_sklearn_converter(type(sub_est))
        fct(g, sts, [sub_output], sub_est, X, name=sub_name)

        if not g.has_type(sub_output):
            g.set_type(sub_output, itype)

        # Reshape (N,) → (N, 1)
        reshaped = g.op.Reshape(
            sub_output,
            np.array([-1, 1], dtype=np.int64),
            name=f"{sub_name}_reshape",
        )
        per_target_preds.append(reshaped)

    # Concatenate along axis 1: (N, 1) × n_targets → (N, n_targets)
    if len(per_target_preds) == 1:
        predictions = g.op.Identity(
            per_target_preds[0], name=f"{name}_pred", outputs=outputs[:1]
        )
    else:
        predictions = g.op.Concat(
            *per_target_preds, axis=1, name=f"{name}_concat", outputs=outputs[:1]
        )

    if not sts:
        g.set_type(predictions, itype)
        n_targets = len(estimator.estimators_)
        if g.has_shape(X):
            batch_dim = g.get_shape(X)[0]
            g.set_shape(predictions, (batch_dim, n_targets))
        elif g.has_rank(X):
            g.set_rank(predictions, 2)

    return predictions


@register_sklearn_converter(MultiOutputClassifier)
def sklearn_multi_output_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: MultiOutputClassifier,
    X: str,
    name: str = "multi_output_classifier",
) -> Union[str, Tuple[str, str]]:
    """
    Converts a :class:`sklearn.multioutput.MultiOutputClassifier` into ONNX.

    The converter iterates over the fitted sub-estimators, calls the registered
    converter for each one to obtain per-target labels, and concatenates the
    reshaped ``(N, 1)`` labels into a final label tensor of shape
    ``(N, n_targets)``.

    When probabilities are requested (i.e. ``len(outputs) > 1``) and every
    target has the same number of classes, the per-target probability matrices
    ``(N, n_classes)`` are unsqueezed to ``(N, 1, n_classes)`` and
    concatenated into a ``(N, n_targets, n_classes)`` tensor.

    Graph structure (labels only):

    .. code-block:: text

        X ──[sub-est 0 converter]──► label_0 (N,) ──Cast(INT64)──Reshape(N,1)──┐
        X ──[sub-est 1 converter]──► label_1 (N,) ──Cast(INT64)──Reshape(N,1)──┤
        ...                                                                      │
                                                        Concat(axis=1) ─────────► labels (N, n_targets)

    Graph structure (with probabilities, all targets same n_classes):

    .. code-block:: text

        X ──[sub-est 0 converter]──► proba_0 (N, n_cls) ──Unsqueeze──► (N,1,n_cls)──┐
        X ──[sub-est 1 converter]──► proba_1 (N, n_cls) ──Unsqueeze──► (N,1,n_cls)──┤
        ...                                                                           │
                                                           Concat(axis=1) ───────────► probabilities (N, n_targets, n_cls)

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names (label, or label + probabilities)
    :param estimator: a fitted :class:`~sklearn.multioutput.MultiOutputClassifier`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: label tensor name, or tuple ``(label, probabilities)``
    :raises NotImplementedError: when probabilities are requested but targets
        have a different number of classes or sub-estimators do not expose
        :meth:`predict_proba`
    """
    assert isinstance(
        estimator, MultiOutputClassifier
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    emit_proba = len(outputs) > 1
    classes_list = estimator.classes_

    # Check that all targets use the same class dtype (no mixing int and string).
    all_integer = all(np.issubdtype(c.dtype, np.integer) for c in classes_list)
    all_string = all(not np.issubdtype(c.dtype, np.integer) for c in classes_list)
    if not all_integer and not all_string:
        raise NotImplementedError(
            "MultiOutputClassifier with mixed integer/string classes across targets "
            "is not supported."
        )

    if emit_proba:
        n_classes_per_target = [c.shape[0] for c in classes_list]
        if len(set(n_classes_per_target)) > 1:
            raise NotImplementedError(
                f"MultiOutputClassifier probabilities requested but targets have "
                f"different numbers of classes: {n_classes_per_target}. "
                "All targets must have the same number of classes to produce a "
                "single probability tensor."
            )

    per_target_labels: List[str] = []
    per_target_probas: List[str] = []

    for i, sub_est in enumerate(estimator.estimators_):
        sub_name = f"{name}__est{i}"
        n_sub_outputs = get_n_expected_outputs(sub_est)

        sub_label = g.unique_name(f"{sub_name}_label")
        if emit_proba and n_sub_outputs >= 2:
            sub_proba = g.unique_name(f"{sub_name}_proba")
            fct = get_sklearn_converter(type(sub_est))
            fct(g, sts, [sub_label, sub_proba], sub_est, X, name=sub_name)

            if not g.has_type(sub_proba):
                g.set_type(sub_proba, onnx.TensorProto.FLOAT)

            per_target_probas.append(sub_proba)
        elif n_sub_outputs >= 2:
            sub_proba_unused = g.unique_name(f"{sub_name}_proba_unused")
            fct = get_sklearn_converter(type(sub_est))
            fct(g, sts, [sub_label, sub_proba_unused], sub_est, X, name=sub_name)
        else:
            fct = get_sklearn_converter(type(sub_est))
            fct(g, sts, [sub_label], sub_est, X, name=sub_name)

        if not g.has_type(sub_label):
            g.set_type(
                sub_label,
                onnx.TensorProto.INT64 if all_integer else onnx.TensorProto.STRING,
            )

        if all_integer:
            # Normalise to INT64 so all targets share the same label dtype.
            label_typed = g.op.Cast(
                sub_label, to=onnx.TensorProto.INT64, name=f"{sub_name}_cast"
            )
        else:
            label_typed = sub_label

        # Reshape (N,) → (N, 1)
        reshaped = g.op.Reshape(
            label_typed,
            np.array([-1, 1], dtype=np.int64),
            name=f"{sub_name}_reshape",
        )
        per_target_labels.append(reshaped)

    # Concatenate labels along axis 1: (N, 1) × n_targets → (N, n_targets)
    if len(per_target_labels) == 1:
        labels = g.op.Identity(
            per_target_labels[0], name=f"{name}_label", outputs=outputs[:1]
        )
    else:
        labels = g.op.Concat(
            *per_target_labels, axis=1, name=f"{name}_label_concat", outputs=outputs[:1]
        )

    if not sts:
        g.set_type(
            labels,
            onnx.TensorProto.INT64 if all_integer else onnx.TensorProto.STRING,
        )
        n_targets = len(estimator.estimators_)
        if g.has_shape(X):
            batch_dim = g.get_shape(X)[0]
            g.set_shape(labels, (batch_dim, n_targets))
        elif g.has_rank(X):
            g.set_rank(labels, 2)

    if not emit_proba:
        return labels

    if emit_proba and not per_target_probas:
        raise NotImplementedError(
            "MultiOutputClassifier: probabilities were requested but sub-estimators "
            "do not expose predict_proba."
        )

    # Unsqueeze each proba: (N, n_classes) → (N, 1, n_classes)
    # then Concat along axis 1: (N, n_targets, n_classes)
    unsqueezed: List[str] = []
    for i, proba in enumerate(per_target_probas):
        unsq = g.op.Unsqueeze(
            proba,
            np.array([1], dtype=np.int64),
            name=f"{name}__est{i}_unsqueeze",
        )
        unsqueezed.append(unsq)

    if len(unsqueezed) == 1:
        probabilities = g.op.Identity(
            unsqueezed[0], name=f"{name}_proba", outputs=outputs[1:]
        )
    else:
        probabilities = g.op.Concat(
            *unsqueezed, axis=1, name=f"{name}_proba_concat", outputs=outputs[1:]
        )

    if not sts:
        n_targets = len(estimator.estimators_)
        n_classes = classes_list[0].shape[0]
        if g.has_shape(X):
            batch_dim = g.get_shape(X)[0]
            g.set_shape(probabilities, (batch_dim, n_targets, n_classes))
        elif g.has_rank(X):
            g.set_rank(probabilities, 3)

    return labels, probabilities
