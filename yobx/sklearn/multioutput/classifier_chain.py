from typing import Tuple, Dict, List, Union

import numpy as np
import onnx
from sklearn.multioutput import ClassifierChain

from ..register import register_sklearn_converter, get_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ..sklearn_helper import get_n_expected_outputs


@register_sklearn_converter(ClassifierChain)
def sklearn_classifier_chain(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: ClassifierChain,
    X: str,
    name: str = "classifier_chain",
) -> Union[str, Tuple[str, str]]:
    """
    Converts a :class:`sklearn.multioutput.ClassifierChain` into ONNX.

    Each sub-estimator in the chain predicts one binary target using the
    original features augmented with the binary predictions from all preceding
    steps (in chain order).  After all steps the per-step predictions and
    probabilities are reordered to match the original target column order.

    Graph structure (labels only, identity order):

    .. code-block:: text

        X ──[est 0 converter]──► label_0 ──Cast(float)──Reshape(N,1)──┐ pred_0_col
        │                                                               │
        Concat(X, pred_0_col) ──[est 1 converter]──► label_1 ──Cast──Reshape──┐ pred_1_col
        │                                                                       │
        ...                                                                    │
                                                    Concat(axis=1) ────────────► labels (N, n_targets)

    When the chain ``order_`` is not the identity permutation, the concatenated
    predictions (in chain order) are reordered via ``Gather`` using the
    inverse permutation so that the output columns match the original target order.

    Graph structure (with probabilities):

    The probability for *class 1* is extracted from each sub-estimator's
    ``(N, 2)`` probability output, reshaped to ``(N, 1)``, concatenated into
    ``(N, n_targets)`` in chain order, then reordered in the same way as the
    labels.

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names (label, or label + probabilities)
    :param estimator: a fitted :class:`~sklearn.multioutput.ClassifierChain`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: label tensor name, or tuple ``(label, probabilities)``
    :raises NotImplementedError: when probabilities are requested but
        sub-estimators do not expose :meth:`predict_proba`
    """
    assert isinstance(
        estimator, ClassifierChain
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    emit_proba = len(outputs) > 1

    order = np.asarray(estimator.order_)
    n_targets = len(estimator.estimators_)
    is_identity_order = np.array_equal(order, np.arange(n_targets))

    # Inverse permutation for reordering from chain order to original column order.
    # If not needed we compute it lazily (only when actually non-identity).
    inv_order = np.argsort(order).astype(np.int64) if not is_identity_order else None

    chain_pred_cols: List[str] = []   # (N, 1) float predictions in chain order
    chain_proba_cols: List[str] = []  # (N, 1) probabilities in chain order

    # X_aug grows with each step: starts as X, then Concat(X_aug, pred_col)
    X_aug: str = X

    for i, sub_est in enumerate(estimator.estimators_):
        sub_name = f"{name}__est{i}"
        n_sub_outputs = get_n_expected_outputs(sub_est)

        sub_label = g.unique_name(f"{sub_name}_label")

        if emit_proba and n_sub_outputs >= 2:
            sub_proba = g.unique_name(f"{sub_name}_proba")
            fct = get_sklearn_converter(type(sub_est))
            fct(g, sts, [sub_label, sub_proba], sub_est, X_aug, name=sub_name)

            if not g.has_type(sub_proba):
                g.set_type(sub_proba, itype)

            # Extract probability of class 1: Gather(sub_proba, [1], axis=1) → (N, 1)
            proba_col = g.op.Gather(
                sub_proba,
                np.array([1], dtype=np.int64),
                axis=1,
                name=f"{sub_name}_proba1",
            )
            chain_proba_cols.append(proba_col)
        else:
            fct = get_sklearn_converter(type(sub_est))
            fct(g, sts, [sub_label], sub_est, X_aug, name=sub_name)

        if not g.has_type(sub_label):
            g.set_type(sub_label, onnx.TensorProto.INT64)

        # Cast label (int64) to input float dtype for augmentation and output.
        pred_float = g.op.Cast(sub_label, to=itype, name=f"{sub_name}_cast")

        # Reshape to (N, 1) for column-wise concatenation.
        pred_col = g.op.Reshape(
            pred_float,
            np.array([-1, 1], dtype=np.int64),
            name=f"{sub_name}_pred_col",
        )
        chain_pred_cols.append(pred_col)

        # Augment input for the next step (not needed after the last step).
        if i < n_targets - 1:
            X_aug = g.op.Concat(X_aug, pred_col, axis=1, name=f"{sub_name}_aug")

    # Concatenate all per-step predictions: (N, n_targets) in chain order.
    if len(chain_pred_cols) == 1:
        chain_preds = g.op.Identity(chain_pred_cols[0], name=f"{name}_chain_preds")
    else:
        chain_preds = g.op.Concat(
            *chain_pred_cols, axis=1, name=f"{name}_chain_preds_concat"
        )

    # Reorder to original column order via Gather on axis 1.
    if is_identity_order:
        labels = g.op.Identity(chain_preds, name=f"{name}_label", outputs=outputs[:1])
    else:
        labels = g.op.Gather(
            chain_preds,
            inv_order,
            axis=1,
            name=f"{name}_label",
            outputs=outputs[:1],
        )

    if not sts:
        g.set_type(labels, itype)
        if g.has_shape(X):
            batch_dim = g.get_shape(X)[0]
            g.set_shape(labels, (batch_dim, n_targets))
        elif g.has_rank(X):
            g.set_rank(labels, 2)

    if not emit_proba:
        return labels

    if not chain_proba_cols:
        raise NotImplementedError(
            "ClassifierChain: probabilities were requested but sub-estimators "
            "do not expose predict_proba."
        )

    # Concatenate all per-step probabilities: (N, n_targets) in chain order.
    if len(chain_proba_cols) == 1:
        chain_probas = g.op.Identity(chain_proba_cols[0], name=f"{name}_chain_probas")
    else:
        chain_probas = g.op.Concat(
            *chain_proba_cols, axis=1, name=f"{name}_chain_probas_concat"
        )

    # Reorder to original column order.
    if is_identity_order:
        probabilities = g.op.Identity(
            chain_probas, name=f"{name}_proba", outputs=outputs[1:]
        )
    else:
        probabilities = g.op.Gather(
            chain_probas,
            inv_order,
            axis=1,
            name=f"{name}_proba",
            outputs=outputs[1:],
        )

    if not sts:
        g.set_type(probabilities, itype)
        if g.has_shape(X):
            batch_dim = g.get_shape(X)[0]
            g.set_shape(probabilities, (batch_dim, n_targets))
        elif g.has_rank(X):
            g.set_rank(probabilities, 2)

    return labels, probabilities
