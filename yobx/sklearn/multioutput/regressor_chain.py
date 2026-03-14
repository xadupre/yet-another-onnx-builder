from typing import Dict, List

import numpy as np
from sklearn.multioutput import RegressorChain

from ..register import register_sklearn_converter, get_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol


@register_sklearn_converter(RegressorChain)
def sklearn_regressor_chain(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: RegressorChain,
    X: str,
    name: str = "regressor_chain",
) -> str:
    """
    Converts a :class:`sklearn.multioutput.RegressorChain` into ONNX.

    The converter iterates over the fitted sub-estimators in chain order.
    Each sub-estimator ``i`` receives an augmented feature matrix formed by
    concatenating the original input ``X`` with all previous predictions along
    axis 1.  The prediction of every sub-estimator is reshaped from ``(N,)``
    to ``(N, 1)`` before being appended to the running feature matrix.

    After all sub-estimators have been applied the per-chain predictions are
    concatenated into a ``(N, n_targets)`` tensor in chain order.  When
    ``estimator.order_`` is not the identity permutation a final ``Gather``
    node reorders the columns to match the original target order.

    Graph structure (default order, 3 targets):

    .. code-block:: text

        X ───────────────────── [est 0] ──► pred_0 (N,) ──Reshape(N,1)──► p0
        Concat(X, p0) ────────── [est 1] ──► pred_1 (N,) ──Reshape(N,1)──► p1
        Concat(X, p0, p1) ────── [est 2] ──► pred_2 (N,) ──Reshape(N,1)──► p2

                    Concat(p0, p1, p2, axis=1) ──► chain_preds (N, n_targets)
                                                              │
                      Gather(chain_preds, inv_order, axis=1) ──► predictions (N, n_targets)

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names
    :param estimator: a fitted :class:`~sklearn.multioutput.RegressorChain`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: name of the output tensor of shape ``(N, n_targets)``
    """
    assert isinstance(
        estimator, RegressorChain
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    n_targets = len(estimator.estimators_)

    # per_chain_preds[i] holds the (N, 1) prediction for chain position i
    per_chain_preds: List[str] = []

    for i, sub_est in enumerate(estimator.estimators_):
        sub_name = f"{name}__est{i}"

        # Build augmented input: [X, pred_0, ..., pred_{i-1}] along axis=1
        if i == 0:
            X_aug = X
        else:
            X_aug = g.op.Concat(X, *per_chain_preds, axis=1, name=f"{sub_name}_aug")
            if not g.has_type(X_aug):
                g.set_type(X_aug, itype)

        # Run the sub-estimator converter
        sub_output = g.unique_name(f"{sub_name}_pred")
        fct = get_sklearn_converter(type(sub_est))
        fct(g, sts, [sub_output], sub_est, X_aug, name=sub_name)

        if not g.has_type(sub_output):
            g.set_type(sub_output, itype)

        # Reshape (N,) or (N, 1) → (N, 1) for feature augmentation
        reshaped = g.op.Reshape(
            sub_output,
            np.array([-1, 1], dtype=np.int64),
            name=f"{sub_name}_reshape",
        )
        per_chain_preds.append(reshaped)

    # Concatenate all (N, 1) chain predictions → (N, n_targets) in chain order
    if len(per_chain_preds) == 1:
        chain_preds = g.op.Identity(per_chain_preds[0], name=f"{name}_chain")
    else:
        chain_preds = g.op.Concat(
            *per_chain_preds, axis=1, name=f"{name}_chain_concat"
        )

    if not g.has_type(chain_preds):
        g.set_type(chain_preds, itype)

    # Reorder columns from chain order to original target order using inv_order.
    # sklearn: inv_order[order_[i]] = i  →  Y_output = Y_chain[:, inv_order]
    order = np.array(estimator.order_, dtype=np.int64)
    inv_order = np.empty(n_targets, dtype=np.int64)
    inv_order[order] = np.arange(n_targets, dtype=np.int64)

    is_identity = np.array_equal(inv_order, np.arange(n_targets, dtype=np.int64))
    if is_identity:
        predictions = g.op.Identity(chain_preds, name=f"{name}_pred", outputs=outputs[:1])
    else:
        predictions = g.op.Gather(
            chain_preds,
            inv_order,
            axis=1,
            name=f"{name}_reorder",
            outputs=outputs[:1],
        )

    if not sts:
        g.set_type(predictions, itype)
        if g.has_shape(X):
            batch_dim = g.get_shape(X)[0]
            g.set_shape(predictions, (batch_dim, n_targets))
        elif g.has_rank(X):
            g.set_rank(predictions, 2)

    return predictions
