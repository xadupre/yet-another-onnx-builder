from typing import Tuple, Dict, List, Union

import numpy as np
import onnx
from sklearn.ensemble import StackingClassifier, StackingRegressor

from ..register import register_sklearn_converter, get_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ..sklearn_helper import get_n_expected_outputs


@register_sklearn_converter(StackingRegressor)
def sklearn_stacking_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: StackingRegressor,
    X: str,
    name: str = "stacking_regressor",
) -> str:
    """
    Converts a :class:`sklearn.ensemble.StackingRegressor` into ONNX.

    For each base estimator, the converter calls the registered converter using
    the ``predict`` stack method, reshapes the 1-D predictions to ``(N, 1)``,
    and concatenates them into a meta-feature matrix of shape
    ``(N, n_estimators)``.  When ``passthrough=True`` the original features are
    appended before the final estimator is applied.

    Graph structure::

        X ──[est 0 converter]──► pred_0 (N,) ──Reshape(N,1)──┐
        X ──[est 1 converter]──► pred_1 (N,) ──Reshape(N,1)──┤
        ...                                                    │
                                              Concat(axis=1) ─►meta (N, n_est)
                                                               │
                                   [passthrough X concat] ─────┤ (optional)
                                                               │
                                       [final estimator] ──────► predictions

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names
    :param estimator: a fitted :class:`~sklearn.ensemble.StackingRegressor`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: name of the output tensor
    """
    assert isinstance(estimator, StackingRegressor)
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)

    meta_features = _collect_meta_features_regressor(g, sts, estimator, X, name, itype)

    if estimator.passthrough:
        meta_features = g.op.Concat(meta_features, X, axis=1, name=f"{name}_passthrough")

    fct = get_sklearn_converter(type(estimator.final_estimator_))
    fct(g, sts, outputs, estimator.final_estimator_, meta_features, name=f"{name}__final")

    return outputs[0]


@register_sklearn_converter(StackingClassifier)
def sklearn_stacking_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: StackingClassifier,
    X: str,
    name: str = "stacking_classifier",
) -> Union[str, Tuple[str, str]]:
    """
    Converts a :class:`sklearn.ensemble.StackingClassifier` into ONNX.

    For each base estimator, the converter dispatches on the ``stack_method_``
    attribute to collect meta-features:

    * ``predict_proba`` — binary problem (``len(classes_) == 2``): only the
      positive-class column ``proba[:, 1:]`` is kept, giving shape ``(N, 1)``;
      multiclass: all probability columns are kept.
    * ``predict`` — the 1-D label output is reshaped to ``(N, 1)`` and cast to
      the input float dtype.

    Graph structure (binary example)::

        X ──[est 0 converter]──► proba_0 (N,2) ──Slice[:,1:]──► (N,1)──┐
        X ──[est 1 converter]──► proba_1 (N,2) ──Slice[:,1:]──► (N,1)──┤
        ...                                                              │
                                                    Concat(axis=1) ─────► meta (N, n_est)
                                                                         │
                                          [passthrough X concat] ────────┤ (optional)
                                                                         │
                                              [final estimator] ─────────► label [, proba]

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names (label, or label + probabilities)
    :param estimator: a fitted :class:`~sklearn.ensemble.StackingClassifier`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: label tensor name, or tuple ``(label, probabilities)``
    :raises NotImplementedError: when a ``stack_method_`` entry other than
        ``'predict_proba'`` or ``'predict'`` is encountered
    """
    assert isinstance(estimator, StackingClassifier)
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)

    meta_features = _collect_meta_features_classifier(g, sts, estimator, X, name, itype)

    if estimator.passthrough:
        meta_features = g.op.Concat(meta_features, X, axis=1, name=f"{name}_passthrough")

    fct = get_sklearn_converter(type(estimator.final_estimator_))
    fct(g, sts, outputs, estimator.final_estimator_, meta_features, name=f"{name}__final")

    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs[:2])


def _collect_meta_features_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    estimator: StackingRegressor,
    X: str,
    name: str,
    itype: int,
) -> str:
    """
    Build the meta-feature matrix for a :class:`~sklearn.ensemble.StackingRegressor`.

    Each base estimator is converted via its registered ``predict`` converter
    and its 1-D output is reshaped to ``(N, 1)``.  The results are concatenated
    along axis 1.
    """
    per_estimator_preds: List[str] = []
    for i, est in enumerate(estimator.estimators_):
        est_name = f"{name}__est{i}"
        est_output = g.unique_name(f"{est_name}_pred")
        fct = get_sklearn_converter(type(est))
        fct(g, sts, [est_output], est, X, name=est_name)

        if not g.has_type(est_output):
            g.set_type(est_output, itype)

        # Predictions are 1-D (N,) → reshape to (N, 1).
        reshaped = g.op.Reshape(
            est_output,
            np.array([-1, 1], dtype=np.int64),
            name=f"{est_name}_reshape",
        )
        per_estimator_preds.append(reshaped)

    if len(per_estimator_preds) == 1:
        return per_estimator_preds[0]
    return g.op.Concat(*per_estimator_preds, axis=1, name=f"{name}_concat_meta")


def _collect_meta_features_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    estimator: StackingClassifier,
    X: str,
    name: str,
    itype: int,
) -> str:
    """
    Build the meta-feature matrix for a :class:`~sklearn.ensemble.StackingClassifier`.

    Mirrors the logic of
    :meth:`sklearn.ensemble.StackingClassifier._concatenate_predictions`.
    """
    binary = len(estimator.classes_) == 2

    per_estimator_preds: List[str] = []
    for i, (est, method) in enumerate(zip(estimator.estimators_, estimator.stack_method_)):
        est_name = f"{name}__est{i}"

        if method == "predict_proba":
            # Obtain (label, proba) from the sub-estimator converter.
            sub_label = g.unique_name(f"{est_name}_label")
            sub_proba = g.unique_name(f"{est_name}_proba")
            fct = get_sklearn_converter(type(est))
            fct(g, sts, [sub_label, sub_proba], est, X, name=est_name)

            if not g.has_type(sub_proba):
                g.set_type(sub_proba, onnx.TensorProto.FLOAT)

            if binary:
                # Binary stacking: drop the first (redundant) column.
                # preds[:, 1:] keeps shape (N, 1).
                meta = g.op.Slice(
                    sub_proba,
                    np.array([1], dtype=np.int64),  # starts
                    np.array([2], dtype=np.int64),  # ends
                    np.array([1], dtype=np.int64),  # axes
                    name=f"{est_name}_slice",
                )
            else:
                # Multiclass: keep all probability columns.
                meta = sub_proba

            per_estimator_preds.append(meta)

        elif method == "predict":
            # Use the label / scalar prediction output from the converter.
            n_outputs = get_n_expected_outputs(est)
            if n_outputs >= 2:
                sub_label = g.unique_name(f"{est_name}_label")
                sub_proba = g.unique_name(f"{est_name}_proba")
                fct = get_sklearn_converter(type(est))
                fct(g, sts, [sub_label, sub_proba], est, X, name=est_name)
                est_output = sub_label
            else:
                est_output = g.unique_name(f"{est_name}_pred")
                fct = get_sklearn_converter(type(est))
                fct(g, sts, [est_output], est, X, name=est_name)

            if not g.has_type(est_output):
                g.set_type(est_output, itype)

            # Cast labels/predictions to the input float dtype so they are
            # compatible with probability-based meta-features.
            casted = g.op.Cast(
                est_output,
                to=itype,
                name=f"{est_name}_cast",
            )
            # Reshape 1-D (N,) → (N, 1).
            reshaped = g.op.Reshape(
                casted,
                np.array([-1, 1], dtype=np.int64),
                name=f"{est_name}_reshape",
            )
            per_estimator_preds.append(reshaped)

        else:
            raise NotImplementedError(
                f"stack_method {method!r} for estimator {type(est).__name__!r} is not "
                "supported. Supported methods are: 'predict_proba' and 'predict'."
            )

    if len(per_estimator_preds) == 1:
        return per_estimator_preds[0]
    return g.op.Concat(*per_estimator_preds, axis=1, name=f"{name}_concat_meta")
