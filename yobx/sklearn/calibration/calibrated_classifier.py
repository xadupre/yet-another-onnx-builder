from typing import Dict, List, Tuple

import numpy as np
import onnx
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from ...typing import GraphBuilderExtendedProtocol
from ..register import get_sklearn_converter, register_sklearn_converter


def _apply_sigmoid_calibration(
    g: GraphBuilderExtendedProtocol, T: str, calibrator, dtype: np.dtype, name: str
) -> str:
    """
    Apply sigmoid calibration: ``expit(-(a * T + b))``.

    :class:`~sklearn.calibration._SigmoidCalibration` uses
    ``scipy.special.expit(-(a * T + b)) = 1 / (1 + exp(a * T + b))``,
    which is equivalent to ``sigmoid(-(a * T + b))``.

    :param g: graph builder
    :param T: name of input tensor of shape ``(N,)``
    :param calibrator: fitted ``_SigmoidCalibration`` with ``a_`` and ``b_``
    :param dtype: numpy float dtype matching the graph input type
    :param name: node name prefix
    :return: name of calibrated output tensor of shape ``(N,)``
    """
    a = np.array([calibrator.a_], dtype=dtype)
    b = np.array([calibrator.b_], dtype=dtype)
    aT = g.op.Mul(T, a, name=f"{name}_mul_a")
    aTb = g.op.Add(aT, b, name=f"{name}_add_b")
    neg_aTb = g.op.Neg(aTb, name=f"{name}_neg")
    return g.op.Sigmoid(neg_aTb, name=f"{name}_sig")


def _apply_isotonic_calibration(
    g: GraphBuilderExtendedProtocol,
    T: str,
    calibrator: IsotonicRegression,
    dtype: np.dtype,
    name: str,
) -> str:
    """
    Apply isotonic calibration via ``np.interp(T, X_thresholds_, y_thresholds_)``.

    Implements piecewise-linear interpolation between the knot points
    (``X_thresholds_``, ``y_thresholds_``) stored in the fitted
    :class:`~sklearn.isotonic.IsotonicRegression`.

    .. code-block:: text

        T (N,)
          ├──GreaterOrEqual(X_thresh (1,K))──► (N,K) bool
          │       Cast(INT64)──ReduceSum──► bin_count (N,)
          │           Sub(1)──Clip(0, K-2) ──► i0 (N,)
          │                   Add(1) ──────────► i1 (N,)
          │
          ├──Gather(X_thresh, i0)──► x0 (N,)
          ├──Gather(X_thresh, i1)──► x1 (N,)
          ├──Gather(Y_thresh, i0)──► y0 (N,)
          ├──Gather(Y_thresh, i1)──► y1 (N,)
          │
          └──y0 + (T - x0) * (y1 - y0) / (x1 - x0) ──Clip([y_min,y_max])──► (N,)

    :param g: graph builder
    :param T: name of input tensor of shape ``(N,)``
    :param calibrator: fitted :class:`~sklearn.isotonic.IsotonicRegression`
    :param dtype: numpy float dtype matching the graph input type
    :param name: node name prefix
    :return: name of calibrated output tensor of shape ``(N,)``
    """
    X_thresh = calibrator.X_thresholds_.astype(dtype)  # (K,) sorted ascending
    Y_thresh = calibrator.y_thresholds_.astype(dtype)  # (K,)
    K = len(X_thresh)

    if K == 1:
        # Degenerate: all inputs map to the single output value.
        t_zero = g.op.Mul(T, np.array([0], dtype=dtype), name=f"{name}_zero")
        return g.op.Add(t_zero, np.array([Y_thresh[0]], dtype=dtype), name=f"{name}_const")

    # Expand T to (N, 1) for broadcasting against X_thresh (1, K).
    T_2d = g.op.Unsqueeze(T, np.array([1], dtype=np.int64), name=f"{name}_unsq")
    X_arr = X_thresh.reshape(1, -1)  # (1, K) constant

    # Count the number of thresholds T is >= to: (N,) in [0, K].
    ge = g.op.GreaterOrEqual(T_2d, X_arr, name=f"{name}_ge")
    ge_int = g.op.Cast(ge, to=onnx.TensorProto.INT64, name=f"{name}_ge_int")
    bin_count = g.op.ReduceSum(
        ge_int, np.array([1], dtype=np.int64), keepdims=0, name=f"{name}_bincount"
    )  # (N,)

    # Lower-bound knot index: bin_count - 1, clipped to [0, K-2].
    i0_raw = g.op.Sub(bin_count, np.array([1], dtype=np.int64), name=f"{name}_i0raw")
    i0 = g.op.Clip(
        i0_raw,
        np.array([0], dtype=np.int64),
        np.array([K - 2], dtype=np.int64),
        name=f"{name}_i0",
    )  # (N,)
    i1 = g.op.Add(i0, np.array([1], dtype=np.int64), name=f"{name}_i1")  # (N,)

    # Gather knot values at the lower and upper bounds.
    x0 = g.op.Gather(X_thresh, i0, axis=0, name=f"{name}_x0")  # (N,)
    x1 = g.op.Gather(X_thresh, i1, axis=0, name=f"{name}_x1")  # (N,)
    y0 = g.op.Gather(Y_thresh, i0, axis=0, name=f"{name}_y0")  # (N,)
    y1 = g.op.Gather(Y_thresh, i1, axis=0, name=f"{name}_y1")  # (N,)

    # Linear interpolation: y0 + (T - x0) * (y1 - y0) / (x1 - x0).
    dx = g.op.Sub(x1, x0, name=f"{name}_dx")  # (N,)
    dy = g.op.Sub(y1, y0, name=f"{name}_dy")  # (N,)
    dt = g.op.Sub(T, x0, name=f"{name}_dt")  # (N,)

    # Guard against division by zero when x0 == x1 (flat step).
    zero_f = np.array([0], dtype=dtype)
    eps = np.array([1e-7], dtype=dtype)
    dx_safe = g.op.Where(
        g.op.Equal(dx, zero_f, name=f"{name}_dx_eq0"), eps, dx, name=f"{name}_dx_safe"
    )  # (N,)
    slope = g.op.Div(dy, dx_safe, name=f"{name}_slope")  # (N,)
    interp = g.op.Add(
        y0, g.op.Mul(slope, dt, name=f"{name}_slope_dt"), name=f"{name}_interp"
    )  # (N,)

    # Clip to [Y_thresh[0], Y_thresh[-1]] to match np.interp boundary behaviour.
    return g.op.Clip(
        interp,
        np.array([Y_thresh[0]], dtype=dtype),
        np.array([Y_thresh[-1]], dtype=dtype),
        name=f"{name}_clip",
    )


def _apply_calibrator(
    g: GraphBuilderExtendedProtocol, T: str, calibrator, dtype: np.dtype, name: str
) -> str:
    """
    Dispatch to the appropriate ONNX calibrator implementation.

    :param g: graph builder
    :param T: name of input tensor of shape ``(N,)``
    :param calibrator: fitted calibrator (``_SigmoidCalibration`` or
        :class:`~sklearn.isotonic.IsotonicRegression`)
    :param dtype: numpy float dtype matching the graph input type
    :param name: node name prefix
    :return: name of calibrated output tensor of shape ``(N,)``
    :raises NotImplementedError: for unsupported calibrator types
    """
    if hasattr(calibrator, "a_") and hasattr(calibrator, "b_"):
        return _apply_sigmoid_calibration(g, T, calibrator, dtype, name)
    if isinstance(calibrator, IsotonicRegression):
        return _apply_isotonic_calibration(g, T, calibrator, dtype, name)
    raise NotImplementedError(
        f"Calibrator of type {type(calibrator).__name__!r} is not supported. "
        "Supported calibrators: '_SigmoidCalibration' (sigmoid method) and "
        "'IsotonicRegression' (isotonic method)."
    )


def _get_base_predictions(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    base_est,
    X: str,
    itype: int,
    dtype: np.dtype,
    n_classes: int,
    name: str,
) -> Tuple[str, bool]:
    """
    Compute the base estimator's raw predictions that will be fed to the
    calibrators, matching sklearn's ``_get_response_values`` with
    ``response_method=["decision_function", "predict_proba"]``.

    * If the estimator has a **linear decision function** (``coef_`` +
      ``intercept_`` attributes), the decision function is computed via
      ``Gemm`` — this covers
      :class:`~sklearn.linear_model.LogisticRegression`,
      :class:`~sklearn.linear_model.LinearSVC`,
      :class:`~sklearn.linear_model.SGDClassifier`, etc.
    * Otherwise (e.g. :class:`~sklearn.ensemble.RandomForestClassifier`),
      the estimator's ``predict_proba`` output is used.

    :param g: graph builder
    :param sts: shapes/types mapping
    :param base_est: the fitted fold base estimator
    :param X: name of the input tensor
    :param itype: ONNX element type of the input
    :param dtype: numpy float dtype
    :param n_classes: number of classes in the outer classifier
    :param name: node name prefix
    :return: tuple ``(predictions_tensor_name, is_binary)``

        *  ``predictions_tensor_name`` — for binary: shape ``(N,)``; for
           multiclass: shape ``(N, C)``
        * ``is_binary`` — ``True`` when ``n_classes == 2``
    :raises NotImplementedError: when the estimator has a
        ``decision_function`` but no ``coef_`` / ``intercept_`` attributes
        (e.g. :class:`~sklearn.svm.SVC` without ``probability=True``,
        :class:`~sklearn.ensemble.GradientBoostingClassifier`)
    """
    is_binary = n_classes == 2

    if hasattr(base_est, "decision_function"):
        # sklearn prefers decision_function over predict_proba.
        if hasattr(base_est, "coef_") and hasattr(base_est, "intercept_"):
            # Linear model: decision = X @ coef_.T + intercept_
            coef = base_est.coef_.astype(dtype)
            intercept = base_est.intercept_.astype(dtype)
            decision = g.op.Gemm(X, coef, intercept, transB=1, name=f"{name}_decision")
            # Binary: Gemm output is (N, 1) → squeeze to (N,)
            if is_binary:
                decision = g.op.Reshape(
                    decision, np.array([-1], dtype=np.int64), name=f"{name}_decision_sq"
                )
            return decision, is_binary
        else:
            raise NotImplementedError(
                f"Base estimator {type(base_est).__name__!r} has "
                "decision_function but is not a linear model with coef_ / "
                "intercept_ attributes. Direct ONNX decision-function "
                "computation is currently only supported for linear models "
                "(LogisticRegression, LinearSVC, SGDClassifier, etc.). "
                "For estimators like SVC(probability=False) or "
                "GradientBoostingClassifier, wrap them in a pipeline or use "
                "a linear base estimator."
            )
    else:
        # Fall back to predict_proba (e.g. RandomForestClassifier).
        if not hasattr(base_est, "predict_proba"):
            raise NotImplementedError(
                f"Base estimator {type(base_est).__name__!r} does not expose "
                "predict_proba or a supported decision_function."
            )
        fct = get_sklearn_converter(type(base_est))
        sub_label = g.unique_name(f"{name}_label")
        sub_proba = g.unique_name(f"{name}_proba")
        fct(g, sts, [sub_label, sub_proba], base_est, X, name=name)
        if not g.has_type(sub_proba):
            g.set_type(sub_proba, itype)

        if is_binary:
            # sklearn uses only the positive-class column: predict_proba[:, 1]
            pos_prob = g.op.Gather(
                sub_proba, np.array(1, dtype=np.int64), axis=1, name=f"{name}_pos_prob"
            )  # (N,)
            return pos_prob, is_binary
        else:
            return sub_proba, is_binary


@register_sklearn_converter(CalibratedClassifierCV)
def sklearn_calibrated_classifier_cv(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: CalibratedClassifierCV,
    X: str,
    name: str = "calibrated_classifier",
) -> Tuple[str, str]:
    """
    Converts a :class:`sklearn.calibration.CalibratedClassifierCV` into ONNX.

    Both ``method='sigmoid'`` (Platt scaling) and ``method='isotonic'`` are
    supported, for binary and multiclass problems. Probabilities from all
    cross-validation folds are averaged to produce the final estimate.

    The raw predictions fed to the calibrators mirror sklearn's
    ``_get_response_values`` priority: **decision_function** is used when the
    base estimator has ``coef_`` and ``intercept_`` attributes (all linear
    models), and **predict_proba** is used otherwise (e.g.
    :class:`~sklearn.ensemble.RandomForestClassifier`).

    **Sigmoid calibration**:

    .. code-block:: text

        predictions (N,)  ──Mul(a)──Add(b)──Neg──Sigmoid──►  cal_prob (N,)

    **Isotonic calibration** — piecewise-linear mapping via
    :func:`numpy.interp`:

    .. code-block:: text

        T (N,)  ──GreaterOrEqual(X_thresh)──ReduceSum──► bin_idx
                                                              │
                 Gather(X_thresh, Y_thresh) ──► x0,x1,y0,y1 (N,)
                                                              │
                 y0 + (T-x0)*(y1-y0)/(x1-x0) ──Clip──►  cal (N,)

    **Binary** (two classes):

    .. code-block:: text

        X ──[base estimator]──► predictions (N,)  [decision or pos-class prob]
                                        │
                         [calibrator_0]──► cal_pos (N,)
                                        │
                      [1-cal_pos, cal_pos] ──► fold_proba (N,2)

    **Multiclass** (C classes):

    .. code-block:: text

        X ──[base estimator]──► predictions (N,C)  [decision or proba]
            predictions[:,k] ──[calibrator_k]──► cal_k (N,)   (k=0..C-1)
                          ──Concat──ReduceSum/Div──► fold_proba (N,C)

    Fold probabilities are averaged:

    .. code-block:: text

        fold_proba_0 (1,N,C)
        fold_proba_1 (1,N,C)  ──Concat(axis=0)──ReduceMean(axis=0)──► (N,C)
        ...
            ArgMax(axis=1)──Cast──Gather(classes_)──► label (N,)

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names (label, probabilities)
    :param estimator: a fitted
        :class:`~sklearn.calibration.CalibratedClassifierCV`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: tuple ``(label_result_name, proba_result_name)``
    :raises NotImplementedError: if the base estimator has a
        ``decision_function`` but is not a linear model with ``coef_`` /
        ``intercept_`` (e.g. :class:`~sklearn.svm.SVC` without
        ``probability=True``)
    """
    assert isinstance(
        estimator, CalibratedClassifierCV
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    classes = estimator.classes_
    n_classes = len(classes)

    fold_probas: List[str] = []

    for fold_idx, cal_clf in enumerate(estimator.calibrated_classifiers_):
        fold_name = f"{name}__fold{fold_idx}"
        base_est = cal_clf.estimator
        calibrators = cal_clf.calibrators

        # Get raw predictions from the base estimator (decision_function or
        # predict_proba), matching sklearn's _get_response_values priority.
        predictions, is_binary = _get_base_predictions(
            g, sts, base_est, X, itype, dtype, n_classes, fold_name
        )

        if is_binary:
            # predictions: (N,) — the single raw score / positive-class prob.
            cal_prob = _apply_calibrator(
                g, predictions, calibrators[0], dtype, f"{fold_name}_cal0"
            )  # (N,)

            # Reconstruct [1 - cal_prob, cal_prob] → (N, 2).
            one_minus = g.op.Sub(
                np.array([1], dtype=dtype), cal_prob, name=f"{fold_name}_one_minus"
            )
            cal_2d_0 = g.op.Unsqueeze(
                one_minus, np.array([1], dtype=np.int64), name=f"{fold_name}_unsq0"
            )  # (N, 1)
            cal_2d_1 = g.op.Unsqueeze(
                cal_prob, np.array([1], dtype=np.int64), name=f"{fold_name}_unsq1"
            )  # (N, 1)
            fold_proba = g.op.Concat(
                cal_2d_0, cal_2d_1, axis=1, name=f"{fold_name}_concat"
            )  # (N, 2)

        else:
            # predictions: (N, C) — one column per class.
            cal_cols: List[str] = []
            for k, calibrator in enumerate(calibrators):
                # Extract class-k score/probability column → (N,).
                pred_k = g.op.Gather(
                    predictions,
                    np.array(k, dtype=np.int64),
                    axis=1,
                    name=f"{fold_name}_pred_k{k}",
                )
                cal_k = _apply_calibrator(
                    g, pred_k, calibrator, dtype, f"{fold_name}_cal_k{k}"
                )  # (N,)
                cal_k_2d = g.op.Unsqueeze(
                    cal_k, np.array([1], dtype=np.int64), name=f"{fold_name}_unsq_k{k}"
                )  # (N, 1)
                cal_cols.append(cal_k_2d)

            fold_proba_unnorm = g.op.Concat(
                *cal_cols, axis=1, name=f"{fold_name}_concat_cal"
            )  # (N, C)
            row_sum = g.op.ReduceSum(
                fold_proba_unnorm,
                np.array([1], dtype=np.int64),
                keepdims=1,
                name=f"{fold_name}_row_sum",
            )  # (N, 1)
            fold_proba = g.op.Div(fold_proba_unnorm, row_sum, name=f"{fold_name}_norm")  # (N, C)

        # Unsqueeze to (1, N, C) for stacking across folds.
        fold_proba_3d = g.op.Unsqueeze(
            fold_proba, np.array([0], dtype=np.int64), name=f"{fold_name}_unsq_fold"
        )  # (1, N, C)
        fold_probas.append(fold_proba_3d)

    # Average calibrated probabilities across all folds.
    if len(fold_probas) == 1:
        avg_proba = g.op.Squeeze(
            fold_probas[0], np.array([0], dtype=np.int64), name=f"{name}_squeeze_single"
        )  # (N, C)
    else:
        stacked = g.op.Concat(*fold_probas, axis=0, name=f"{name}_stack_folds")  # (K, N, C)
        avg_proba = g.op.ReduceMean(
            stacked, np.array([0], dtype=np.int64), keepdims=0, name=f"{name}_mean_folds"
        )  # (N, C)

    # Derive predicted class labels from the averaged probabilities.
    label_idx_raw = g.op.ArgMax(avg_proba, axis=1, keepdims=0, name=f"{name}_argmax")
    label_idx = g.op.Cast(label_idx_raw, to=onnx.TensorProto.INT64, name=f"{name}_cast_idx")

    if np.issubdtype(classes.dtype, np.integer):
        classes_arr = classes.astype(np.int64)
        label = g.op.Gather(
            classes_arr, label_idx, axis=0, name=f"{name}_label", outputs=outputs[:1]
        )
        g.set_type(label, onnx.TensorProto.INT64)
    else:
        classes_arr = np.array(classes.astype(str))
        label = g.op.Gather(
            classes_arr, label_idx, axis=0, name=f"{name}_label_str", outputs=outputs[:1]
        )
        g.set_type(label, onnx.TensorProto.STRING)

    assert isinstance(label, str)
    proba = g.op.Identity(avg_proba, name=f"{name}_proba", outputs=outputs[1:])
    assert isinstance(proba, str)
    return label, proba
