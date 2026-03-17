from typing import Dict, List

import numpy as np
import onnx
from sklearn.multiclass import OutputCodeClassifier

try:
    from sklearn.multiclass import _ConstantPredictor as _CP
except ImportError:  # pragma: no cover
    _CP = None  # type: ignore[assignment,misc]

from ..register import register_sklearn_converter, get_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ..sklearn_helper import get_n_expected_outputs
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(OutputCodeClassifier)
def sklearn_output_code_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: OutputCodeClassifier,
    X: str,
    name: str = "output_code",
) -> str:
    """
    Converts a :class:`sklearn.multiclass.OutputCodeClassifier` into ONNX.

    The converter iterates over the fitted binary sub-estimators, calls the
    registered converter for each one to obtain per-class positive-class
    probabilities (``predict_proba[:, 1]``), stacks them into a score matrix
    ``Y`` of shape ``(N, M)`` (where *M* is the number of sub-estimators),
    and then finds the nearest row in ``code_book_`` using squared Euclidean
    distance.

    .. note::

        sklearn's :meth:`~sklearn.multiclass.OutputCodeClassifier.predict`
        uses :func:`~sklearn.multiclass._predict_binary`, which calls
        ``decision_function`` when available and falls back to
        ``predict_proba[:, 1]``.  This ONNX converter always uses
        ``predict_proba[:, 1]`` for all sub-estimators, matching sklearn
        exactly for classifiers that expose only ``predict_proba`` (such as
        :class:`~sklearn.tree.DecisionTreeClassifier`).  For classifiers with
        a ``decision_function`` (e.g. :class:`~sklearn.linear_model.LogisticRegression`),
        the ONNX output may differ from sklearn's prediction in borderline cases.

    **Graph structure:**

    .. code-block:: text

        X ──[sub-est 0]──► proba_0 (N,2) ──Slice[:,1]──► pred_0 (N,1) ──┐
        X ──[sub-est 1]──► proba_1 (N,2) ──Slice[:,1]──► pred_1 (N,1) ──┤ Concat
        ...                                                               │  axis=1
        X ──[sub-est M-1]─► proba_{M-1}  ──Slice[:,1]──► pred_{M-1}    ──┘
                                                                          │
                                                                         Y (N,M)
                                                                          │
        code_book_T (M,C) ──MatMul(Y) ──────────────────────────────► cross (N,C)
        y_sq (N,M) ──ReduceSum(axis=1,keepdims=1) ─────────────────► y_norms (N,1)
        cb_sq (1,C) ───────────────────────────────── Add(y_norms) ──► y_plus_cb (N,C)
                                 Sub(Mul(2, cross)) ──────────────► sq_dists (N,C)
                                                                          │
                                                         ArgMin(axis=1) ──► label_idx (N,)
                                                                          │
                              Gather(classes_, label_idx) ───────────► label (N,)

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names (label only; OutputCodeClassifier
        has no ``predict_proba``)
    :param estimator: a fitted :class:`~sklearn.multiclass.OutputCodeClassifier`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: label tensor name
    :raises NotImplementedError: when a sub-estimator does not expose
        :meth:`predict_proba`
    """
    assert isinstance(
        estimator, OutputCodeClassifier
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)
    classes = estimator.classes_

    # ------------------------------------------------------------------
    # Step 1: collect binary predictions (predict_proba[:, 1]) from each
    # sub-estimator → one [N, 1] tensor per sub-estimator.
    # ------------------------------------------------------------------
    binary_preds: List[str] = []
    for i, sub_est in enumerate(estimator.estimators_):
        sub_name = f"{name}__est{i}"

        # sklearn uses _ConstantPredictor internally when a binary split
        # has only one class; handle it without a registered converter.
        if _CP is not None and isinstance(sub_est, _CP):
            if sub_est.y_.size == 0:
                raise ValueError(
                    f"Sub-estimator {sub_name} is a _ConstantPredictor with empty y_."
                )
            y_val = float(sub_est.y_[0])
            const_val = np.array([[y_val]], dtype=dtype)  # (1, 1)
            # Expand (1, 1) → (N, 1) using the input batch dimension.
            x_shape = g.op.Shape(X, name=f"{sub_name}_x_shape")
            n_1d = g.op.Slice(
                x_shape,
                np.array([0], dtype=np.int64),
                np.array([1], dtype=np.int64),
                name=f"{sub_name}_n_1d",
            )
            shape_n1 = g.op.Concat(
                n_1d, np.array([1], dtype=np.int64), axis=0, name=f"{sub_name}_shape_n1"
            )
            pos_prob = g.op.Expand(const_val, shape_n1, name=f"{sub_name}_slice")
            binary_preds.append(pos_prob)
            continue

        n_sub_outputs = get_n_expected_outputs(sub_est)
        if n_sub_outputs < 2:
            raise NotImplementedError(
                f"Sub-estimator {type(sub_est).__name__} does not expose predict_proba. "
                "Only sub-estimators with predict_proba are supported."
            )

        sub_label = g.unique_name(f"{sub_name}_label")
        sub_proba = g.unique_name(f"{sub_name}_proba")
        fct = get_sklearn_converter(type(sub_est))
        fct(g, sts, [sub_label, sub_proba], sub_est, X, name=sub_name)

        # Some converters do not register the output type; fall back to FLOAT.
        if not g.has_type(sub_proba):
            g.set_type(sub_proba, onnx.TensorProto.FLOAT)

        # Extract column 1 (positive-class probability): [N, 2] → [N, 1].
        pos_prob = g.op.Slice(
            sub_proba,
            np.array([1], dtype=np.int64),  # starts
            np.array([2], dtype=np.int64),  # ends
            np.array([1], dtype=np.int64),  # axes
            name=f"{sub_name}_slice",
        )
        binary_preds.append(pos_prob)

    # ------------------------------------------------------------------
    # Step 2: stack predictions into Y: [N, n_estimators].
    # ------------------------------------------------------------------
    if len(binary_preds) == 1:
        Y = binary_preds[0]  # Already [N, 1]
    else:
        Y = g.op.Concat(*binary_preds, axis=1, name=f"{name}_Y")  # [N, M]

    # ------------------------------------------------------------------
    # Step 3: compute squared Euclidean distances to each row of
    # code_book_ using the identity:
    #   ||Y[i] - code_book_[c]||² = ||Y[i]||² + ||code_book_[c]||² - 2·Y[i]@code_book_[c]
    # ------------------------------------------------------------------
    code_book = estimator.code_book_.astype(dtype)  # [C, M]
    code_book_T = code_book.T  # [M, C]

    # Cross term: Y @ code_book_.T  →  [N, C]
    cross = g.op.MatMul(Y, code_book_T, name=f"{name}_cross")

    # ||Y[i]||²  →  [N, 1]
    y_sq = g.op.Mul(Y, Y, name=f"{name}_y_sq")
    y_norms = g.op.ReduceSum(
        y_sq, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_y_norms"
    )

    # ||code_book_[c]||²  →  constant [1, C]
    cb_sq = np.sum(code_book**2, axis=1, keepdims=True).T.astype(dtype)  # (1, C)

    # sq_dists = y_norms + cb_sq - 2 * cross  →  [N, C]
    two = np.array([2], dtype=dtype)
    two_cross = g.op.Mul(two, cross, name=f"{name}_two_cross")
    y_plus_cb = g.op.Add(y_norms, cb_sq, name=f"{name}_y_plus_cb")
    sq_dists = g.op.Sub(y_plus_cb, two_cross, name=f"{name}_sq_dists")

    # Clip to zero for numerical safety before argmin.
    zero = np.array([0], dtype=dtype)
    sq_dists_clipped = g.op.Max(sq_dists, zero, name=f"{name}_clip")

    # ------------------------------------------------------------------
    # Step 4: argmin → class label.
    # ------------------------------------------------------------------
    label_idx_raw = g.op.ArgMin(sq_dists_clipped, axis=1, keepdims=0, name=f"{name}_argmin")
    label_idx = g.op.Cast(label_idx_raw, to=onnx.TensorProto.INT64, name=f"{name}_cast")

    if np.issubdtype(classes.dtype, np.integer):
        classes_arr = classes.astype(np.int64)
        label = g.op.Gather(
            classes_arr, label_idx, axis=0, name=f"{name}_label", outputs=outputs[:1]
        )
        assert isinstance(label, str)
        if not sts:
            g.set_type(label, onnx.TensorProto.INT64)
    else:
        classes_arr = np.array(classes.astype(str))
        label = g.op.Gather(
            classes_arr, label_idx, axis=0, name=f"{name}_label_string", outputs=outputs[:1]
        )
        assert isinstance(label, str)
        if not sts:
            g.set_type(label, onnx.TensorProto.STRING)

    return label
