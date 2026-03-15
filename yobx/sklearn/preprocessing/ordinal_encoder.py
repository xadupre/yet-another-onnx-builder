import math
from typing import Dict, List

import numpy as np
import onnx

from sklearn.preprocessing import OrdinalEncoder

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(OrdinalEncoder)
def sklearn_ordinal_encoder(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: OrdinalEncoder,
    X: str,
    name: str = "ordinal_encoder",
) -> str:
    """
    Converts a :class:`sklearn.preprocessing.OrdinalEncoder` into ONNX.

    Each feature column is independently mapped from its category value to an
    integer ordinal (0-based position in the sorted :attr:`~sklearn.preprocessing.OrdinalEncoder.categories_`
    list).  Unknown categories and missing (``NaN``) inputs are handled via
    ``Where`` overrides.

    The conversion for a single feature *i* with categories
    ``[c_0, c_1, …, c_{K-1}]`` is:

    .. code-block:: text

        X ──Gather(col i)──► col_i  (N×1)
                                │
                Equal(col_i, [[c_0, …, c_{K-1}]])  ──► (N×K) bool
                                │
                        Cast(int64)  ──► (N×K) int64
                                │
              ┌─────────────────┴──────────────────┐
              │                                     │
        ArgMax(axis=1)                    ReduceMax(axis=1)
        (N×1) int64 ordinal           (N×1) int64  any-match flag
              │                                     │
        Cast(float)                       Cast(bool)
              └──── Where(any_match, ordinal, unk) ─┘
                                │
                        Where(IsNaN(col_i), nan_val, …)
                                │
                         feature_i_out  (N×1)

    All per-feature tensors are concatenated along ``axis=1``.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names
    :param estimator: a fitted :class:`~sklearn.preprocessing.OrdinalEncoder`
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added by this converter
    :return: name of the output tensor
    """
    assert isinstance(
        estimator, OrdinalEncoder
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    handle_unknown = estimator.handle_unknown
    unknown_value = getattr(estimator, "unknown_value", None)
    encoded_missing_value = getattr(estimator, "encoded_missing_value", np.nan)

    emv_is_nan = isinstance(encoded_missing_value, float) and math.isnan(encoded_missing_value)
    nan_val = np.array([[np.nan if emv_is_nan else float(encoded_missing_value)]], dtype=dtype)

    unk_is_nan = isinstance(unknown_value, float) and math.isnan(unknown_value) if unknown_value is not None else False
    if handle_unknown == "use_encoded_value":
        unk_val = np.array([[np.nan if unk_is_nan else float(unknown_value)]], dtype=dtype)
    else:
        unk_val = None  # unused when handle_unknown == 'error'

    col_tensors: List[str] = []
    for i, cats in enumerate(estimator.categories_):
        # ------------------------------------------------------------------
        # 1. Extract column i: shape (N, 1)
        # ------------------------------------------------------------------
        col_i = g.op.Gather(X, np.array([i], dtype=np.int64), axis=1, name=f"{name}_col{i}")

        # ------------------------------------------------------------------
        # 2. Compare col_i against every known category: shape (N, K) bool.
        #    Unknown values (including NaN) produce an all-False row.
        # ------------------------------------------------------------------
        cats_row = cats.astype(dtype).reshape(1, -1)  # (1, K)
        eq = g.op.Equal(col_i, cats_row, name=f"{name}_eq{i}")

        # ------------------------------------------------------------------
        # 3. Cast bool → int64 for ArgMax / ReduceMax
        # ------------------------------------------------------------------
        eq_int = g.op.Cast(eq, to=onnx.TensorProto.INT64, name=f"{name}_cast_int{i}")

        # ------------------------------------------------------------------
        # 4. Ordinal index via ArgMax: (N, 1) int64
        #    When there is no match the result is 0 (will be overridden below).
        # ------------------------------------------------------------------
        ordinal_i64 = g.op.ArgMax(eq_int, axis=1, keepdims=1, name=f"{name}_argmax{i}")

        # Cast ordinal to the output float dtype.
        ordinal = g.op.Cast(ordinal_i64, to=int(itype), name=f"{name}_cast_ord{i}")

        # ------------------------------------------------------------------
        # 5. Detect whether any category matched: ReduceMax → (N, 1) int64.
        #    0 means no match, 1 means matched.
        # ------------------------------------------------------------------
        any_match_i64 = g.op.ReduceMax(
            eq_int, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_any{i}"
        )
        any_match_bool = g.op.Cast(
            any_match_i64, to=onnx.TensorProto.BOOL, name=f"{name}_match{i}"
        )

        # ------------------------------------------------------------------
        # 6. Override non-matching rows with unknown_value when applicable.
        # ------------------------------------------------------------------
        if handle_unknown == "use_encoded_value":
            ordinal = g.op.Where(any_match_bool, ordinal, unk_val, name=f"{name}_unk{i}")

        # ------------------------------------------------------------------
        # 7. Override NaN inputs with encoded_missing_value.
        # ------------------------------------------------------------------
        is_nan = g.op.IsNaN(col_i, name=f"{name}_isnan{i}")
        ordinal = g.op.Where(is_nan, nan_val, ordinal, name=f"{name}_nan{i}")

        col_tensors.append(ordinal)

    n_features = len(col_tensors)
    if n_features == 1:
        res = g.op.Identity(col_tensors[0], name=name, outputs=outputs)
    else:
        res = g.op.Concat(*col_tensors, axis=1, name=name, outputs=outputs)

    assert isinstance(res, str)
    if not sts:
        g.set_type_shape_unary_op(res, X)
    return res
