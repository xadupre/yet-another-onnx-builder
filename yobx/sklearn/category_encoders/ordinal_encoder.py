from typing import Dict, List

import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(OrdinalEncoder)
def category_encoders_ordinal_encoder(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: OrdinalEncoder,
    X: str,
    name: str = "ce_ordinal_encoder",
) -> str:
    """
    Converts a :class:`category_encoders.OrdinalEncoder` into ONNX.

    The encoder replaces each categorical column with an integer ordinal
    (1-based position of the value in the fitted category list).
    Non-categorical columns pass through unchanged.

    .. code-block:: text

        X  ──col_j (categorical)──►  Equal(val_i)?──►  ordinal_i
                                      ...
                                      IsNaN?──────────►  nan_ordinal  (or NaN)
                                      default──────────►  unknown_ordinal  (or NaN)

        X  ──col_k (numerical)──►  unchanged

    The conversion pre-computes an (original category value → ordinal) lookup
    from the fitted :attr:`~category_encoders.OrdinalEncoder.mapping` attribute.

    **Unknown categories** and **missing values** are handled according to the
    fitted ``handle_unknown`` and ``handle_missing`` parameters:

    * ``'value'`` – produces ``-1`` for unknown categories and ``-2`` for
      missing values (the encoder's default numeric sentinel values).
    * ``'return_nan'`` – produces ``NaN`` for the corresponding input.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted :class:`~category_encoders.OrdinalEncoder`
    :param outputs: desired output names
    :param X: input name (shape ``(N, F)``)
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(
        estimator, OrdinalEncoder
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"
    assert (
        estimator.mapping is not None
    ), f"estimator {estimator} is not fitted{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    handle_unknown = getattr(estimator, "handle_unknown", "value")
    handle_missing = getattr(estimator, "handle_missing", "value")

    unknown_is_nan = handle_unknown == "return_nan"
    missing_is_nan = handle_missing == "return_nan"

    # Build a mapping from feature name to column index.
    feat_idx = {feat: i for i, feat in enumerate(estimator.feature_names_in_)}
    cols_set = set(estimator.cols)

    # Build per-column lookup tables from the fitted mapping attribute.
    # mapping: list of dicts with 'col' and 'mapping'
    # (Series: orig_val → ordinal int, NaN → -2).
    col_lookup: Dict[str, dict] = {}
    for col_info in estimator.mapping:
        col = col_info["col"]
        ord_map = col_info["mapping"]  # pandas Series: orig_val → ordinal int

        known_vals_list = []
        known_ord_list = []
        for orig_val, ordinal in ord_map.items():
            if pd.isna(orig_val):
                continue
            known_vals_list.append(float(orig_val))
            known_ord_list.append(float(ordinal))

        col_lookup[col] = {
            "known_vals": np.array(known_vals_list, dtype=dtype),
            "known_ord": np.array(known_ord_list, dtype=dtype),
        }

    n_features = len(estimator.feature_names_in_)

    # Build each output column tensor and collect them for a final Concat.
    col_tensors = []
    for feat_name in estimator.feature_names_in_:
        col_i = feat_idx[feat_name]
        # Extract column: shape (N, 1)
        col_slice = g.op.Gather(
            X, np.array([col_i], dtype=np.int64), axis=1, name=f"{name}_gather{col_i}"
        )

        if feat_name in cols_set:
            info = col_lookup[feat_name]
            col_out = _ordinal_encode_column(
                g,
                col_slice,
                info["known_vals"],
                info["known_ord"],
                unknown_is_nan,
                missing_is_nan,
                dtype,
                f"{name}_col{col_i}",
            )
        else:
            # Non-categorical column: pass through as-is.
            col_out = col_slice

        col_tensors.append(col_out)

    if n_features == 1:
        res = g.op.Identity(col_tensors[0], name=name, outputs=outputs)
    else:
        res = g.op.Concat(*col_tensors, axis=1, name=name, outputs=outputs)

    g.set_type_shape_unary_op(res, X)
    return res


def _ordinal_encode_column(
    g: GraphBuilderExtendedProtocol,
    col_tensor: str,
    known_vals: np.ndarray,
    known_ord: np.ndarray,
    unknown_is_nan: bool,
    missing_is_nan: bool,
    dtype: np.dtype,
    name: str,
) -> str:
    """Emit ONNX nodes that apply ordinal encoding to a single column tensor.

    For each input value the output is determined by the following priority:

    1. If the value is NaN → ``nan_ordinal`` (``-2.0`` or ``NaN`` depending on
       ``missing_is_nan``)
    2. If the value matches a known category → the corresponding ordinal integer
    3. Otherwise (unknown category) → ``unknown_ordinal`` (``-1.0`` or ``NaN``
       depending on ``unknown_is_nan``)

    :param g: graph builder
    :param col_tensor: ONNX tensor name for a single-column slice, shape ``(N, 1)``
    :param known_vals: 1-D numpy array of known category values (float)
    :param known_ord: 1-D numpy array of corresponding ordinal integers (float)
    :param unknown_is_nan: if ``True``, unknown categories produce ``NaN``
    :param missing_is_nan: if ``True``, missing (NaN) inputs produce ``NaN``
    :param dtype: numpy dtype for all float constants
    :param name: node name prefix
    :return: ONNX tensor name, shape ``(N, 1)``
    """
    unknown_val = float("nan") if unknown_is_nan else -1.0
    # Start with the default (unknown) value
    result = np.array([[unknown_val]], dtype=dtype)

    # For each known category, Where(Equal(col, cat), ordinal, result)
    for i, (cat_val, ord_val) in enumerate(zip(known_vals, known_ord)):
        cat_const = np.array([[cat_val]], dtype=dtype)
        eq = g.op.Equal(col_tensor, cat_const, name=f"{name}_eq{i}")
        ord_const = np.array([[ord_val]], dtype=dtype)
        result = g.op.Where(eq, ord_const, result, name=f"{name}_where{i}")

    # Handle NaN inputs: IsNaN(col) → nan_val
    nan_val = float("nan") if missing_is_nan else -2.0
    is_nan = g.op.IsNaN(col_tensor, name=f"{name}_isnan")
    nan_const = np.array([[nan_val]], dtype=dtype)
    result = g.op.Where(is_nan, nan_const, result, name=f"{name}_where_nan")

    return result
