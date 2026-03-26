from typing import Dict, List

import numpy as np
import onnx
import pandas as pd
from category_encoders import PolynomialEncoder

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


def _polynomial_encode_column(
    g: GraphBuilderExtendedProtocol,
    col_tensor: str,
    known_vals: np.ndarray,
    known_contrast: np.ndarray,
    unknown_contrast: np.ndarray,
    nan_contrast: np.ndarray,
    dtype: np.dtype,
    name: str,
    is_string: bool = False,
) -> List[str]:
    """Emit ONNX nodes that apply polynomial (contrast) encoding to a single column tensor.

    For each input value the output row is determined by the following priority:

    1. If the value is NaN → *nan_contrast* (skipped for string inputs)
    2. If the value matches a known category → the corresponding contrast row
    3. Otherwise (unknown category) → *unknown_contrast*

    :param g: graph builder
    :param col_tensor: ONNX tensor name for a single-column slice, shape ``(N, 1)``
    :param known_vals: 1-D numpy array of known original category values; dtype
        ``object`` for string inputs, float otherwise
    :param known_contrast: 2-D numpy array of contrast values, shape ``(n_known, n_contrasts)``
    :param unknown_contrast: 1-D numpy array of values for unknown categories,
        shape ``(n_contrasts,)``
    :param nan_contrast: 1-D numpy array of values for NaN inputs, shape ``(n_contrasts,)``
    :param dtype: numpy dtype for output (float) constants
    :param name: node name prefix
    :param is_string: if ``True``, skip ``IsNaN`` handling (strings cannot be NaN)
    :return: list of ONNX tensor names, each shape ``(N, 1)``, one per contrast column
    """
    n_contrasts = known_contrast.shape[1]

    # Pre-compute Equal masks for all known categories (shared across contrasts)
    eq_masks = []
    for i, cat_val in enumerate(known_vals):
        cat_const = np.array([[cat_val]], dtype=known_vals.dtype)
        eq = g.op.Equal(col_tensor, cat_const, name=f"{name}_eq{i}")
        eq_masks.append(eq)

    # Pre-compute IsNaN mask (shared across contrasts; skipped for string inputs)
    if not is_string:
        is_nan = g.op.IsNaN(col_tensor, name=f"{name}_isnan")

    results = []
    for j in range(n_contrasts):
        # Start with the default (unknown) value
        result = np.array([[unknown_contrast[j]]], dtype=dtype)

        # For each known category, Where(Equal(col, cat), contrast_j, result)
        for i, (eq_mask, contrast_val) in enumerate(zip(eq_masks, known_contrast[:, j])):
            c_const = np.array([[contrast_val]], dtype=dtype)
            result = g.op.Where(eq_mask, c_const, result, name=f"{name}_where{i}_c{j}")

        if not is_string:
            # Handle NaN: IsNaN(col) → nan_contrast_j (not applicable to string tensors)
            nan_const = np.array([[nan_contrast[j]]], dtype=dtype)
            result = g.op.Where(is_nan, nan_const, result, name=f"{name}_where_nan_c{j}")

        results.append(result)

    return results


@register_sklearn_converter(PolynomialEncoder)
def category_encoders_polynomial_encoder(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: PolynomialEncoder,
    X: str,
    name: str = "polynomial_encoder",
) -> str:
    """
    Converts a :class:`category_encoders.PolynomialEncoder` into ONNX.

    The encoder replaces each categorical column with polynomial contrast
    coding columns.  The number of output columns per categorical feature is
    ``n_categories - 1`` (one column is dropped to avoid perfect
    multicollinearity).  Non-categorical columns pass through unchanged.

    .. code-block:: text

        X  ──col_j (categorical)──►  Equal(val_i)?──►  contrast_i_0
                                      ...              contrast_i_1
                                      IsNaN?──────────►  nan_contrast_0
                                                         nan_contrast_1
                                      default──────────►  unknown_contrast_0
                                                          unknown_contrast_1

        X  ──col_k (numerical)──►  unchanged

    The conversion pre-computes a combined lookup table
    (original category value → contrast row) from the fitted
    :attr:`~category_encoders.PolynomialEncoder.ordinal_encoder` and
    :attr:`~category_encoders.PolynomialEncoder.mapping` attributes.
    Unknown categories and NaN inputs are handled via separate
    ``Where`` nodes that override the default value.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``PolynomialEncoder``
    :param outputs: desired output names
    :param X: input name (shape ``(N, F)``)
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(
        estimator, PolynomialEncoder
    ), f"Unexpected type {type(estimator)} for estimator."
    assert estimator.mapping is not None, f"estimator properly set up, {estimator=}"
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    is_string = itype == onnx.TensorProto.STRING
    # For string inputs the output is always float32 (contrast values are numeric).
    out_itype = onnx.TensorProto.FLOAT if is_string else itype
    out_dtype = np.float32 if is_string else dtype

    # Build a mapping from feature name to column index.
    feat_idx = {feat: i for i, feat in enumerate(estimator.feature_names_in_)}

    # Build per-column lookup tables from the fitted encoder state.
    # ordinal_encoder.category_mapping: list of dicts with 'col' and 'mapping'
    # (orig_val → ordinal int).  estimator.mapping: list of dicts with 'col'
    # and 'mapping' DataFrame (ordinal int → contrast row).
    col_lookup: Dict[str, dict] = {}
    for ord_info in estimator.ordinal_encoder.category_mapping:
        col = ord_info["col"]
        ord_map = ord_info["mapping"]  # pandas Series: orig_val → ordinal

        # Find the corresponding contrast matrix in estimator.mapping
        contrast_df = None
        for m in estimator.mapping:
            if m["col"] == col:
                contrast_df = m["mapping"]  # DataFrame: ordinal → contrast row
                break
        assert contrast_df is not None, f"No mapping found for column {col!r}"

        # unknown_contrast is the row indexed by -1 (unknown category ordinal)
        unknown_contrast = contrast_df.loc[-1].values.astype(out_dtype)
        # nan_contrast is the row indexed by -2 (NaN ordinal)
        nan_contrast = contrast_df.loc[-2].values.astype(out_dtype)

        # Build lookup for all known (non-NaN) original values
        known_vals_list = []
        known_contrast_rows = []
        for orig_val, ordinal in ord_map.items():
            if pd.isna(orig_val):
                continue
            known_vals_list.append(str(orig_val) if is_string else float(orig_val))
            known_contrast_rows.append(contrast_df.loc[ordinal].values.astype(out_dtype))

        col_lookup[col] = {
            "known_vals": np.array(known_vals_list, dtype=dtype),
            "known_contrast": (
                np.stack(known_contrast_rows, axis=0)
                if known_contrast_rows
                else np.zeros((0, len(unknown_contrast)), dtype=out_dtype)
            ),
            "unknown_contrast": unknown_contrast,
            "nan_contrast": nan_contrast,
            "out_cols": contrast_df.columns.tolist(),
        }

    cols_set = set(estimator.cols)

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
            contrast_tensors = _polynomial_encode_column(
                g,
                col_slice,
                info["known_vals"],
                info["known_contrast"],
                info["unknown_contrast"],
                info["nan_contrast"],
                out_dtype,
                f"{name}_col{col_i}",
                is_string=is_string,
            )
            col_tensors.extend(contrast_tensors)
        else:
            # Non-categorical column: pass through as-is.
            col_tensors.append(col_slice)

    n_out = len(col_tensors)
    if n_out == 1:
        res = g.op.Identity(col_tensors[0], name=name, outputs=outputs)
    else:
        res = g.op.Concat(*col_tensors, axis=1, name=name, outputs=outputs)

    g.set_type(res, out_itype)
    if g.has_shape(X):
        shape = g.get_shape(X)
        g.set_shape(res, (shape[0], n_out))
    elif g.has_rank(X):
        g.set_rank(res, 2)
    return res
