from typing import Dict, List

import numpy as np
import pandas as pd
from category_encoders import QuantileEncoder

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


def _quantile_encode_column(
    g: GraphBuilderExtendedProtocol,
    col_tensor: str,
    known_vals: np.ndarray,
    known_q: np.ndarray,
    unknown_q: float,
    nan_q: float,
    dtype: np.dtype,
    name: str,
) -> str:
    """Emit ONNX nodes that apply quantile encoding to a single column tensor.

    For each input value the output is determined by the following priority:

    1. If the value is NaN → *nan_q*
    2. If the value matches a known category → the corresponding quantile value
    3. Otherwise (unknown category) → *unknown_q*

    :param g: graph builder
    :param col_tensor: ONNX tensor name for a single-column slice, shape ``(N, 1)``
    :param known_vals: 1-D numpy array of known category values (float)
    :param known_q: 1-D numpy array of corresponding quantile values (float)
    :param unknown_q: quantile value for unknown categories
    :param nan_q: quantile value for missing (NaN) inputs
    :param dtype: numpy dtype for all float constants
    :param name: node name prefix
    :return: ONNX tensor name, shape ``(N, 1)``
    """
    # Start with the default (unknown) value
    result = np.array([[unknown_q]], dtype=dtype)

    # For each known category, Where(Equal(col, cat), q_val, result)
    for i, (cat_val, q_val) in enumerate(zip(known_vals, known_q)):
        cat_const = np.array([[cat_val]], dtype=dtype)
        eq = g.op.Equal(col_tensor, cat_const, name=f"{name}_eq{i}")
        q_const = np.array([[q_val]], dtype=dtype)
        result = g.op.Where(eq, q_const, result, name=f"{name}_where{i}")

    # Handle NaN: IsNaN(col) → nan_q
    is_nan = g.op.IsNaN(col_tensor, name=f"{name}_isnan")
    nan_const = np.array([[nan_q]], dtype=dtype)
    result = g.op.Where(is_nan, nan_const, result, name=f"{name}_where_nan")

    return result


@register_sklearn_converter(QuantileEncoder)
def category_encoders_quantile_encoder(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: QuantileEncoder,
    X: str,
    name: str = "quantile_encoder",
) -> str:
    """
    Converts a :class:`category_encoders.QuantileEncoder` into ONNX.

    The encoder replaces each categorical column with the quantile of the
    target distribution conditioned on that category value.  Non-categorical
    columns pass through unchanged.

    .. code-block:: text

        X  ──col_j (categorical)──►  Equal(val_i)?──►  quantile_i
                                      ...
                                      IsNaN?──────────►  nan_quantile
                                      default──────────►  unknown_quantile

        X  ──col_k (numerical)──►  unchanged

    The conversion pre-computes a combined lookup table
    (original category value → quantile) from the fitted
    :attr:`~category_encoders.QuantileEncoder.ordinal_encoder` and
    :attr:`~category_encoders.QuantileEncoder.mapping` attributes.
    Unknown categories and NaN inputs are handled via separate
    ``Where`` nodes that override the default value.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``QuantileEncoder``
    :param outputs: desired output names
    :param X: input name (shape ``(N, F)``)
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(
        estimator, QuantileEncoder
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    # Build a mapping from feature name to column index.
    feat_idx = {feat: i for i, feat in enumerate(estimator.feature_names_in_)}

    # Build per-column lookup tables from the fitted encoder state.
    # ordinal_encoder.category_mapping: list of dicts with 'col' and 'mapping'
    # (orig_val → ordinal int).  estimator.mapping: dict col → Series
    # (ordinal int → quantile float).
    col_lookup: Dict[str, dict] = {}
    for col_info in estimator.ordinal_encoder.category_mapping:
        col = col_info["col"]
        ord_map = col_info["mapping"]  # pandas Series: orig_val → ordinal
        q_map = estimator.mapping[col]  # pandas Series: ordinal → quantile

        unknown_q = float(q_map[-1])
        nan_q = float(q_map[-2])

        known_vals_list = []
        known_q_list = []
        for orig_val, ordinal in ord_map.items():
            if pd.isna(orig_val):
                continue
            known_vals_list.append(float(orig_val))
            known_q_list.append(float(q_map[ordinal]))

        col_lookup[col] = {
            "known_vals": np.array(known_vals_list, dtype=dtype),
            "known_q": np.array(known_q_list, dtype=dtype),
            "unknown_q": unknown_q,
            "nan_q": nan_q,
        }

    n_features = len(estimator.feature_names_in_)
    cols_set = set(estimator.cols)

    # Build each output column tensor and collect them for a final Concat.
    col_tensors = []
    for feat_name in estimator.feature_names_in_:
        col_i = feat_idx[feat_name]
        # Extract column: shape (N, 1)
        col_slice = g.op.Gather(
            X,
            np.array([col_i], dtype=np.int64),
            axis=1,
            name=f"{name}_gather{col_i}",
        )

        if feat_name in cols_set:
            info = col_lookup[feat_name]
            col_out = _quantile_encode_column(
                g,
                col_slice,
                info["known_vals"],
                info["known_q"],
                info["unknown_q"],
                info["nan_q"],
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

    assert isinstance(res, str)
    if not sts:
        g.set_type_shape_unary_op(res, X)
    return res
