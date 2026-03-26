from typing import Dict, List

import numpy as np
import onnx
import pandas as pd
from category_encoders import TargetEncoder

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


def _target_encode_column(
    g: GraphBuilderExtendedProtocol,
    col_tensor: str,
    known_vals: np.ndarray,
    known_target: np.ndarray,
    unknown_target: float,
    nan_target: float,
    dtype: np.dtype,
    name: str,
    is_string: bool = False,
) -> str:
    """Emit ONNX nodes that apply target encoding to a single column tensor.

    For each input value the output is determined by the following priority:

    1. If the value is NaN → *nan_target* (skipped for string inputs)
    2. If the value matches a known category → the corresponding target mean
    3. Otherwise (unknown category) → *unknown_target*

    :param g: graph builder
    :param col_tensor: ONNX tensor name for a single-column slice, shape ``(N, 1)``
    :param known_vals: 1-D numpy array of known category values; dtype ``object``
        for string inputs, float otherwise
    :param known_target: 1-D numpy array of corresponding target mean values (float)
    :param unknown_target: target mean for unknown categories
    :param nan_target: target mean for missing (NaN) inputs
    :param dtype: numpy dtype for output (float) constants
    :param name: node name prefix
    :param is_string: if ``True``, skip ``IsNaN`` handling (strings cannot be NaN)
    :return: ONNX tensor name, shape ``(N, 1)``
    """
    # Start with the default (unknown) value
    result = np.array([[unknown_target]], dtype=dtype)

    # For each known category, Where(Equal(col, cat), target_val, result)
    for i, (cat_val, target_val) in enumerate(zip(known_vals, known_target)):
        cat_const = np.array([[cat_val]], dtype=known_vals.dtype)
        eq = g.op.Equal(col_tensor, cat_const, name=f"{name}_eq{i}")
        target_const = np.array([[target_val]], dtype=dtype)
        result = g.op.Where(eq, target_const, result, name=f"{name}_where{i}")

    if not is_string:
        # Handle NaN: IsNaN(col) → nan_target (not applicable to string tensors)
        is_nan = g.op.IsNaN(col_tensor, name=f"{name}_isnan")
        nan_const = np.array([[nan_target]], dtype=dtype)
        result = g.op.Where(is_nan, nan_const, result, name=f"{name}_where_nan")

    return result


@register_sklearn_converter(TargetEncoder)
def category_encoders_target_encoder(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: TargetEncoder,
    X: str,
    name: str = "target_encoder",
) -> str:
    """
    Converts a :class:`category_encoders.TargetEncoder` into ONNX.

    The encoder replaces each categorical column with the smoothed mean of the
    target distribution conditioned on that category value.  Non-categorical
    columns pass through unchanged.

    .. code-block:: text

        X  ──col_j (categorical)──►  Equal(val_i)?──►  target_mean_i
                                      ...
                                      IsNaN?──────────►  nan_target
                                      default──────────►  unknown_target

        X  ──col_k (numerical)──►  unchanged

    The conversion pre-computes a combined lookup table
    (original category value → smoothed target mean) from the fitted
    :attr:`~category_encoders.TargetEncoder.ordinal_encoder` and
    :attr:`~category_encoders.TargetEncoder.mapping` attributes.
    Unknown categories and NaN inputs are handled via separate
    ``Where`` nodes that override the default value.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``TargetEncoder``
    :param outputs: desired output names
    :param X: input name (shape ``(N, F)``)
    :param name: prefix name for the added nodes
    :return: output name
    """
    assert isinstance(
        estimator, TargetEncoder
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"
    assert (
        estimator.mapping is not None
    ), f"estimator {estimator} is not fitted{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    is_string = itype == onnx.TensorProto.STRING
    # For string inputs the output is always float32 (target mean values are numeric).
    out_itype = onnx.TensorProto.FLOAT if is_string else itype
    out_dtype = np.float32 if is_string else dtype

    # Build a mapping from feature name to column index.
    feat_idx = {feat: i for i, feat in enumerate(estimator.feature_names_in_)}

    # Build per-column lookup tables from the fitted encoder state.
    # ordinal_encoder.category_mapping: list of dicts with 'col' and 'mapping'
    # (orig_val → ordinal int).  estimator.mapping: dict col → Series
    # (ordinal int → smoothed target mean float).
    col_lookup: Dict[str, dict] = {}
    for col_info in estimator.ordinal_encoder.category_mapping:
        col = col_info["col"]
        ord_map = col_info["mapping"]  # pandas Series: orig_val → ordinal
        target_map = estimator.mapping[col]  # pandas Series: ordinal → target mean

        unknown_target = float(target_map[-1])
        nan_target = float(target_map[-2])

        known_vals_list = []
        known_target_list = []
        for orig_val, ordinal in ord_map.items():
            if pd.isna(orig_val):
                continue
            known_vals_list.append(str(orig_val) if is_string else float(orig_val))
            known_target_list.append(float(target_map[ordinal]))

        col_lookup[col] = {
            "known_vals": np.array(known_vals_list, dtype=dtype),
            "known_target": np.array(known_target_list, dtype=out_dtype),
            "unknown_target": unknown_target,
            "nan_target": nan_target,
        }

    n_features = len(estimator.feature_names_in_)
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
            col_out = _target_encode_column(
                g,
                col_slice,
                info["known_vals"],
                info["known_target"],
                info["unknown_target"],
                info["nan_target"],
                out_dtype,
                f"{name}_col{col_i}",
                is_string=is_string,
            )
        else:
            # Non-categorical column: pass through as-is.
            col_out = col_slice

        col_tensors.append(col_out)

    if n_features == 1:
        res = g.op.Identity(col_tensors[0], name=name, outputs=outputs)
    else:
        res = g.op.Concat(*col_tensors, axis=1, name=name, outputs=outputs)

    g.set_type_shape_unary_op(res, X, itype=out_itype)
    return res
