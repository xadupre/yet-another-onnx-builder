from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import pandas as pd
from category_encoders import BaseNEncoder

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


def _basen_encode_column(
    g: GraphBuilderExtendedProtocol,
    col_tensor: str,
    known_cats: List[Tuple],
    n_digits: int,
    dtype: np.dtype,
    itype: int,
    name: str,
    handle_unknown: str = "value",
    handle_missing: str = "value",
    is_float_input: bool = True,
    cat_dtype: Optional[np.dtype] = None,
) -> str:
    """Emit ONNX nodes that apply base-N encoding to a single column tensor.

    For each input value the output is a row of *n_digits* base-N digits
    (most significant digit first) encoding the ordinal index assigned to that
    category:

    1. If the value matches a known category → base-N representation of its ordinal
    2. If the value is NaN and *handle_missing* is ``'return_nan'`` → NaN for all
       digits (skipped when *is_float_input* is ``False``)
    3. If the value is an unknown category and *handle_unknown* is ``'return_nan'``
       → NaN for all digits
    4. Otherwise (unknown or NaN with ``'value'``) → 0 for all digits

    :param g: graph builder
    :param col_tensor: ONNX tensor name for a single-column slice, shape ``(N, 1)``
    :param known_cats: list of ``(cat_val, digits)`` pairs for known categories,
        where ``digits`` is a sequence of *n_digits* integer digits (MSB first);
        NaN entries are excluded
    :param n_digits: number of output digits (columns) per row
    :param dtype: numpy dtype for the output constants
    :param itype: ONNX integer type code for the output tensor
    :param name: node name prefix
    :param handle_unknown: ``'value'`` (→ 0) or ``'return_nan'`` (→ NaN)
    :param handle_missing: ``'value'`` (→ 0) or ``'return_nan'`` (→ NaN)
    :param is_float_input: ``True`` if the input tensor is floating-point
        (required for ``IsNaN`` support); set to ``False`` for string inputs
    :param cat_dtype: numpy dtype for category value constants; defaults to
        *dtype* when ``None``; use ``object`` for string inputs
    :return: ONNX tensor name, shape ``(N, n_digits)``
    """
    if cat_dtype is None:
        cat_dtype = dtype
    is_float_dtype = np.issubdtype(dtype, np.floating)
    nan_const = np.array([[float("nan")]], dtype=dtype) if is_float_dtype else None

    # Build one digit column per output position (MSB first).
    digit_cols = []
    for j in range(n_digits):
        # Default digit value: 0 (used when no category matches).
        result: Union[str, np.ndarray] = np.zeros((1, 1), dtype=dtype)

        # For every known category with a non-zero digit here, conditionally set it.
        for cat_idx, (cat_val, digits) in enumerate(known_cats):
            digit = int(digits[j])
            if digit != 0:
                cat_const = np.array([[cat_val]], dtype=cat_dtype)
                eq = g.op.Equal(col_tensor, cat_const, name=f"{name}_eq_d{j}_c{cat_idx}")
                digit_const = np.full((1, 1), digit, dtype=dtype)
                result = g.op.Where(eq, digit_const, result, name=f"{name}_where_d{j}_c{cat_idx}")

        digit_cols.append(result)

    # Apply handle_missing='return_nan': NaN inputs → NaN for all digits.
    if handle_missing == "return_nan" and is_float_input:
        is_nan = g.op.IsNaN(col_tensor, name=f"{name}_isnan_missing")
        digit_cols = [
            g.op.Where(is_nan, nan_const, dc, name=f"{name}_miss_where_d{j}")
            for j, dc in enumerate(digit_cols)
        ]

    # Apply handle_unknown='return_nan': unknown inputs → NaN for all digits.
    # Only applicable for floating-point inputs (NaN cannot be represented in integer dtype).
    if handle_unknown == "return_nan" and known_cats and is_float_input:
        # Compute a float indicator: 1.0 if *any* known category matched.
        eq_indicators = []
        for i, (cat_val, _) in enumerate(known_cats):
            cat_const = np.array([[cat_val]], dtype=cat_dtype)
            eq = g.op.Equal(col_tensor, cat_const, name=f"{name}_equk{i}")
            eq_float = g.op.Cast(eq, to=itype, name=f"{name}_castuk{i}")
            eq_indicators.append(eq_float)

        if len(eq_indicators) == 1:
            any_match_val = eq_indicators[0]
        else:
            stacked = g.op.Concat(*eq_indicators, axis=1, name=f"{name}_stackuk")
            any_match_val = g.op.ReduceMax(
                stacked, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_anymaxuk"
            )

        half = np.array([[0.5]], dtype=dtype)
        any_match = g.op.Greater(any_match_val, half, name=f"{name}_anymatchuk")

        if is_float_input:
            # Treat NaN inputs as "known" so that the handle_missing path is
            # solely responsible for NaN outputs (avoids double-application).
            is_nan_uk = g.op.IsNaN(col_tensor, name=f"{name}_isnan_uk")
            known_or_nan = g.op.Or(any_match, is_nan_uk, name=f"{name}_kornan")
            unknown_mask = g.op.Not(known_or_nan, name=f"{name}_unknown")
        else:
            unknown_mask = g.op.Not(any_match, name=f"{name}_unknown")

        digit_cols = [
            g.op.Where(unknown_mask, nan_const, dc, name=f"{name}_unk_where_d{j}")
            for j, dc in enumerate(digit_cols)
        ]

    if len(digit_cols) == 1:
        return digit_cols[0]
    return g.op.Concat(*digit_cols, axis=1, name=f"{name}_digitconcat")


@register_sklearn_converter(BaseNEncoder)
def category_encoders_basen_encoder(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: BaseNEncoder,
    X: str,
    name: str = "basen_encoder",
) -> str:
    """
    Converts a :class:`category_encoders.BaseNEncoder` into ONNX.

    Each categorical column is replaced by a block of digit columns that encode
    the ordinal index of the category value in base ``N`` (most significant
    digit first).  Non-categorical columns pass through unchanged.

    .. code-block:: text

        X  ──col_j (categorical, K cats)──►  digit_0 (MSD)  (N, 1)
                                              digit_1         (N, 1)
                                              ...
                                              digit_D (LSD)   (N, 1)
                                              Concat(digit_0 ... digit_D, axis=1)──► block (N, D)

        X  ──col_k (numerical)──►  unchanged  (N, 1)

        Concat(all blocks and pass-through cols, axis=1)──► output (N, F_out)

    where ``D`` is the number of base-``N`` digits required to represent the
    largest ordinal and ``F_out`` is the total number of output columns.

    The conversion reads the fitted
    :attr:`~category_encoders.BaseNEncoder.mapping` attribute (ordinal → digit
    row) together with the
    :attr:`~category_encoders.BaseEncoder.ordinal_encoder` attribute (category
    value → ordinal) to determine the per-column digit blocks.

    **Unknown categories** (values not seen during training):

    * ``handle_unknown='value'`` (default): all digit columns for that row are 0.
    * ``handle_unknown='return_nan'``: all digit columns for that row are ``NaN``.

    **Missing values** (NaN inputs):

    * ``handle_missing='value'`` (default): all digit columns for that row are 0.
    * ``handle_missing='return_nan'``: all digit columns for that row are ``NaN``.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names
    :param estimator: a fitted :class:`~category_encoders.BaseNEncoder`
    :param X: name of the input tensor (shape ``(N, F)``)
    :param name: prefix used for names of nodes added by this converter
    :return: name of the output tensor
    :raises AssertionError: if ``estimator`` is not fitted or type info is
        missing from the graph
    """
    assert isinstance(
        estimator, BaseNEncoder
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"
    assert (
        estimator.mapping is not None
    ), f"estimator {estimator} is not fitted{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)

    is_string = itype == onnx.TensorProto.STRING
    # For string inputs the output is int64, matching BaseNEncoder.transform() output dtype.
    out_itype = onnx.TensorProto.INT64 if is_string else itype
    out_dtype = np.int64 if is_string else dtype

    # Detect floating-point input (required for IsNaN support).
    _FLOAT_TYPES = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.DOUBLE}
    is_float_input = itype in _FLOAT_TYPES

    handle_unknown = estimator.handle_unknown
    handle_missing = estimator.handle_missing

    # Build a mapping from feature name to column index.
    feat_idx = {feat: i for i, feat in enumerate(estimator.feature_names_in_)}
    cols_set = set(estimator.cols)

    # Build per-column digit lookup tables from the fitted encoders.
    # ordinal_encoder.category_mapping: list of dicts with 'col' and 'mapping'
    # (orig_val → ordinal int).  Ordinals for known categories are 1, 2, 3, …;
    # NaN maps to -2 (excluded here); unknown maps to -1 (never stored).
    # estimator.mapping: list of dicts with 'col' and 'mapping', a DataFrame
    # indexed by ordinal whose columns hold the base-N digits (MSD first).
    digit_maps: Dict[str, pd.DataFrame] = {
        switch["col"]: switch["mapping"] for switch in estimator.mapping
    }

    col_info: Dict[str, dict] = {}
    for col_info_item in estimator.ordinal_encoder.category_mapping:
        col = col_info_item["col"]
        ord_map = col_info_item["mapping"]  # pandas Series: orig_val → ordinal
        digit_df = digit_maps[col]
        n_digits = digit_df.shape[1]

        known_cats: List[Tuple[object, Sequence[int]]] = []
        for orig_val, ordinal in ord_map.items():
            if pd.isna(orig_val):
                continue
            cat_val = str(orig_val) if is_string else float(orig_val)
            digits = digit_df.loc[int(ordinal)].to_numpy()
            known_cats.append((cat_val, digits))

        col_info[col] = {"known_cats": known_cats, "n_digits": n_digits}

    col_tensors = []
    n_out = 0

    for feat_name in estimator.feature_names_in_:
        col_i = feat_idx[feat_name]
        # Extract column: shape (N, 1).
        col_slice = g.op.Gather(
            X, np.array([col_i], dtype=np.int64), axis=1, name=f"{name}_gather{col_i}"
        )

        if feat_name not in cols_set:
            # Non-categorical column: pass through unchanged.
            col_tensors.append(col_slice)
            n_out += 1
            continue

        info = col_info[feat_name]
        n_digits = info["n_digits"]
        known_cats = info["known_cats"]

        if n_digits == 0:
            # Edge case: no known categories for this column; skip it.
            continue

        block = _basen_encode_column(
            g,
            col_slice,
            known_cats,
            n_digits,
            out_dtype,
            out_itype,
            f"{name}_col{col_i}",
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            is_float_input=is_float_input,
            cat_dtype=dtype,
        )
        col_tensors.append(block)
        n_out += n_digits

    if not col_tensors:
        raise ValueError(
            f"category_encoders BaseNEncoder produces no output columns "
            f"(estimator.feature_names_in_={list(estimator.feature_names_in_)!r})."
        )

    if len(col_tensors) == 1:
        assert (
            len(outputs) == 1
        ), f"Inconsistencies {col_tensors=} and {outputs=}{g.get_debug_msg()}"
        res = g.op.Identity(col_tensors[0], name=name, outputs=outputs)
    elif len(outputs) == 1:
        res = g.op.Concat(*col_tensors, axis=1, name=name, outputs=outputs)
    elif len(outputs) == len(col_tensors):
        assert len(outputs) == len(
            col_tensors
        ), f"Inconsistencies {col_tensors=} and {outputs=}{g.get_debug_msg()}"
        res = tuple(
            g.op.Identity(c, name=name, outputs=[o]) for c, o in zip(col_tensors, outputs)
        )
    else:
        raise AssertionError(f"Inconsistencies {col_tensors=} and {outputs=}{g.get_debug_msg()}")
    return res
