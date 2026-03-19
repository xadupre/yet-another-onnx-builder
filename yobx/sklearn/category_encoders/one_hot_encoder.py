from typing import Dict, List

import numpy as np
import onnx
import pandas as pd
from category_encoders import OneHotEncoder

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype


@register_sklearn_converter(OneHotEncoder)
def category_encoders_one_hot_encoder(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: OneHotEncoder,
    X: str,
    name: str = "ce_one_hot_encoder",
) -> str:
    """
    Converts a :class:`category_encoders.OneHotEncoder` into ONNX.

    The encoder replaces each categorical column with a block of binary
    indicator columns (one per known category) and passes non-categorical
    columns through unchanged.

    .. code-block:: text

        X  ──col_j (categorical, K cats)──►  Equal(c_1)?──Cast(float)──► ind_1 (N,1)
                                              Equal(c_2)?──Cast(float)──► ind_2 (N,1)
                                              ...
                                              Equal(c_K)?──Cast(float)──► ind_K (N,1)
                                              Concat(ind_1,...,ind_K, axis=1)──► block (N,K)

        X  ──col_k (numerical)──►  unchanged (N,1)

        Concat(all blocks and pass-through cols, axis=1)──► output (N, F_out)

    The conversion reads the fitted
    :attr:`~category_encoders.BaseEncoder.ordinal_encoder` and
    :attr:`~category_encoders.BaseEncoder.mapping` attributes to determine
    the known category values and their one-hot positions.

    **Unknown categories** (values not seen during training):

    * ``handle_unknown='value'`` (default): the entire block for that row is
      all-zero (naturally produced by the ``Equal`` comparisons returning
      ``False``).
    * ``handle_unknown='return_nan'``: the entire block is ``NaN``. This is
      detected by checking that no ``Equal`` node fired (``ReduceMax`` of
      indicator values is 0) and, for floating-point inputs, that the value
      is not itself ``NaN`` (NaN inputs always produce a zero block).

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names
    :param estimator: a fitted :class:`~category_encoders.OneHotEncoder`
    :param X: name of the input tensor (shape ``(N, F)``)
    :param name: prefix used for names of nodes added by this converter
    :return: name of the output tensor
    :raises AssertionError: if ``estimator`` is not fitted or type info is
        missing from the graph
    """
    assert isinstance(
        estimator, OneHotEncoder
    ), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"
    assert (
        estimator.mapping is not None
    ), f"estimator {estimator} is not fitted{g.get_debug_msg()}"

    itype = g.get_type(X)
    dtype = tensor_dtype_to_np_dtype(itype)
    float_onnx_type = int(itype)

    # Detect floating-point input (required for IsNaN handling).
    _FLOAT_TYPES = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.DOUBLE}
    is_float_input = itype in _FLOAT_TYPES

    handle_return_nan = estimator.handle_unknown == "return_nan"

    # Map feature name -> column index.
    feat_idx = {feat: i for i, feat in enumerate(estimator.feature_names_in_)}
    cols_set = set(estimator.cols)

    # Build ordinal-encoding lookup: col -> Series(orig_val -> ordinal).
    ord_lookup: Dict[str, pd.Series] = {}
    for cm in estimator.ordinal_encoder.category_mapping:
        ord_lookup[cm["col"]] = cm["mapping"]

    col_tensors = []
    for feat_name in estimator.feature_names_in_:
        col_i = feat_idx[feat_name]
        col_slice = g.op.Gather(
            X, np.array([col_i], dtype=np.int64), axis=1, name=f"{name}_gather{col_i}"
        )  # (N, 1)

        if feat_name not in cols_set:
            # Non-categorical column: pass through unchanged.
            col_tensors.append(col_slice)
            continue

        # Categorical column: build K indicator columns.
        ord_map = ord_lookup[feat_name]

        # Collect known (non-NaN) categories sorted by their ordinal so that
        # the output columns match the order expected by the encoder.
        known_cats = [
            (orig_val, int(ordinal))
            for orig_val, ordinal in ord_map.items()  # type: ignore
            if not pd.isna(orig_val)  # type: ignore
        ]
        known_cats.sort(key=lambda x: x[1])

        if not known_cats:
            # Edge case: no known categories - skip this column entirely.
            continue

        # Emit one Equal+Cast node per category.
        cast_list = []
        for cat_idx, (cat_val, _ordinal) in enumerate(known_cats):
            cat_const = np.array([[cat_val]], dtype=dtype)
            eq_k = g.op.Equal(col_slice, cat_const, name=f"{name}_eq{col_i}_{cat_idx}")
            cast_k = g.op.Cast(eq_k, to=float_onnx_type, name=f"{name}_cast{col_i}_{cat_idx}")
            cast_list.append(cast_k)

        if handle_return_nan:
            # Detect rows that are neither a known category nor NaN.
            # For such rows every indicator column must be set to NaN.
            K = len(cast_list)
            if K == 1:
                stacked = cast_list[0]
            else:
                stacked = g.op.Concat(*cast_list, axis=1, name=f"{name}_stack{col_i}")

            # ReduceMax over indicators: 1.0 if any category matched, else 0.0.
            any_match_val = g.op.ReduceMax(
                stacked, np.array([1], dtype=np.int64), keepdims=1, name=f"{name}_anymax{col_i}"
            )  # (N, 1)
            # Threshold at 0.5 to convert from float indicator (0.0 or 1.0) to bool.
            half = np.array([[0.5]], dtype=dtype)
            any_match = g.op.Greater(
                any_match_val, half, name=f"{name}_anymatch{col_i}"
            )  # (N, 1) bool

            if is_float_input:
                is_nan = g.op.IsNaN(col_slice, name=f"{name}_isnan{col_i}")
                match_or_nan = g.op.Or(any_match, is_nan, name=f"{name}_matchornan{col_i}")
                unknown_mask = g.op.Not(match_or_nan, name=f"{name}_unknown{col_i}")
            else:
                unknown_mask = g.op.Not(any_match, name=f"{name}_unknown{col_i}")

            nan_scalar = np.array([[float("nan")]], dtype=dtype)
            result_list = [
                g.op.Where(
                    unknown_mask, nan_scalar, cast_k, name=f"{name}_where{col_i}_{cat_idx}"
                )
                for cat_idx, cast_k in enumerate(cast_list)
            ]
        else:
            result_list = cast_list

        if len(result_list) == 1:
            block = g.op.Identity(result_list[0], name=f"{name}_block{col_i}")
        else:
            block = g.op.Concat(*result_list, axis=1, name=f"{name}_concat{col_i}")

        col_tensors.append(block)

    if not col_tensors:
        raise ValueError(
            f"category_encoders OneHotEncoder produces no output columns "
            f"(estimator.feature_names_in_={list(estimator.feature_names_in_)!r})."
        )

    if len(col_tensors) == 1:
        res = g.op.Identity(col_tensors[0], name=name, outputs=outputs)
    else:
        res = g.op.Concat(*col_tensors, axis=1, name=name, outputs=outputs)

    return res
