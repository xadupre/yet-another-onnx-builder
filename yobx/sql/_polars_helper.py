"""
Polars helper utilities for the SQL-to-ONNX converter.

This module provides the mapping from polars column dtypes to numpy dtypes
and the :func:`polars_schema_to_input_dtypes` helper used by
:func:`~yobx.sql.to_onnx` to extract column types from a
``polars.LazyFrame`` or ``polars.DataFrame`` without collecting the frame.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

# ---------------------------------------------------------------------------
# Polars dtype → numpy dtype mapping
# ---------------------------------------------------------------------------

# Mapping from polars DataType class name to numpy dtype.
_POLARS_DTYPE_TO_NP: Dict[str, np.dtype] = {
    "Float32": np.dtype("float32"),
    "Float64": np.dtype("float64"),
    "Int8": np.dtype("int8"),
    "Int16": np.dtype("int16"),
    "Int32": np.dtype("int32"),
    "Int64": np.dtype("int64"),
    "UInt8": np.dtype("uint8"),
    "UInt16": np.dtype("uint16"),
    "UInt32": np.dtype("uint32"),
    "UInt64": np.dtype("uint64"),
    "Boolean": np.dtype("bool"),
    "String": np.dtype("object"),
    "Utf8": np.dtype("object"),
}


def polars_schema_to_input_dtypes(
    frame,
) -> Dict[str, np.dtype]:
    """Extract a column-name → numpy-dtype mapping from a polars frame.

    Accepts a ``polars.LazyFrame`` (schema read via :meth:`collect_schema`,
    no data collected) or a ``polars.DataFrame`` (schema read via
    :attr:`schema`).

    :param frame: a ``polars.LazyFrame`` or ``polars.DataFrame``.
    :return: dict mapping each column name to its numpy dtype.
    :raises ImportError: if *polars* is not installed.
    :raises TypeError: if *frame* is neither a LazyFrame nor a DataFrame.
    :raises ValueError: if any column dtype has no supported numpy equivalent.

    Supported polars dtypes: ``Float32``, ``Float64``, ``Int8``, ``Int16``,
    ``Int32``, ``Int64``, ``UInt8``, ``UInt16``, ``UInt32``, ``UInt64``,
    ``Boolean``, ``String``, ``Utf8``.
    """
    try:
        import polars as pl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "polars is required for yobx.sql.to_onnx. "
            "Install it with: pip install polars"
        ) from exc

    if hasattr(frame, "collect_schema"):
        # polars.LazyFrame — cheap, no execution
        schema = frame.collect_schema()
    elif hasattr(frame, "schema"):
        # polars.DataFrame
        schema = frame.schema
    else:
        raise TypeError(
            f"Expected a polars.LazyFrame or polars.DataFrame, got {type(frame)!r}"
        )

    result: Dict[str, np.dtype] = {}
    for col_name, dtype in schema.items():
        key = type(dtype).__name__
        if key not in _POLARS_DTYPE_TO_NP:
            raise ValueError(
                f"Column {col_name!r} has polars dtype {dtype!r} which cannot be "
                f"mapped to a numpy dtype supported by sql_to_onnx. "
                f"Supported polars dtypes: {sorted(_POLARS_DTYPE_TO_NP)}"
            )
        result[col_name] = _POLARS_DTYPE_TO_NP[key]
    return result
