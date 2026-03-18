"""
yobx.polars — ONNX conversion for :class:`polars.DataFrame` schemas.
"""

from .convert import to_onnx, polars_dtype_to_onnx_element_type, schema_to_numpy_dtypes
