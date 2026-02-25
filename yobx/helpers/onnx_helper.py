import numpy as np
import onnx.helper as oh
from onnx_diagnostic.helpers.onnx_helper import pretty_onnx  # noqa: F401


def np_dtype_to_tensor_dtype(dtype: np.dtype) -> int:
    """Converts a numpy dtype to an onnx element type."""
    return oh.np_dtype_to_tensor_dtype(dtype)
