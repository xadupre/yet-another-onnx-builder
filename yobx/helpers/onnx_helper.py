import numpy as np
import onnx.helper as oh


def np_dtype_to_tensor_dtype(dtype: np.dtype) -> int:
    """Converts a numpy dtype to an onnx element type."""
    return oh.np_dtype_to_tensor_dtype(dtype)
