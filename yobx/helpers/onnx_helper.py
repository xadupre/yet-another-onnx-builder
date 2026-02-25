from typing import Union
import numpy as np
import onnx.helper as oh


def np_dtype_to_tensor_dtype(dtype: np.dtype) -> int:
    """Converts a numpy dtype to an onnx element type."""
    return oh.np_dtype_to_tensor_dtype(dtype)


def dtype_to_tensor_dtype(dt: Union[np.dtype, "torch.dtype"]) -> int:  # noqa: F821
    """
    Converts a torch dtype or numpy dtype into a onnx element type.

    :param to: dtype
    :return: onnx type
    """
    try:
        return np_dtype_to_tensor_dtype(dt)
    except (KeyError, TypeError, ValueError):
        pass
    from .torch_helper import torch_dtype_to_onnx_dtype

    return torch_dtype_to_onnx_dtype(dt)
