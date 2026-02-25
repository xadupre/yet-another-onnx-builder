import functools
from typing import Dict, Union
import numpy as np
import onnx
import onnx.helper as oh


@functools.cache
def onnx_dtype_name(itype: int, exc: bool = True) -> str:
    """
    Returns the ONNX name for a specific element type.

    .. runpython::
        :showcode:

        import onnx
        from yobx.helpers.onnx_helper import onnx_dtype_name

        itype = onnx.TensorProto.BFLOAT16
        print(onnx_dtype_name(itype))
        print(onnx_dtype_name(7))
    """
    for k in dir(onnx.TensorProto):
        if k.upper() == k and k not in {"DESCRIPTOR", "EXTERNAL", "DEFAULT"}:
            v = getattr(onnx.TensorProto, k)
            if v == itype:
                return k
    if exc:
        raise ValueError(f"Unexpected value itype: {itype}")
    if itype == 0:
        return "UNDEFINED"
    return "UNEXPECTED"


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


def tensor_dtype_to_np_dtype(tensor_dtype: int) -> np.dtype:
    """
    Converts a TensorProto's data_type to corresponding numpy dtype.
    It can be used while making tensor.

    :param tensor_dtype: TensorProto's data_type
    :return: numpy's data_type
    """
    if tensor_dtype >= 16:
        try:
            import ml_dtypes  # noqa: F401
        except ImportError as e:
            raise ValueError(
                f"Unsupported value for tensor_dtype, "
                f"numpy does not support onnx type {tensor_dtype}. "
                f"ml_dtypes can be used."
            ) from e

        # pyrefly: ignore[bad-assignment]
        mapping: Dict[int, np.dtype] = {
            onnx.TensorProto.BFLOAT16: ml_dtypes.bfloat16,
            onnx.TensorProto.FLOAT8E4M3FN: ml_dtypes.float8_e4m3fn,
            onnx.TensorProto.FLOAT8E4M3FNUZ: ml_dtypes.float8_e4m3fnuz,
            onnx.TensorProto.FLOAT8E5M2: ml_dtypes.float8_e5m2,
            onnx.TensorProto.FLOAT8E5M2FNUZ: ml_dtypes.float8_e5m2fnuz,
        }
        assert (
            tensor_dtype in mapping
        ), f"Unable to find tensor_dtype={tensor_dtype!r} in mapping={mapping}"
        return mapping[tensor_dtype]

    return oh.tensor_dtype_to_np_dtype(tensor_dtype)
