from __future__ import annotations

import numpy as np
import onnx
from .onnx_helper import onnx_dtype_name

_TYPENAME = dict(
    FLOAT=onnx.TensorProto.FLOAT,
    INT64=onnx.TensorProto.INT64,
    INT32=onnx.TensorProto.INT32,
    FLOAT16=onnx.TensorProto.FLOAT16,
    BFLOAT16=onnx.TensorProto.BFLOAT16,
)


def onnx_dtype_to_torch_dtype(itype: int) -> torch.dtype:
    """
    Converts an onnx type into a torch dtype.

    :param to: onnx dtype
    :return: torch dtype
    """
    import torch

    if itype == onnx.TensorProto.FLOAT:
        return torch.float32
    if itype == onnx.TensorProto.FLOAT16:
        return torch.float16
    if itype == onnx.TensorProto.BFLOAT16:
        return torch.bfloat16
    if itype == onnx.TensorProto.DOUBLE:
        return torch.float64
    if itype == onnx.TensorProto.INT32:
        return torch.int32
    if itype == onnx.TensorProto.INT64:
        return torch.int64
    if itype == onnx.TensorProto.UINT32:
        return torch.uint32
    if itype == onnx.TensorProto.UINT64:
        return torch.uint64
    if itype == onnx.TensorProto.BOOL:
        return torch.bool
    if itype == onnx.TensorProto.INT16:
        return torch.int16
    if itype == onnx.TensorProto.UINT16:
        return torch.uint16
    if itype == onnx.TensorProto.INT8:
        return torch.int8
    if itype == onnx.TensorProto.UINT8:
        return torch.uint8
    if itype == onnx.TensorProto.COMPLEX64:
        return torch.complex64
    if itype == onnx.TensorProto.COMPLEX128:
        return torch.complex128
    raise NotImplementedError(
        f"Unable to convert onnx type {onnx_dtype_name(itype)} to torch.type."
    )


def torch_dtype_to_onnx_dtype(to: torch.dtype) -> int:
    """
    Converts a torch dtype into a onnx element type.

    :param to: torch dtype
    :return: onnx type
    """
    import torch

    if to == torch.float32:
        return onnx.TensorProto.FLOAT
    if to == torch.float16:
        return onnx.TensorProto.FLOAT16
    if to == torch.bfloat16:
        return onnx.TensorProto.BFLOAT16
    if to == torch.float64:
        return onnx.TensorProto.DOUBLE
    if to == torch.int64:
        return onnx.TensorProto.INT64
    if to == torch.int32:
        return onnx.TensorProto.INT32
    if to == torch.uint64:
        return onnx.TensorProto.UINT64
    if to == torch.uint32:
        return onnx.TensorProto.UINT32
    if to == torch.bool:
        return onnx.TensorProto.BOOL
    if to == torch.SymInt:
        return onnx.TensorProto.INT64
    if to == torch.int16:
        return onnx.TensorProto.INT16
    if to == torch.uint16:
        return onnx.TensorProto.UINT16
    if to == torch.int8:
        return onnx.TensorProto.INT8
    if to == torch.uint8:
        return onnx.TensorProto.UINT8
    if to == torch.SymFloat:
        return onnx.TensorProto.FLOAT
    if to == torch.complex64:
        return onnx.TensorProto.COMPLEX64
    if to == torch.complex128:
        return onnx.TensorProto.COMPLEX128
    # SymbolicTensor
    sto = str(to)
    if sto in _TYPENAME:
        return _TYPENAME[sto]
    raise NotImplementedError(f"Unable to convert torch dtype {to!r} ({type(to)}) to onnx dtype.")


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts a :class:`torch.Tensor` to :class:`numpy.ndarray`."""
    try:
        return tensor.detach().cpu().numpy()
    except TypeError:
        # We try with ml_dtypes
        pass

    import torch
    import ml_dtypes

    conv = {torch.bfloat16: ml_dtypes.bfloat16}
    assert tensor.dtype in conv, f"Unsupported type {tensor.dtype}, not in {conv}"
    return tensor.detach().to(torch.float32).cpu().numpy().astype(conv[tensor.dtype])
