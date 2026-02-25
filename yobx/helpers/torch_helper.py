import onnx
import torch

_TYPENAME = dict(
    FLOAT=onnx.TensorProto.FLOAT,
    INT64=onnx.TensorProto.INT64,
    INT32=onnx.TensorProto.INT32,
    FLOAT16=onnx.TensorProto.FLOAT16,
    BFLOAT16=onnx.TensorProto.BFLOAT16,
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
