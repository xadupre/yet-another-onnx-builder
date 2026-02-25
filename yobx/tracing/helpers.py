import onnx
import onnx.helper as oh
import torch


def torch_dtype_to_onnx(dtype: torch.dtype) -> int:
    """Map a PyTorch dtype to an ONNX TensorProto data type integer."""

    _map = {
        torch.float16: onnx.TensorProto.FLOAT16,
        torch.float32: onnx.TensorProto.FLOAT,
        torch.float64: onnx.TensorProto.DOUBLE,
        torch.int8: onnx.TensorProto.INT8,
        torch.int16: onnx.TensorProto.INT16,
        torch.int32: onnx.TensorProto.INT32,
        torch.int64: onnx.TensorProto.INT64,
        torch.uint8: onnx.TensorProto.UINT8,
        torch.bool: onnx.TensorProto.BOOL,
    }
    return _map.get(dtype, onnx.TensorProto.FLOAT)


def make_value_info(
    name: str,
    onnx_dtype: int,
    shape: list | None,
    dynamic_dims: dict | None = None,
) -> onnx.ValueInfoProto:
    """Create an ONNX ValueInfoProto, replacing dynamic dimensions."""
    if shape is None:
        return oh.make_tensor_value_info(name, onnx_dtype, None)
    onnx_shape = [
        dynamic_dims[i] if (dynamic_dims and i in dynamic_dims) else d
        for i, d in enumerate(shape)
    ]
    return oh.make_tensor_value_info(name, onnx_dtype, onnx_shape)


def to_list(v: object, ndim: int) -> list:
    """Convert an int or tuple/list to a list of *ndim* elements."""
    if v is None:
        return [0] * ndim
    if hasattr(v, "__iter__"):
        return list(v)
    return [int(v)] * ndim
