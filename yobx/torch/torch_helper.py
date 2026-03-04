from __future__ import annotations
import sys
import warnings
from typing import Any
import numpy as np
import onnx
import onnx.numpy_helper as onh
import torch
from ..helpers.onnx_helper import onnx_dtype_name, tensor_dtype_to_np_dtype

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

    import ml_dtypes

    conv = {torch.bfloat16: ml_dtypes.bfloat16}
    assert tensor.dtype in conv, f"Unsupported type {tensor.dtype}, not in {conv}"
    return tensor.detach().to(torch.float32).cpu().numpy().astype(conv[tensor.dtype])


def torch_deepcopy(value: Any) -> Any:
    """
    Makes a deep copy.

    :param value: any value
    :return: a deep copy
    """
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, tuple):
        return tuple(torch_deepcopy(v) for v in value)
    if isinstance(value, list):
        if type(value) is list:
            return [torch_deepcopy(v) for v in value]
    if isinstance(value, set):
        return {torch_deepcopy(v) for v in value}
    if isinstance(value, dict):
        if type(value) is dict:
            return {k: torch_deepcopy(v) for k, v in value.items()}
        # for BaseModelOutput
        return value.__class__(**{k: torch_deepcopy(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        return value.copy()
    if hasattr(value, "clone"):
        return value.clone()
    import torch.utils._pytree as pytree

    if value.__class__ in pytree.SUPPORTED_NODES:
        args, spec = pytree.tree_flatten(value)
        new_args = torch_deepcopy(args)
        return pytree.tree_unflatten(new_args, spec)

    if hasattr(value, "__nocopy__"):
        return value

    if value.__class__.__name__ == "DynamicCache":
        # No flattening registration. Let's do something anyway.
        from .in_transformers.flatten_class import flatten_dynamic_cache, unflatten_dynamic_cache

        flat, context = flatten_dynamic_cache(value)
        return unflatten_dynamic_cache(torch_deepcopy(flat), context)

    if value.__class__.__name__ == "StaticCache":
        # No flattening registration. Let's do something anyway.
        from .in_transformers.flatten_class import flatten_static_cache, unflatten_static_cache

        flat, context = flatten_static_cache(value)
        return unflatten_static_cache(torch_deepcopy(flat), context)

    # We should have a code using serialization, deserialization assuming a model
    # cannot be exported without them.
    raise NotImplementedError(
        f"torch_deepcopy not implemented for type {type(value)}, "
        f"add attribute '__nocopy__' to return it as is."
    )


def to_tensor(tensor: onnx.TensorProto, base_dir: str = "") -> torch.Tensor:
    """
    Converts a TensorProto to a torch.Tensor.

    :param tensor: a TensorProto object.
    :param base_dir: if external tensor exists, base_dir can help to find the path to it
    :return: the converted torch tensor
    """
    assert not tensor.HasField("segment"), "Currently not supporting loading segments."
    assert (
        tensor.data_type != onnx.TensorProto.UNDEFINED
    ), "The element type in the input tensor is not defined."
    assert tensor.data_type != onnx.TensorProto.STRING, "to_tensor not implemented for strings"

    tensor_dtype = tensor.data_type
    torch_dtype = onnx_dtype_to_torch_dtype(tensor_dtype)
    dims = tuple(tensor.dims)
    if onnx.external_data_helper.uses_external_data(tensor):
        # Load raw data from external tensor if it exists
        onnx.external_data_helper.load_external_data_for_tensor(tensor, base_dir)

    if tensor.HasField("raw_data"):
        raw_data = tensor.raw_data
        if len(raw_data) == 0:
            return torch.tensor([], dtype=torch_dtype).reshape(dims)
        if sys.byteorder == "big":
            # Convert endian from little to big
            raw_data = (
                np.frombuffer(raw_data, dtype=tensor_dtype_to_np_dtype(tensor_dtype))
                .byteswap()
                .tobytes()
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.frombuffer(raw_data, dtype=torch_dtype).reshape(dims)

    # Other cases, it should be small tensor. We use numpy.
    np_tensor = onh.to_array(tensor)
    return torch.from_numpy(np_tensor)
