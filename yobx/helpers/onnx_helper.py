import functools
from typing import Set, Optional, Union
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh


@functools.cache
def onnx_dtype_name(itype: int, exc: bool = True) -> str:
    """
    Returns the ONNX name for a specific element type.

    .. runpython::
        :showcode:

        import onnx
        from yobx.helpers.onnx_helper import onnx_dtype_name

        itype = onnx.onnx.TensorProto.BFLOAT16
        print(onnx_dtype_name(itype))
        print(onnx_dtype_name(7))
    """
    for k in dir(onnx.onnx.TensorProto):
        if k.upper() == k and k not in {"DESCRIPTOR", "EXTERNAL", "DEFAULT"}:
            v = getattr(onnx.onnx.TensorProto, k)
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


def dtype_to_tensor_dtype(dt: Union[np.dtype, "torch.dtype"]) -> int:  # type: ignore[arg-type,name-defined] # noqa: F821
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
    Converts a onnx.TensorProto's data_type to corresponding numpy dtype.
    It can be used while making tensor.

    :param tensor_dtype: onnx.TensorProto's data_type
    :return: numpy's data_type
    """
    return oh.tensor_dtype_to_np_dtype(tensor_dtype)


def pretty_onnx(
    onx: Union[
        onnx.AttributeProto,
        onnx.FunctionProto,
        onnx.GraphProto,
        onnx.ModelProto,
        onnx.NodeProto,
        onnx.SparseTensorProto,
        onnx.TensorProto,
        onnx.ValueInfoProto,
        str,
    ],
    with_attributes: bool = False,
    highlight: Optional[Set[str]] = None,
    shape_inference: bool = False,
) -> str:
    """
    Displays an onnx proto in a better way.

    :param with_attributes: displays attributes as well, if only a node is printed
    :param highlight: to highlight some names
    :param shape_inference: run shape inference before printing the model
    :return: text
    """
    assert onx is not None, "onx cannot be None"
    if isinstance(onx, str):
        onx = onnx.load(onx, load_external_data=False)
    assert onx is not None, "onx cannot be None"

    if shape_inference:
        assert isinstance(
            onx, onnx.ModelProto
        ), f"shape inference only works for ModelProto, not {type(onx)}"
        onx = onnx.shape_inference.infer_shapes(onx)

    if isinstance(onx, onnx.ValueInfoProto):
        name = onx.name
        itype = onx.type.tensor_type.elem_type
        shape = tuple((d.dim_param or d.dim_value) for d in onx.type.tensor_type.shape.dim)
        shape_str = ",".join(map(str, shape))
        return f"{onnx_dtype_name(itype, exc=False)}[{shape_str}] {name}"

    if isinstance(onx, onnx.TypeProto):
        itype = onx.tensor_type.elem_type
        shape = tuple((d.dim_param or d.dim_value) for d in onx.tensor_type.shape.dim)
        shape_str = ",".join(map(str, shape))
        return f"{onnx_dtype_name(itype, exc=False)}[{shape_str}]"

    if isinstance(onx, onnx.AttributeProto):
        att = onx
        if att.type == onnx.AttributeProto.INT:
            return f"{att.name}={att.i}"
        if att.type == onnx.AttributeProto.INTS:
            return f"{att.name}={att.ints}"
        if att.type == onnx.AttributeProto.FLOAT:
            return f"{att.name}={att.f}"
        if att.type == onnx.AttributeProto.FLOATS:
            return f"{att.name}={att.floats}"
        if att.type == onnx.AttributeProto.STRING:
            return f"{att.name}={att.s!r}"
        if att.type == onnx.AttributeProto.TENSOR:
            v = onh.to_array(att.t)
            assert hasattr(v, "reshape"), f"not a tensor {type(v)}"
            assert hasattr(v, "shape"), f"not a tensor {type(v)}"
            vf = v.reshape((-1,))
            if vf.size < 10:
                tt = f"[{', '.join(map(str, vf))}]"
            else:
                tt = f"[{', '.join(map(str, vf[:10]))}, ...]"
            if len(v.shape) != 1:
                return f"{att.name}=tensor({tt}, dtype={v.dtype}).reshape({v.shape})"
            return f"{att.name}=tensor({tt}, dtype={v.dtype})"
        raise NotImplementedError(f"pretty_onnx not implemented yet for AttributeProto={att!r}")

    if isinstance(onx, onnx.NodeProto):

        def _high(n):
            if highlight and n in highlight:
                return f"**{n}**"
            return n

        text = (
            f"{onx.op_type}({', '.join(map(_high, onx.input))})"
            f" -> {', '.join(map(_high, onx.output))}"
        )
        if onx.domain:
            text = f"{onx.domain}.{text}"
        if not with_attributes or not onx.attribute:
            return text
        rows = []
        for att in onx.attribute:
            rows.append(pretty_onnx(att))
        if len(rows) > 1:
            suffix = "\n".join(f"    {s}" for s in rows)
            return f"{text}\n{suffix}"
        return f"{text}  ---  {rows[0]}"

    if isinstance(onx, onnx.TensorProto):
        shape = "x".join(d.dim_param or str(d.dim_value) for d in onx.dims)
        return f"onnx.TensorProto:{onx.data_type}:{shape}:{onx.name}"

    assert not isinstance(
        onx, onnx.SparseTensorProto
    ), "Sparseonnx.TensorProto is not handled yet."

    from ._onnx_simple_text_plot import onnx_simple_text_plot

    if isinstance(onx, onnx.FunctionProto):
        return (
            f"function: {onx.name}[{onx.domain}]\n"
            f"{onnx_simple_text_plot(onx, recursive=True)}"
        )
    return onnx_simple_text_plot(onx, recursive=True)
