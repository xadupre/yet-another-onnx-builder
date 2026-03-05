from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import numpy as np
from ..helpers.onnx_helper import np_dtype_to_tensor_dtype
from ..xbuilder import GraphBuilder
from .register import get_tensorflow_converter
from .tensorflow_helper import get_output_names, tf_dtype_to_np_dtype


def to_onnx(
    model,
    args: Tuple[Any],
    input_names: Optional[Sequence[str]] = None,
    dynamic_shapes: Optional[Tuple[Dict[int, str]]] = None,
    target_opset: Union[int, Dict[str, int]] = 20,
    verbose: int = 0,
    extra_converters: Optional[Dict[type, Callable]] = None,
):
    """
    Converts a :epkg:`TensorFlow`/:epkg:`Keras` model into ONNX.

    :param model: a fitted Keras model or layer
    :param args: dummy inputs (numpy arrays or TensorFlow tensors)
    :param input_names: optional list of names for the input tensors
    :param dynamic_shapes: optional per-input axis-to-dim-name mappings.
        When *None*, axis 0 is treated as a dynamic batch dimension named
        ``"batch"`` for every input.
    :param target_opset: opset to use; either an integer for the default domain
        (``""``), or a dictionary mapping domain names to opset versions
    :param verbose: verbosity
    :param extra_converters: optional mapping from model/layer type to converter
        function; entries here take priority over the built-in converters and
        allow converting custom layers that are not natively supported
    :return: onnx model
    """
    from . import register_tensorflow_converters

    register_tensorflow_converters()
    g = GraphBuilder(target_opset)
    cls = type(model)
    if extra_converters and cls in extra_converters:
        fct = extra_converters[cls]
    else:
        fct = get_tensorflow_converter(cls)

    if input_names:
        if len(input_names) != len(args):
            raise ValueError(
                f"Length mismatch: {len(args)=} but input_names={input_names!r}"
            )
    else:
        input_names = ["X"] if len(args) == 1 else [f"X{i}" for i in range(len(args))]

    for i, (name, arg) in enumerate(zip(input_names, args)):
        if dynamic_shapes:
            ds = dynamic_shapes[i]
        else:
            ds = {0: "batch"}
        arr = np.asarray(arg)
        shape = list(arr.shape)
        for axis, dim in ds.items():
            shape[axis] = dim
        g.make_tensor_input(
            name, np_dtype_to_tensor_dtype(arr.dtype), tuple(shape), device=-1
        )

    output_names = list(get_output_names(model))
    fct(g, {}, output_names, model, *input_names, name="main")

    for out_name in output_names:
        g.make_tensor_output(out_name, indexed=False, allow_untyped_output=True)
    return g.to_onnx()
