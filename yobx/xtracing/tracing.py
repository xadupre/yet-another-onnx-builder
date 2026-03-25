"""
Standalone tracing of numpy functions to ONNX models.

:func:`trace_numpy_function` is the low-level converter-API function that
records numpy operations as ONNX nodes in an existing
:class:`~yobx.xbuilder.GraphBuilder`.

:func:`trace_numpy_to_onnx` is the high-level entry point that creates a new
graph, calls :func:`trace_numpy_function`, and returns a self-contained
:class:`onnx.ModelProto`.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ..helpers.onnx_helper import tensor_dtype_to_np_dtype
from ..typing import GraphBuilderExtendedProtocol
from .numpy_array import NumpyArray


def trace_numpy_function(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: Optional[List[str]],
    func: Callable,
    inputs: List[str],
    name: str = "trace",
    kw_args: Optional[Dict[str, Any]] = None,
) -> Union[str, Tuple[str, ...]]:
    """
    Trace a numpy function by wrapping named tensors in *g* as
    :class:`~yobx.xtracing.numpy_array.NumpyArray` proxies, then recording all
    numpy operations as ONNX nodes in *g*.

    This function follows the same API convention as other converters in this
    package: it takes an existing :class:`~yobx.xbuilder.GraphBuilder`, the
    scikit-learn shape/type dictionary *sts*, the desired output tensor names,
    and then the callable and input tensor names.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by scikit-learn (may be empty; forwarded to
        ``g.set_type_shape_unary_op`` when non-empty)
    :param outputs: desired output tensor names (one per output returned by
        *func*)
    :param func: the numpy function to trace; must accept ``len(inputs)``
        positional arguments and return a :class:`NumpyArray` or a tuple/list
        of :class:`NumpyArray` objects
    :param inputs: existing input tensor names already present in *g*
    :param name: node name prefix used when emitting ``Identity`` rename nodes
    :param kw_args: optional keyword arguments forwarded to *func*
    :return: the first output tensor name (``outputs[0]``)

    Example:

    .. runpython::
        :showcode:

        from yobx.helpers.onnx_helper import pretty_onnx
        from yobx.xbuilder import GraphBuilder
        from yobx.xtracing import trace_numpy_function
        import numpy as np
        from onnx import TensorProto

        g = GraphBuilder({"": 21, "ai.onnx.ml": 1})
        g.make_tensor_input("X", TensorProto.FLOAT, ("batch", 3))

        def my_func(X):
            return np.sqrt(np.abs(X) + np.float32(1))

        trace_numpy_function(g, {}, ["output_0"], my_func, ["X"])
        g.make_tensor_output("output_0", indexed=False, allow_untyped_output=True)
        art = g.to_onnx()
        print(pretty_onnx(art))
    """
    # Build NumpyArray proxies from the named tensors already in g,
    # preserving the dtype information where available.
    proxies: List[NumpyArray] = []
    for inp in inputs:
        dtype = tensor_dtype_to_np_dtype(g.get_type(inp)) if g.has_type(inp) else None
        proxies.append(NumpyArray(inp, g, dtype=dtype))

    # Run the function in tracing mode; every numpy operation records an ONNX node.
    result = func(*proxies, **(kw_args or {}))

    # Normalise result to a list of NumpyArray.
    if isinstance(result, NumpyArray):
        raw_outputs: List[NumpyArray] = [result]
    elif isinstance(result, (list, tuple)):
        raw_outputs = list(result)
    else:
        raise TypeError(
            f"trace_numpy_function: function returned {type(result).__name__!r}; "
            "expected a NumpyArray or a list/tuple of NumpyArrays."
        )

    if not outputs:
        return (
            raw_outputs[0].name if len(raw_outputs) == 1 else tuple(r.name for r in raw_outputs)
        )

    if len(raw_outputs) != len(outputs):
        raise ValueError(
            f"trace_numpy_function: function produced {len(raw_outputs)} output(s) "
            f"but outputs list has {len(outputs)} element(s)."
        )

    # Emit Identity rename nodes so that each result lands under the expected name.
    for out_arr, out_name in zip(raw_outputs, outputs):
        if not isinstance(out_arr, NumpyArray):
            raise TypeError(
                f"trace_numpy_function: expected NumpyArray output, "
                f"got {type(out_arr).__name__!r}."
            )
        g.op.Identity(out_arr.name, outputs=[out_name], name=g.unique_name(name))

    return outputs[0] if len(outputs) == 1 else tuple(outputs)
