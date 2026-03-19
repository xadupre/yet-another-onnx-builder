"""
Standalone tracing of numpy functions to ONNX models.

:func:`trace_numpy_function` is the low-level converter-API function that
records numpy operations as ONNX nodes in an existing
:class:`~yobx.xbuilder.GraphBuilder`.

:func:`trace_numpy_to_onnx` is the high-level entry point that creates a new
graph, calls :func:`trace_numpy_function`, and returns a self-contained
:class:`onnx.ModelProto`.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from .. import DEFAULT_TARGET_OPSET
from ..container import ExportArtifact
from ..helpers.onnx_helper import np_dtype_to_tensor_dtype, tensor_dtype_to_np_dtype
from ..typing import GraphBuilderExtendedProtocol
from ..xbuilder import GraphBuilder
from .numpy_array import NumpyArray

#: Name used for the dynamic (batch) first dimension when none is specified.
_BATCH_DIM = "batch"


def trace_numpy_function(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    func: Callable,
    inputs: List[str],
    name: str = "trace",
    kw_args: Optional[Dict[str, Any]] = None,
) -> str:
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

    Example::

    .. runpython::
        :showcode:

        from yobx.helpers.onnx_hepler import pretty_onnx
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
        onx = g.to_onnx()
        print(pretty_onnx(onx))
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

    return outputs[0]


def trace_numpy_to_onnx(
    func: Callable,
    *inputs: np.ndarray,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    target_opset: Union[int, Dict[str, int]] = DEFAULT_TARGET_OPSET,
    batch_dim: str = _BATCH_DIM,
) -> ExportArtifact:
    """
    Trace a numpy function and return the equivalent ONNX model.

    This is the high-level entry point.  Internally it creates a fresh
    :class:`~yobx.xbuilder.GraphBuilder`, registers the graph inputs derived
    from the *inputs* sample arrays, delegates the actual tracing to
    :func:`trace_numpy_function`, registers the graph outputs, and exports to
    :class:`onnx.ModelProto`.

    :param func: a Python callable that accepts one or more numpy arrays and
        returns a numpy array (or a tuple/list of arrays).  The function may
        use numpy ufuncs, arithmetic operators, reductions, and shape
        manipulations; see :mod:`yobx.xtracing.numpy_array` for the full list
        of supported operations.
    :param inputs: sample numpy arrays used to determine the element type and
        shape of each input.  Only ``dtype`` and ``shape`` are used; the
        actual values are ignored.
    :param input_names: optional list of tensor names for the ONNX graph
        inputs.  Defaults to ``["X"]`` for a single input or ``["X0", "X1",
        …]`` for multiple inputs.
    :param output_names: optional list of tensor names for the ONNX graph
        outputs.  Defaults to ``["output_0"]``.  For functions that return
        multiple arrays supply the correct number of names, e.g.
        ``["out_a", "out_b"]``.
    :param target_opset: ONNX opset version.  Can be an integer (default
        domain only) or a dictionary mapping domain names to opset versions.
    :param batch_dim: name of the dynamic first dimension (default:
        ``"batch"``).  Change this when the default name conflicts with
        another symbolic dimension in the same graph.
    :return: an :class:`~yobx.container.ExportArtifact` representing the traced function.

    Example::

        import numpy as np
        from yobx.xtracing import trace_numpy_to_onnx

        def my_func(X):
            return np.sqrt(np.abs(X) + 1)

        X = np.random.randn(4, 3).astype(np.float32)
        onx = trace_numpy_to_onnx(my_func, X)
    """
    if not inputs:
        raise ValueError("At least one sample input array must be provided.")

    if isinstance(target_opset, int):
        opsets: Dict[str, int] = {"": target_opset, "ai.onnx.ml": 1}
    else:
        opsets = dict(target_opset)
        if "" not in opsets:
            opsets[""] = DEFAULT_TARGET_OPSET
        if "ai.onnx.ml" not in opsets:
            opsets["ai.onnx.ml"] = 1

    if input_names is None:
        resolved_input_names: List[str] = (
            ["X"] if len(inputs) == 1 else [f"X{i}" for i in range(len(inputs))]
        )
    else:
        resolved_input_names = list(input_names)
        if len(resolved_input_names) != len(inputs):
            raise ValueError(
                f"Length mismatch: {len(inputs)} sample inputs but "
                f"input_names has {len(resolved_input_names)} elements."
            )

    resolved_output_names: List[str] = (
        list(output_names) if output_names is not None else ["output_0"]
    )

    g = GraphBuilder(opsets)

    for iname, arr in zip(resolved_input_names, inputs):
        itype = np_dtype_to_tensor_dtype(arr.dtype)
        # Make the first (batch) dimension dynamic; keep the rest static.
        shape: Tuple = (batch_dim, *arr.shape[1:])  # type: ignore[assignment]
        g.make_tensor_input(iname, itype, shape)

    trace_numpy_function(g, {}, resolved_output_names, func, resolved_input_names, name="trace")  # type: ignore

    for out_name in resolved_output_names:
        g.make_tensor_output(out_name, indexed=False, allow_untyped_output=True)

    onx = g.to_onnx(return_optimize_report=True)  # type: ignore
    return onx
