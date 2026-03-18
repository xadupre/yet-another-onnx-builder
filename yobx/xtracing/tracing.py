"""
Standalone tracing of numpy functions to ONNX models.

:func:`trace_numpy_to_onnx` executes a Python function that uses numpy
operations with :class:`~yobx.xtracing.numpy_array.NumpyArray` proxies,
records every operation as an ONNX node, and returns the resulting
:class:`onnx.ModelProto`.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from onnx import ModelProto

from .. import DEFAULT_TARGET_OPSET
from ..helpers.onnx_helper import np_dtype_to_tensor_dtype
from ..xbuilder import GraphBuilder
from .numpy_array import NumpyArray

#: Name used for the dynamic (batch) first dimension when none is specified.
_BATCH_DIM = "batch"


def trace_numpy_to_onnx(
    func: Callable,
    *inputs: np.ndarray,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    target_opset: Union[int, Dict[str, int]] = DEFAULT_TARGET_OPSET,
    batch_dim: str = _BATCH_DIM,
) -> ModelProto:
    """
    Trace a numpy function and return the equivalent ONNX model.

    The function *func* is called with :class:`~yobx.xtracing.numpy_array.NumpyArray`
    proxies instead of real numpy arrays.  Every arithmetic operation, ufunc
    call, or reduction is recorded as an ONNX node in an internal
    :class:`~yobx.xbuilder.GraphBuilder`, which is then exported to an
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
        outputs.  Defaults to ``["output_0"]`` (or ``["output_0", "output_1",
        …]`` for multiple outputs).
    :param target_opset: ONNX opset version.  Can be an integer (default
        domain only) or a dictionary mapping domain names to opset versions.
    :param batch_dim: name of the dynamic first dimension (default:
        ``"batch"``).  Change this when the default name conflicts with
        another symbolic dimension in the same graph.
    :return: an :class:`onnx.ModelProto` representing the traced function.

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

    # ------------------------------------------------------------------
    # Build the opset dictionary.
    # ------------------------------------------------------------------
    if isinstance(target_opset, int):
        opsets: Dict[str, int] = {"": target_opset, "ai.onnx.ml": 1}
    else:
        opsets = dict(target_opset)
        if "" not in opsets:
            opsets[""] = DEFAULT_TARGET_OPSET
        if "ai.onnx.ml" not in opsets:
            opsets["ai.onnx.ml"] = 1

    # ------------------------------------------------------------------
    # Resolve input names.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Create the graph builder and register graph inputs.
    # ------------------------------------------------------------------
    g = GraphBuilder(opsets)

    proxy_inputs: List[NumpyArray] = []
    for iname, arr in zip(resolved_input_names, inputs):
        itype = np_dtype_to_tensor_dtype(arr.dtype)
        # Make the first (batch) dimension dynamic; keep the rest static.
        shape: Tuple = (batch_dim,) + arr.shape[1:]  # type: ignore[assignment]
        g.make_tensor_input(iname, itype, shape)
        proxy_inputs.append(NumpyArray(iname, g, dtype=arr.dtype, shape=arr.shape))

    # ------------------------------------------------------------------
    # Run the function with proxy inputs to capture the ONNX graph.
    # ------------------------------------------------------------------
    result = func(*proxy_inputs)

    # ------------------------------------------------------------------
    # Normalise output to a list of NumpyArray objects.
    # ------------------------------------------------------------------
    if isinstance(result, NumpyArray):
        raw_outputs: List[NumpyArray] = [result]
    elif isinstance(result, (list, tuple)):
        raw_outputs = list(result)
    else:
        raise TypeError(
            f"trace_numpy_to_onnx: function returned {type(result).__name__!r}; "
            "expected a NumpyArray or a list/tuple of NumpyArrays."
        )

    # ------------------------------------------------------------------
    # Resolve output names.
    # ------------------------------------------------------------------
    if output_names is None:
        resolved_output_names: List[str] = (
            ["output_0"]
            if len(raw_outputs) == 1
            else [f"output_{i}" for i in range(len(raw_outputs))]
        )
    else:
        resolved_output_names = list(output_names)
        if len(resolved_output_names) != len(raw_outputs):
            raise ValueError(
                f"Length mismatch: function returned {len(raw_outputs)} outputs but "
                f"output_names has {len(resolved_output_names)} elements."
            )

    # ------------------------------------------------------------------
    # Register graph outputs (rename via Identity nodes if necessary).
    # ------------------------------------------------------------------
    for out_arr, out_name in zip(raw_outputs, resolved_output_names):
        if not isinstance(out_arr, NumpyArray):
            raise TypeError(
                f"trace_numpy_to_onnx: expected NumpyArray output, got {type(out_arr).__name__!r}."
            )
        # Rename to the desired output name via an Identity node so that the
        # graph output tensor always has the expected name.
        g.op.Identity(out_arr.name, outputs=[out_name], name=g.unique_name("output_id"))
        g.make_tensor_output(out_name, indexed=False, allow_untyped_output=True)

    # ------------------------------------------------------------------
    # Export.
    # ------------------------------------------------------------------
    onx, _ = g.to_onnx(return_optimize_report=True)
    return onx  # type: ignore[return-value]
