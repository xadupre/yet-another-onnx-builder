"""
Utilities to decompose an einsum equation into basic ONNX operators.

The decomposition replaces a single ``Einsum`` node with a sequence of
``Transpose``, ``Reshape``, ``MatMul``, ``Mul``, ``ReduceSum``,
``Unsqueeze``, ``Squeeze`` and ``Identity`` nodes that are equivalent
for any input shapes.  The resulting sub-graph is embedded in a
stand-alone :class:`onnx.ModelProto` so it can be inspected, optimised,
or stitched into a larger graph.

The decomposition algorithm is implemented in the private
:mod:`yobx.helpers._einsum` sub-package, which is a self-contained port
of the einsum decomposition logic from
https://github.com/sdpython/onnx-extended/tree/main/onnx_extended/tools/einsum
(MIT licence).
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import onnx
from ._einsum import decompose_einsum_equation as _decompose_einsum_equation


def decompose_einsum(
    equation: str,
    *input_shapes: Tuple[int, ...],
    dtype: Union[np.dtype, type] = np.float32,
    opset: Optional[int] = None,
    strategy: str = "numpy",
    clean: bool = True,
    verbose: bool = False,
) -> onnx.ModelProto:
    """
    Decomposes an einsum equation into a sequence of standard ONNX operators.

    Replaces the single ``Einsum`` node with primitive operations—
    ``Transpose``, ``Reshape``, ``MatMul``, ``Mul``, ``ReduceSum``,
    ``Unsqueeze``, ``Squeeze``, ``Gemm``, and ``Identity``—that are
    equivalent for any input shapes.  The result is returned as a
    stand-alone :class:`onnx.ModelProto`.

    :param equation: einsum equation string (e.g. ``"ij,jk->ik"``).
        The equation must contain an explicit output (``->``).
    :param input_shapes: optional shapes for each input operand, used as
        hints for the decomposition algorithm.  When omitted, shapes with
        all dimensions equal to ``2`` are assumed.  The shapes do **not**
        constrain the resulting ONNX model—any conforming shape works at
        runtime.
    :param dtype: numpy scalar type used for the model inputs and output,
        defaults to ``numpy.float32``.  Supported values are
        ``numpy.float32``, ``numpy.float64``, ``numpy.int32``, and
        ``numpy.int64``.
    :param opset: ONNX opset version for the produced model; defaults to
        the current ONNX opset version (capped at 18).
    :param strategy: decomposition strategy.
        Use ``"numpy"`` (default) for a fully element-wise decomposition that
        avoids any remaining ``Einsum`` call, or ``"simple"`` for a simpler
        decomposition that may retain a 2-operand ``Einsum`` internally.
    :param clean: when ``True`` (default), removes unused intermediate nodes
        from the decomposed graph.
    :param verbose: print intermediate decomposition steps.
    :return: :class:`onnx.ModelProto` whose graph computes the same result as
        ``numpy.einsum(equation, *inputs)``.

    Example: matrix multiplication::

        import numpy as np
        import onnxruntime
        from yobx.helpers.einsum_helper import decompose_einsum

        model = decompose_einsum("ij,jk->ik", (3, 4), (4, 5))
        sess = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        a = np.random.rand(3, 4).astype(np.float32)
        b = np.random.rand(4, 5).astype(np.float32)
        (result,) = sess.run(None, {"X0": a, "X1": b})
        expected = np.einsum("ij,jk->ik", a, b)
        assert np.allclose(result, expected, atol=1e-5)

    .. note::

        Equations with repeated indices in a single operand (diagonal
        operations, e.g. ``"ii->i"``) are **not** supported and will raise
        :exc:`NotImplementedError`.

    .. runpython::
        :showcode:

        import numpy as np
        from yobx.helpers.einsum_helper import decompose_einsum

        model = decompose_einsum("bij,bjk->bik", (2, 3, 4), (2, 4, 5))
        ops = [n.op_type for n in model.graph.node]
        print("ONNX node types:", ops)
    """
    n_inputs = len(equation.split("->")[0].split(","))
    input_names = [f"X{i}" for i in range(n_inputs)]

    graph = _decompose_einsum_equation(
        equation, *input_shapes, strategy=strategy, clean=clean, verbose=verbose
    )

    kwargs = {}
    if opset is not None:
        kwargs["opset"] = opset

    model: onnx.ModelProto = graph.to_onnx(
        "Z", *input_names, dtype=dtype, verbose=verbose, **kwargs
    )
    return model


def list_decomposed_nodes(
    equation: str, *input_shapes: Tuple[int, ...], verbose: bool = False
) -> List[str]:
    """
    Returns the list of ONNX operator types that result from decomposing
    *equation*.

    This is a convenience wrapper around :func:`decompose_einsum` that
    runs the decomposition and extracts the ``op_type`` attribute from
    every node in the produced graph.

    :param equation: einsum equation string.
    :param input_shapes: optional shapes for each input operand.
    :param verbose: print intermediate decomposition steps.
    :return: list of operator type strings (e.g.
        ``["Transpose", "MatMul", "ReduceSum", ...]``).

    .. runpython::
        :showcode:

        from yobx.helpers.einsum_helper import list_decomposed_nodes

        ops = list_decomposed_nodes("ij,jk->ik")
        print(ops)
    """
    model = decompose_einsum(equation, *input_shapes, verbose=verbose)
    return [node.op_type for node in model.graph.node]
