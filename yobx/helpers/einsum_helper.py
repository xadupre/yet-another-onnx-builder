"""
Utilities to decompose an einsum equation into basic ONNX operators.

The decomposition replaces a single ``Einsum`` node with a sequence of
``Transpose``, ``Reshape``, ``MatMul``, ``Mul``, ``ReduceSum``,
``Unsqueeze``, ``Squeeze`` and ``Identity`` nodes that are equivalent
for any input shapes.  The resulting sub-graph is embedded in a
stand-alone :class:`onnx.ModelProto` so it can be inspected, optimised,
or stitched into a larger graph.
"""

from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx
from ._einsum import decompose_einsum_equation
from ._einsum.einsum_2_onnx import decompose_einsum_2inputs as _decompose_einsum_2inputs
from .onnx_helper import np_dtype_to_tensor_dtype


def decompose_einsum(
    equation: str,
    *input_shapes: Tuple[Union[int, str, None], ...],
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
    :param input_shapes: optional shapes for each input operand.
        Each element of the tuple may be an integer (concrete size),
        a string (symbolic dimension name, e.g. ``"batch"``), or
        ``None`` (unknown dimension).  When omitted, shapes with all
        dimensions equal to ``2`` are used internally and the produced
        ONNX model will have fully dynamic input shapes.  When provided,
        the shapes are reflected in the ``value_info`` of the returned
        model—concrete integers become fixed-size dimensions, strings
        become named symbolic dimensions, and ``None`` values remain
        dynamic.
    :param dtype: numpy scalar type used for the model inputs and output,
        defaults to ``numpy.float32``.  Supported values are
        ``numpy.float32``, ``numpy.float64``, ``numpy.int32``, and
        ``numpy.int64``.
    :param opset: ONNX opset version for the produced model; defaults to
        the current ONNX opset version (capped at 18).
    :param strategy: decomposition strategy.
        Use ``"numpy"`` (default) for a fully element-wise decomposition that
        avoids any remaining ``Einsum`` call.  ``"simple"`` is supported for
        numpy evaluation but cannot be converted to ONNX.
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

    .. plot::

        from yobx.doc import plot_dot
        from yobx.helpers.einsum_helper import decompose_einsum

        model = decompose_einsum("bij,bjk->bik", (2, 3, 4), (2, 4, 5))
        plot_dot(model)
    """
    n_inputs = len(equation.split("->")[0].split(","))
    input_names = [f"X{i}" for i in range(n_inputs)]

    # The decomposition algorithm only uses the rank (number of dimensions),
    # not the actual dim values, so convert any symbolic dims to concrete ints.
    concrete_shapes: tuple = (
        tuple(tuple(d if isinstance(d, int) else 2 for d in sh) for sh in input_shapes)
        if input_shapes
        else ()
    )

    graph = decompose_einsum_equation(
        equation, *concrete_shapes, strategy=strategy, clean=clean, verbose=verbose
    )

    kwargs: dict = {}
    if opset is not None:
        kwargs["opset"] = opset

    # When caller provides shapes (possibly with symbolic string dims), forward
    # them to to_onnx via the (name, (elem_type, shape)) tuple format so that
    # the produced value_info carries the correct shape information.
    if input_shapes:
        proto = np_dtype_to_tensor_dtype(np.dtype(dtype))
        shaped_inputs = [(name, (proto, list(sh))) for name, sh in zip(input_names, input_shapes)]
        model: onnx.ModelProto = graph.to_onnx(
            "Z", *shaped_inputs, dtype=dtype, verbose=verbose, **kwargs
        )
    else:
        model = graph.to_onnx("Z", *input_names, dtype=dtype, verbose=verbose, **kwargs)
    # Optimize: apply GraphBuilder pattern rewrites, identity removal, and
    # constant folding.  Import deferred to avoid a circular import with
    # yobx.xbuilder.
    from yobx.xbuilder.graph_builder import GraphBuilder

    gb = GraphBuilder(model, verbose=0)
    gb.optimize()
    artifact = gb.to_onnx(optimize=False)
    opt_model = artifact.get_proto()
    # GraphBuilder embeds extra metadata_props (e.g. statistics) that ORT
    # does not expect.  Stripping them and doing an onnx round-trip normalises
    # the protobuf so ORT can load it directly from SerializeToString().
    del opt_model.metadata_props[:]
    return onnx.load_from_string(opt_model.SerializeToString())


def list_decomposed_nodes(
    equation: str, *input_shapes: Tuple[Union[int, str, None], ...], verbose: bool = False
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


def decompose_einsum_2inputs(
    equation: str,
    shape0: Optional[Sequence[Union[int, str, None]]] = None,
    shape1: Optional[Sequence[Union[int, str, None]]] = None,
    dtype: Union[np.dtype, type] = np.float32,
    opset: Optional[int] = None,
) -> onnx.ModelProto:
    """
    Decomposes a 2-input einsum equation directly into basic ONNX operators.

    This is a completely independent implementation — it does **not** use the
    :class:`~yobx.helpers._einsum.EinsumSubOp` /
    :class:`~yobx.helpers._einsum.GraphEinsumSubOp` framework.  It analyses
    the equation, classifies every index letter into one of four roles
    (*batch*, *contract*, *left*, *right*), and emits a fixed sequence of
    ``Transpose``, ``Reshape``, ``MatMul``, ``Reshape``, ``Transpose`` nodes
    that compute the result for *any* input shape.

    :param equation: einsum equation string with exactly two inputs and an
        explicit output, e.g. ``"ij,jk->ik"`` or ``"bij,bjk->bik"``.
    :param shape0: optional shape of the first input.  Each element may be
        an integer (rank hint — the ONNX graph input is made fully dynamic),
        a string (symbolic name, e.g. ``"batch"``), or ``None`` (dynamic
        dimension).  Use string dims when you need symbolic FLOPs estimation.
    :param shape1: optional shape of the second input (same convention).
    :param dtype: numpy scalar type for the model inputs and output
        (default ``numpy.float32``).
    :param opset: ONNX opset version; defaults to the current ONNX opset
        capped at 18.
    :return: :class:`onnx.ModelProto` that computes
        ``numpy.einsum(equation, X0, X1)``.
    :raises ValueError: if *equation* does not have exactly two inputs.

    Example::

        import numpy as np
        import onnxruntime
        from yobx.helpers.einsum_helper import decompose_einsum_2inputs

        model = decompose_einsum_2inputs("bij,bjk->bik", (2, 3, 4), (2, 4, 5))
        sess = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        a = np.random.rand(2, 3, 4).astype(np.float32)
        b = np.random.rand(2, 4, 5).astype(np.float32)
        (result,) = sess.run(None, {"X0": a, "X1": b})
        assert np.allclose(result, np.einsum("bij,bjk->bik", a, b), atol=1e-5)
    """
    dtype_map = {
        np.float32: onnx.TensorProto.FLOAT,
        np.float64: onnx.TensorProto.DOUBLE,
        np.int32: onnx.TensorProto.INT32,
        np.int64: onnx.TensorProto.INT64,
    }
    dtype_key = np.dtype(dtype).type
    onnx_dtype = dtype_map.get(dtype_key, onnx.TensorProto.FLOAT)
    return _decompose_einsum_2inputs(
        equation, shape0=shape0, shape1=shape1, dtype=onnx_dtype, opset=opset
    )
