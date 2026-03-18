"""
Converter for :class:`sklearn.preprocessing.FunctionTransformer`.

The converter traces the user-supplied function with a
:class:`~yobx.xtracing.NumpyArray` proxy so that every numpy operation is
recorded as an ONNX node directly in the graph being built.

When ``func`` is ``None`` (identity transformer) an ``Identity`` node is
emitted instead.
"""

from typing import Dict, List
from sklearn.preprocessing import FunctionTransformer

from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from ...xtracing.numpy_array import NumpyArray


@register_sklearn_converter(FunctionTransformer)
def sklearn_function_transformer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: FunctionTransformer,
    X: str,
    name: str = "func_transformer",
) -> str:
    """
    Converts a :class:`sklearn.preprocessing.FunctionTransformer` into ONNX.

    The function stored in ``estimator.func`` is traced using
    :class:`~yobx.xtracing.NumpyArray` proxies: every arithmetic operation,
    ufunc, or reduction is recorded as an ONNX node in *g*.  If ``func`` is
    ``None`` the transformer is an identity and a single ``Identity`` node is
    emitted.

    :param g: graph builder to add nodes to
    :param sts: shapes defined by scikit-learn (may be empty)
    :param outputs: desired output tensor names
    :param estimator: a :class:`~sklearn.preprocessing.FunctionTransformer`
        instance
    :param X: input tensor name
    :param name: prefix for added node names
    :return: output tensor name
    """
    assert isinstance(estimator, FunctionTransformer), (
        f"Unexpected type {type(estimator)} for estimator."
    )

    # ----------------------------------------------------------------
    # Identity transformer (func=None)
    # ----------------------------------------------------------------
    if estimator.func is None:
        res = g.op.Identity(X, outputs=outputs, name=name)
        assert isinstance(res, str)
        if not sts:
            g.set_type_shape_unary_op(res, X)
        return res

    # ----------------------------------------------------------------
    # Traced transformer
    # ----------------------------------------------------------------
    # Determine the numpy dtype from the ONNX element type of X so that
    # scalar constants created during tracing use the right dtype.
    if g.has_type(X):
        itype = g.get_type(X)
        dtype = tensor_dtype_to_np_dtype(itype)
    else:
        dtype = None

    # Wrap the input tensor as a NumpyArray proxy and call the function.
    proxy = NumpyArray(X, g, dtype=dtype)
    kw_args = estimator.kw_args if estimator.kw_args is not None else {}
    result = estimator.func(proxy, **kw_args)

    # Normalise result to a single NumpyArray (FunctionTransformer.transform
    # returns a single array).
    if isinstance(result, (list, tuple)):
        result = result[0]
    if not isinstance(result, NumpyArray):
        raise TypeError(
            f"FunctionTransformer.func must return a NumpyArray when traced; "
            f"got {type(result).__name__!r}.  Ensure the function uses only "
            "numpy operations that are supported by the ONNX tracing mechanism."
        )

    # Rename the result tensor to the expected output name via an Identity node.
    res = g.op.Identity(result.name, outputs=outputs, name=name)
    assert isinstance(res, str)
    if not sts:
        g.set_type_shape_unary_op(res, X)
    return res
