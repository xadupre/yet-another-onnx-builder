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
from ...xtracing.tracing import trace_numpy_function


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
    :func:`~yobx.xtracing.trace_numpy_function`: every arithmetic operation,
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
    # Traced transformer — delegate to trace_numpy_function which follows
    # the same API convention as other converters.
    # ----------------------------------------------------------------
    kw_args = estimator.kw_args
    res = trace_numpy_function(g, estimator.func, [X], outputs, name=name, kw_args=kw_args)
    assert isinstance(res, str)
    if not sts:
        g.set_type_shape_unary_op(res, X)
    return res
