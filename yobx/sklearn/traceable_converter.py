import inspect
from typing import Dict, List, Optional, Tuple, Union
from ..typing import GraphBuilderExtendedProtocol
from ..helpers.onnx_helper import tensor_dtype_to_np_dtype
from .sklearn_helper import TraceableTransformerMixin


def build_traceable_inputs_from_inputs(
    g: GraphBuilderExtendedProtocol, estimator: TraceableTransformerMixin, *inputs: str
) -> Union[Tuple["NumpyArray", ...], Tuple["TracedDataFrame", ...]]:  # noqa: F821
    assert all(
        g.has_type(x) for x in inputs
    ), f"Some times are missing in {inputs!r}{g.get_debug_msg()}"
    assert all(
        g.has_shape(x) for x in inputs
    ), f"Some shapes are missing in {inputs!r}{g.get_debug_msg()}"
    shapes = [g.get_shape(x) for x in inputs]
    unique = set(shapes)
    assert len(unique) == 1, (
        f"Unique shapes are {shapes} for {inputs=}, estimator is {estimator}. "
        f"This is not yet handled.{g.get_debug_msg()}"
    )
    shape = shapes.pop()
    assert len(shape) == 2, f"unique shape={shape} is not 2D{g.get_debug_msg()}"
    assert hasattr(
        estimator, "transform"
    ), f"Missing method transform from {estimator}{g.get_debug_msg()}"
    sig = inspect.signature(estimator.transform)
    assert len(sig.parameters) == 1, (
        f"Unexpected number of parameters {list(sig.parameters)}, "
        "not implemented right now{g.get_debug_msg()}"
    )
    if len(inputs) == 1:
        # numpy arrays
        from ..xtracing.numpy_array import NumpyArray

        return (
            NumpyArray(
                inputs[0],
                g,
                tensor_dtype_to_np_dtype(g.get_type(inputs[0])),
                g.get_shape(inputs[0]),
            ),
        )

    assert (
        shape[1] == 1
    ), f"Unexpected second dimension for unique shape={shape}{g.get_debug_msg()}"

    # dataframe
    from ..sql import TracedDataFrame
    from ..sql.parse import ColumnRef

    return (TracedDataFrame(dict(zip(inputs, [ColumnRef(i) for i in inputs]))),)


def sklearn_traceable_converter(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: Optional[List[str]],
    estimator: TraceableTransformerMixin,
    *inputs: str,
    name: str = "traceable",
) -> List[str]:
    """
    The function assumes the estimator processes dataframe or numpy arrays.
    For dataframe, the function will group in a single dataframes all
    columns sharing the same dimensions ('N', 1).
    Then it tries to match the signature.
    """
    args = build_traceable_inputs_from_inputs(g, estimator, *inputs)

    from ..sql import trace_dataframe, TracedDataFrame
    from ..sql.sql_convert import parsed_query_to_onnx_graph
    from ..xtracing.numpy_array import NumpyArray

    if isinstance(args[0], TracedDataFrame):
        dtypes = dict(zip(inputs, [tensor_dtype_to_np_dtype(g.get_type(i)) for i in inputs]))
        pq = trace_dataframe(estimator.transform, dtypes)
        out_names = parsed_query_to_onnx_graph(
            g, sts, list(outputs) if outputs else None, pq, dtypes, _finalize=False
        )
        return out_names[0] if len(out_names) == 1 else tuple(out_names)

    if isinstance(args[0], NumpyArray):
        from ..xtracing import trace_numpy_function

        res = trace_numpy_function(
            g, sts, list(outputs) if outputs else None, estimator.transform, inputs, name=name
        )
        return tuple(n.name for n in res) if isinstance(res, tuple) else res.name

    raise NotImplementedError(f"Unable to trace estimator {estimator}{g.get_debug_msg()}")
