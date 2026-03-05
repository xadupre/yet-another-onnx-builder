from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
from sklearn.base import BaseEstimator
from ..helpers.onnx_helper import np_dtype_to_tensor_dtype
from ..xbuilder import GraphBuilder
from .register import get_sklearn_converter
from .sklearn_helper import get_output_names


def to_onnx(
    estimator: BaseEstimator,
    args: Tuple[Any],
    input_names: Optional[Sequence[str]] = None,
    dynamic_shapes: Optional[Tuple[Dict[int, str]]] = None,
    target_opset: int = 20,
    verbose: int = 0,
    builder_cls: Union[type, Callable] = GraphBuilder,
    extra_converters: Optional[Dict[type, Callable]] = None,
):
    """
    Converts a :epkg:`scikit-learn` estimator into ONNX.

    :param estimator: estimator
    :param args: dummy inputs
    :param dynamic_shapes: dynamic shapes
    :param target_opset: opset to use, it must be specified
    :param verbose: verbosity
    :param builder_cls: by default the graph builder is a
        :class:`yobx.xbuilder.GraphBuilder` but any builder can
        be used as long it implements the apis :ref:`builder-api`
        and :ref:`builder-api-make`
    :param extra_converters: optional mapping from estimator type to converter
        function; entries here take priority over the built-in converters and
        allow converting custom estimators that are not natively supported
    :return: onnx model
    """
    from . import register_sklearn_converters

    register_sklearn_converters()
    g = builder_cls(target_opset)

    cls = type(estimator)
    if extra_converters and cls in extra_converters:
        fct = extra_converters[cls]
    else:
        fct = get_sklearn_converter(cls)

    if input_names:
        if len(input_names) != len(args):
            raise ValueError(f"Length mismatch: {len(args)=} but input_names={input_names!r}")
    else:
        input_names = ["X"] if len(args) == 1 else [f"X{i}" for i in range(len(args))]
    for i, (name, arg) in enumerate(zip(input_names, args)):
        if dynamic_shapes:
            ds = dynamic_shapes[i]
        else:
            ds = {0: "batch"}
        shape = list(arg.shape)
        for axis, dim in ds.items():
            shape[axis] = dim
        g.make_tensor_input(name, np_dtype_to_tensor_dtype(arg.dtype), tuple(shape), device=-1)

    output_names = get_output_names(estimator)
    fct(g, {}, output_names, estimator, *input_names, name="main")

    for name in output_names:
        g.make_tensor_output(name, indexed=False, allow_untyped_output=True)
    return g.to_onnx()
