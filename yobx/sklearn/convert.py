from typing import Any, Dict, Optional, Sequence, Tuple
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from .register import get_sklearn_converter
from ..helpers.onnx_helper import np_dtype_to_tensor_dtype
from ..xbuilder import GraphBuilder


def get_output_names(estimator):
    if hasattr(estimator, "get_feature_names_out"):
        return estimator.get_feature_names_out()
    if is_classifier(estimator):
        return ["labels", "probabilities"]
    if is_regressor(estimator):
        return ["predictions"]
    return ["Y"]


def to_onnx(
    estimator: BaseEstimator,
    args: Tuple[Any],
    input_names: Optional[Sequence[str]] = None,
    dynamic_shapes: Optional[Tuple[Dict[int, str]]] = None,
    target_opset: int = 20,
    verbose: int = 0,
):
    """
    Converts a :epkg:`scikit-learn` estimatior into ONNX.

    :param estimator: estimator
    :param args: dummy inputs
    :param dynamic_shapes: dynamic shapes
    :param target_opset: opset to use, it mush be specified
    :param verbose: verbosity
    :return: onnx model
    """
    from . import register_sklearn_converters

    register_sklearn_converters()
    g = GraphBuilder(target_opset)
    fct = get_sklearn_converter(type(estimator))

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
        for i, dim in ds.items():
            shape[i] = dim
        g.make_tensor_input(name, np_dtype_to_tensor_dtype(arg.dtype), shape)

    output_names = get_output_names(estimator)
    fct(g, {}, output_names, estimator, *input_names, name="main")

    for name in output_names:
        g.make_tensor_output(name, indexed=False, allow_untyped_output=True)
    return g.to_onnx()
