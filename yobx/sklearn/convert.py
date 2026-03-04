from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
from sklearn.base import BaseEstimator
from ..helpers.onnx_helper import np_dtype_to_tensor_dtype
from ..xbuilder import GraphBuilder
from .register import get_sklearn_converter
from .sklearn_helper import get_output_names


def convert_standard_scaler(
    estimator,
    X: np.ndarray,
    **kwargs,
):
    """
    Converts a fitted :class:`sklearn.preprocessing.StandardScaler` into ONNX.

    :param estimator: a fitted ``StandardScaler``
    :param X: sample input array (used to infer dtype and shape)
    :param kwargs: extra keyword arguments forwarded to :func:`to_onnx`
    :return: onnx model
    """
    return to_onnx(estimator, (X,), **kwargs)


def convert_logistic_regression(
    estimator,
    X: np.ndarray,
    **kwargs,
):
    """
    Converts a fitted :class:`sklearn.linear_model.LogisticRegression` into ONNX.

    :param estimator: a fitted ``LogisticRegression``
    :param X: sample input array (used to infer dtype and shape)
    :param kwargs: extra keyword arguments forwarded to :func:`to_onnx`
    :return: onnx model
    """
    return to_onnx(estimator, (X,), **kwargs)


def convert_pipeline(
    estimator,
    X: np.ndarray,
    **kwargs,
):
    """
    Converts a fitted :class:`sklearn.pipeline.Pipeline` into ONNX.

    :param estimator: a fitted ``Pipeline``
    :param X: sample input array (used to infer dtype and shape)
    :param kwargs: extra keyword arguments forwarded to :func:`to_onnx`
    :return: onnx model
    """
    return to_onnx(estimator, (X,), **kwargs)


def to_onnx(
    estimator: BaseEstimator,
    args: Tuple[Any],
    input_names: Optional[Sequence[str]] = None,
    dynamic_shapes: Optional[Tuple[Dict[int, str]]] = None,
    target_opset: int = 20,
    verbose: int = 0,
):
    """
    Converts a :epkg:`scikit-learn` estimator into ONNX.

    :param estimator: estimator
    :param args: dummy inputs
    :param dynamic_shapes: dynamic shapes
    :param target_opset: opset to use, it must be specified
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
        for axis, dim in ds.items():
            shape[axis] = dim
        g.make_tensor_input(name, np_dtype_to_tensor_dtype(arg.dtype), shape, device=-1)

    output_names = get_output_names(estimator)
    fct(g, {}, output_names, estimator, *input_names, name="main")

    for name in output_names:
        g.make_tensor_output(name, indexed=False, allow_untyped_output=True)
    return g.to_onnx()
