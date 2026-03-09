from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from .. import DEFAULT_TARGET_OPSET
from ..helpers.onnx_helper import np_dtype_to_tensor_dtype
from ..xbuilder import GraphBuilder
from .register import get_sklearn_converter
from .sklearn_helper import get_output_names


def _default_ai_onnx_ml(main_opset: int) -> int:
    if main_opset >= 21:
        return 5
    if main_opset >= 18:
        return 3
    if main_opset >= 6:
        return 2
    return 1


def to_onnx(
    estimator: BaseEstimator,
    args: Tuple[Any],
    input_names: Optional[Sequence[str]] = None,
    dynamic_shapes: Optional[Tuple[Dict[int, str]]] = None,
    target_opset: Union[int, Dict[str, int]] = DEFAULT_TARGET_OPSET,
    verbose: int = 0,
    builder_cls: Union[type, Callable] = GraphBuilder,
    extra_converters: Optional[Dict[type, Callable]] = None,
    large_model: bool = False,
    external_threshold: int = 1024,
):
    """
    Converts a :epkg:`scikit-learn` estimator into ONNX.
    By default, the first dimension is considered as dynamic,
    the others are static.

    :param estimator: estimator
    :param args: dummy inputs
    :param dynamic_shapes: dynamic shapes, if not specified, the first dimension
        is dynamic, the others are static
    :param target_opset: opset to use; either an integer for the default domain
        (``""``), or a dictionary mapping domain names to opset versions,
        e.g. ``{"": 20, "ai.onnx.ml": 5}``.  When ``"ai.onnx.ml"`` is set to
        ``5`` the converter emits the unified ``TreeEnsemble`` operator
        introduced in that opset instead of the older per-task operators.
    :param verbose: verbosity
    :param builder_cls: by default the graph builder is a
        :class:`yobx.xbuilder.GraphBuilder` but any builder can
        be used as long it implements the apis :ref:`builder-api`
        and :ref:`builder-api-make`
    :param extra_converters: optional mapping from estimator type to converter
        function; entries here take priority over the built-in converters and
        allow converting custom estimators that are not natively supported
    :param large_model: if True returns a
        :class:`onnx.model_container.ModelContainer`, which lets the user
        decide later whether weights should be embedded in the model or saved
        as external data
    :param external_threshold: if ``large_model`` is True, every tensor whose
        element count exceeds this threshold is stored as external data
    :return: onnx model or :class:`onnx.model_container.ModelContainer`
        when *large_model* is True
    """
    check_is_fitted(
        estimator,
        attributes=["transform", "predict"],
        all_or_any=any,
        msg=(
            "This %(name)s instance has neither a 'transform' nor a 'predict' method "
            "and cannot be converted to ONNX."
        ),
    )
    if isinstance(target_opset, int):
        dict_target_opset = {"": target_opset, "ai.onnx.ml": _default_ai_onnx_ml(target_opset)}
    else:
        if not isinstance(target_opset, dict):
            raise TypeError(f"target_opset must be a dictionary or an integer not {target_opset}")
        dict_target_opset = target_opset.copy()
        if "" not in dict_target_opset:
            dict_target_opset[""] = 21
        if "ai.onnx.ml" not in dict_target_opset:
            dict_target_opset["ai.onnx.ml"] = _default_ai_onnx_ml(dict_target_opset[""])

    from . import register_sklearn_converters

    register_sklearn_converters()
    g = builder_cls(dict_target_opset)

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
    return g.to_onnx(large_model=large_model, external_threshold=external_threshold)
