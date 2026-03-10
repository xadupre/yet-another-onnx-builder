from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from onnx import ModelProto
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from .. import DEFAULT_TARGET_OPSET
from ..container import ExtendedModelContainer
from ..helpers.onnx_helper import np_dtype_to_tensor_dtype
from ..xbuilder import GraphBuilder
from ..xbuilder.function_options import FunctionOptions
from .register import get_sklearn_converter
from .sklearn_helper import get_output_names

# Key used in the *sts* dict to propagate FunctionOptions through converters.
_FUNCTION_OPTIONS_KEY = "_sklearn_function_options"


def _default_ai_onnx_ml(main_opset: int) -> int:
    if main_opset >= 21:
        return 5
    if main_opset >= 18:
        return 3
    if main_opset >= 6:
        return 2
    return 1


def _wrap_step_as_function(
    g: GraphBuilder,
    fopts: FunctionOptions,
    estimator: BaseEstimator,
    input_names: List[str],
    output_names: List[str],
    converter: Callable,
    name: str,
) -> None:
    """
    Converts *estimator* to ONNX via *converter* and wraps the result as an
    ONNX local function registered in *g*.

    :param g: the main graph builder
    :param fopts: :class:`~yobx.xbuilder.FunctionOptions` carrying the target
        domain and other options; the ``name`` field is ignored — the function
        name is always derived from the estimator's class name
    :param estimator: the fitted estimator to convert
    :param input_names: input tensor names already present in *g*
    :param output_names: desired output tensor names to produce in *g*
    :param converter: the registered sklearn→ONNX converter function
    :param name: node-name prefix passed to the converter
    """
    # Create an isolated sub-builder for the function body.
    sub_g = GraphBuilder(g.opsets, as_function=True)

    # Use stable, collision-free internal names for the function inputs so that
    # the sub-builder's namespace is never polluted by main-graph names (which
    # could match output names and cause "name already exists" errors).
    function_input_names = [f"_in{i}_" for i in range(len(input_names))]
    for func_inp, orig_inp in zip(function_input_names, input_names):
        itype = g.get_type(orig_inp) if g.has_type(orig_inp) else 0
        ishape = g.get_shape(orig_inp) if g.has_shape(orig_inp) else None
        sub_g.make_tensor_input(func_inp, itype, ishape)

    # Determine generic sub-builder output names (independent of the main graph).
    function_output_names = list(get_output_names(estimator))

    # Run the converter inside the sub-builder (no function_options propagation).
    converter(sub_g, {}, function_output_names, estimator, *function_input_names, name=name)

    # Register the sub-builder outputs.
    for out_name in function_output_names:
        sub_g.make_tensor_output(out_name, indexed=False, allow_untyped_output=True)

    # Derive a sanitised function name from the estimator class name.
    cls_name = type(estimator).__name__.replace("-", "_").replace(".", "_")

    opts = FunctionOptions(
        name=cls_name,
        domain=fopts.domain,
        move_initializer_to_constant=fopts.move_initializer_to_constant,
        rename_allowed=True,
    )
    _new_inits, (fdomain, fname) = g.make_local_function(sub_g, opts, optimize=False)

    # When an output name is already defined in the main graph (e.g. when a
    # pipeline step's output name collides with a previously produced name)
    # we write to a fresh temporary name first and then rename via Identity.
    actual_output_names = []
    renames: List[Tuple[str, str]] = []
    for out in output_names:
        if g.has_name(out):
            tmp = g.unique_name(f"_tmp_{cls_name}_out_")
            actual_output_names.append(tmp)
            renames.append((tmp, out))
        else:
            actual_output_names.append(out)

    # Call the local function in the main graph.
    g.make_node(fname, list(input_names), actual_output_names, domain=fdomain, name=name)

    # Apply any needed Identity renames.
    for tmp, final in renames:
        g.make_node("Identity", [tmp], [final], name=f"{name}_rename")

    # Propagate type/shape metadata from the sub-builder to the main graph.
    for func_out, actual, desired in zip(function_output_names, actual_output_names, output_names):
        for out in (actual, desired):
            if sub_g.has_type(func_out):
                g.set_type(out, sub_g.get_type(func_out))
            if sub_g.has_shape(func_out):
                g.set_shape(out, sub_g.get_shape(func_out))
            elif sub_g.has_rank(func_out):
                g.set_rank(out, sub_g.get_rank(func_out))
            if sub_g.has_device(func_out):
                g.set_device(out, sub_g.get_device(func_out))


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
    function_options: Union[bool, FunctionOptions] = False,
) -> Union[ModelProto, ExtendedModelContainer]:
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
    :param function_options: when a :class:`~yobx.xbuilder.FunctionOptions`
        is provided every non-container estimator is exported as a separate ONNX
        local function.  :class:`~sklearn.pipeline.Pipeline` and
        :class:`~sklearn.compose.ColumnTransformer` are treated as
        orchestrators — their individual steps/sub-transformers are each
        wrapped as a function instead of the container itself.  The
        ``name`` field of the provided :class:`~yobx.xbuilder.FunctionOptions`
        is used as a template; the actual function name for each step is
        always derived from the estimator's class name.  Pass ``False``
        (the default) to disable function wrapping and produce a flat graph.
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

    # Build the sts dict; if function_options is set, propagate it so that
    # Pipeline / ColumnTransformer converters can wrap their steps.
    sts: Dict[str, Any] = {}
    if function_options is not False:
        sts[_FUNCTION_OPTIONS_KEY] = function_options

    output_names = list(get_output_names(estimator))

    is_container = isinstance(estimator, (Pipeline, ColumnTransformer))

    if function_options is not False and not is_container:
        # Wrap the single top-level estimator as a local function.
        _wrap_step_as_function(
            g,
            function_options,
            estimator,
            list(input_names),
            output_names,
            fct,
            name="main",
        )
    else:
        fct(g, sts, output_names, estimator, *input_names, name="main")

    for name in output_names:
        g.make_tensor_output(name, indexed=False, allow_untyped_output=True)
    # When local functions are requested we must NOT inline them; pass inline=False
    # so the function bodies are preserved in the returned ModelProto.
    keep_functions = function_options is not False
    return g.to_onnx(
        large_model=large_model,
        external_threshold=external_threshold,
        inline=not keep_functions,
    )
