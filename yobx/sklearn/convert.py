from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils.validation import check_is_fitted
from .. import DEFAULT_TARGET_OPSET
from ..typing import ConvertOptionsProtocol
from ..container import ExportArtifact
from ..xbuilder import GraphBuilder, OptimizationOptions
from ..xbuilder.function_options import FunctionOptions
from ..helpers.to_onnx_helper import register_inputs
from .register import get_sklearn_converter, sklearn_exportable_methods
from .sklearn_helper import get_output_names


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
) -> Union[str, Tuple[str, ...]]:
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
    function_output_names = get_output_names(estimator, g.convert_options, name)
    function_output_names = (
        [sub_g.unique_name(n) for n in function_output_names] if function_input_names else None
    )

    # Run the converter inside the sub-builder (no function_options propagation).
    out_names = converter(
        sub_g, {}, function_output_names, estimator, *function_input_names, name=name
    )
    if function_input_names is None:
        function_output_names = out_names
    else:
        function_output_names = tuple(function_output_names)

    # Register the sub-builder outputs.
    assert (
        function_output_names
    ), f"No output found for convert {converter} applied to {estimator}{g.get_debug_msg()}"
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
    for func_out, actual, desired in zip(
        function_output_names, actual_output_names, output_names
    ):
        for out in (actual, desired):
            if sub_g.has_type(func_out):
                g.set_type(out, sub_g.get_type(func_out))
            if sub_g.has_shape(func_out):
                g.set_shape(out, sub_g.get_shape(func_out))
            elif sub_g.has_rank(func_out):
                g.set_rank(out, sub_g.get_rank(func_out))
            if sub_g.has_device(func_out):
                g.set_device(out, sub_g.get_device(func_out))
    return function_output_names if len(function_output_names) > 1 else function_output_names[0]


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
    function_options: Optional[FunctionOptions] = None,
    convert_options: Optional[ConvertOptionsProtocol] = None,
    filename: Optional[str] = None,
) -> ExportArtifact:
    """
    Converts a :epkg:`scikit-learn` estimator into ONNX.
    By default, the first dimension is considered as dynamic,
    the others are static.

    :param estimator: estimator
    :param args: dummy inputs; each element may be a numpy array, a
        :class:`pandas.DataFrame`, an
        :class:`onnx.ValueInfoProto` that explicitly describes the input
        tensor's name, element type and shape, **or** a
        ``(name, dtype, shape)`` tuple.  A :class:`~pandas.DataFrame` is
        expanded column-by-column: each column is registered as a separate
        1-D ONNX graph input named after the column, and an
        ``Unsqueeze`` + ``Concat`` node sequence assembles them back into a
        2-D matrix ``(batch, n_cols)`` that is passed to the converter.  When
        a :class:`~onnx.ValueInfoProto` or a ``(name, dtype, shape)`` tuple is
        provided no actual data is required, and the ``dynamic_shapes``
        parameter is ignored for that input (the shape is taken directly from
        the descriptor).  The ``(name, dtype, shape)`` tuple format uses a
        plain string for the name, a numpy dtype (or scalar-type class such
        as ``np.float32``) for the element type, and a sequence of ints
        and/or strings for the shape (strings denote symbolic / dynamic
        dimensions).  Example::

            to_onnx(estimator, (('x', np.float32, ('N', 4)),))
    :param dynamic_shapes: dynamic shapes, if not specified, the first dimension
        is dynamic, the others are static
    :param target_opset: opset to use; either an integer for the default domain
        (``""``), or a dictionary mapping domain names to opset versions,
        e.g. ``{"": 20, "ai.onnx.ml": 5}``.  When ``"ai.onnx.ml"`` is set to
        ``5`` the converter emits the unified ``TreeEnsemble`` operator
        introduced in that opset instead of the older per-task operators.
        If it includes ``{'com.microsoft': 1}``, the converted model
        may include optimized kernels specific to :epkg:`onnxruntime`.
    :param verbose: verbosity
    :param builder_cls: by default the graph builder is a
        :class:`yobx.xbuilder.GraphBuilder` but any builder can
        be used as long it implements the apis :ref:`builder-api`
        and :ref:`builder-api-make`
    :param extra_converters: optional mapping from estimator type to converter
        function; entries here take priority over the built-in converters and
        allow converting custom estimators that are not natively supported
    :param large_model: if True the returned :class:`~yobx.container.ExportArtifact`
        has its :attr:`~yobx.container.ExportArtifact.container` attribute set to
        an :class:`~yobx.container.ExtendedModelContainer`, which lets the user
        decide later whether weights should be embedded in the model or saved
        as external data
    :param external_threshold: if ``large_model`` is True, every tensor whose
        element count exceeds this threshold is stored as external data
    :param function_options: when a :class:`~yobx.xbuilder.FunctionOptions`
        is provided every non-container estimator is exported as a separate ONNX
        local function.  :class:`~sklearn.pipeline.Pipeline` and
        :class:`~sklearn.compose.ColumnTransformer` are treated as
        orchestrators — their individual steps/sub-transformers are each
        wrapped as a function instead of the container itself.  Function
        names for each step are always derived from the estimator's class
        name; the ``name`` field of the provided
        :class:`~yobx.xbuilder.FunctionOptions` is not used by this helper
        to customize function naming.  Pass ``None`` (the default) to disable
        function wrapping and produce a flat graph.
        when *large_model* is True
    :param convert_options: see :class:`yobx.sklearn.ConvertOptions`
    :param filename: if set, the exported ONNX model is saved to this path and
        the :class:`~yobx.container.ExportReport` is written as a companion
        Excel file (same base name with ``.xlsx`` extension).
    :return: :class:`~yobx.container.ExportArtifact` wrapping the exported
        ONNX proto together with an :class:`~yobx.container.ExportReport`.

    .. note::

        `scikit-learn==1.8` is more strict with computation types and
        the number of discrepancies is reduced. Switch to float32 in a matrix
        multiplication when the order of magnitude of the coefficient is quite
        large usually introduces discrepancies. That is often the case when
        a matrix is the inverse of another one.
        See :ref:`l-plot-sklearn-pls-float32`.

    Example::

        import numpy as np
        from sklearn.linear_model import LinearRegression
        from yobx.sklearn import to_onnx

        X = np.random.randn(10, 3).astype(np.float32)
        y = X @ np.array([1.0, 2.0, 3.0], dtype=np.float32)
        reg = LinearRegression().fit(X, y)

        artifact = to_onnx(reg, (X,))
        # Access the raw proto:
        proto = artifact.proto
        # Save to disk:
        artifact.save("model.onnx")
    """
    check_is_fitted(
        estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator,
        attributes=sklearn_exportable_methods(),
        all_or_any=any,
        msg=(
            "This %(name)s instance has neither a 'transform', 'predict', "
            "'mahalanobis', nor 'score_samples' method and cannot be converted to ONNX."
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

    kwargs = (
        dict(optimization_options=OptimizationOptions(patterns="default+onnxruntime"))
        if "com.microsoft" in dict_target_opset
        else {}
    )
    if convert_options:
        kwargs["convert_options"] = convert_options  # type: ignore

    if verbose:
        if issubclass(builder_cls, GraphBuilder):  # type: ignore
            kwargs["verbose"] = verbose  # type: ignore

        from ..helpers import string_type

        print(f"[yobx.sklearn.to_onnx] export with builder {builder_cls!r})")
        print(f"[yobx.sklearn.to_onnx] {dict_target_opset=}")
        print(f"[yobx.sklearn.to_onnx] args={string_type(args, with_shape=True)}")
        if extra_converters:
            print(f"[yobx.sklearn.to_onnx] with {len(extra_converters)} extra converters")

    g = builder_cls(dict_target_opset, **kwargs)

    cls = type(estimator)
    if extra_converters and cls in extra_converters:
        fct = extra_converters[cls]
    else:
        fct = get_sklearn_converter(cls)

    input_names = register_inputs(g, args, input_names, dynamic_shapes)

    # Build the sts dict (shared state for converters). function_options, if set,
    # is passed explicitly to container converters below.
    sts: Dict[str, Any] = {}
    top_level_node_name = (
        f"main__{estimator.steps[-1][0]}" if isinstance(estimator, Pipeline) else "main"
    )
    output_names = get_output_names(estimator, g.convert_options, top_level_node_name)
    output_names = [g.unique_name(n) for n in output_names] if output_names else None

    is_container = isinstance(estimator, (Pipeline, ColumnTransformer, FeatureUnion))

    if function_options and function_options.export_as_function and not is_container:
        # Wrap the single top-level estimator as a local function.
        out_names = _wrap_step_as_function(
            g, function_options, estimator, list(input_names), output_names, fct, name="main"
        )
    elif is_container:
        out_names = fct(
            g,
            sts,
            output_names,
            estimator,
            *input_names,
            name="main",
            function_options=function_options,
        )
    else:
        out_names = fct(g, sts, output_names, estimator, *input_names, name="main")

    assert (
        output_names is None
        or (out_names == output_names[0] and len(output_names) == 1)
        or (out_names == tuple(output_names) and len(output_names) > 1)
    ), (
        f"estimator={cls}, {fct=}, output mismatch, {input_names=}, {output_names=}, "
        f"{out_names=}, {is_container=}, {function_options=}, {estimator=}{g.get_debug_msg()}"
    )
    if output_names is None:
        output_names = out_names
    else:
        output_names = tuple(output_names)
    for name in output_names:
        g.make_tensor_output(name, indexed=False, allow_untyped_output=True)
    # When local functions are requested we must NOT inline them; pass inline=False
    # so the function bodies are preserved in the returned ModelProto.
    if isinstance(g, GraphBuilder):
        onx = g.to_onnx(  # type: ignore
            large_model=large_model,
            external_threshold=external_threshold,
            inline=(not function_options) or not function_options.export_as_function,
            return_optimize_report=True,
        )
        if verbose and onx.report:
            print(f"[yobx.sklearn.to_onnx] done, output type is {type(onx)}")
            text = onx.report.to_string()
            if text:
                print(text)
        assert isinstance(onx, ExportArtifact), f"Unexpected type {type(onx)} for onx."
        if filename:
            if verbose:
                print(f"[yobx.sklearn.to_onnx] saving model to {filename!r}")
            onx.save(filename)
        return onx
    onx = g.to_onnx(
        large_model=large_model,
        external_threshold=external_threshold,
        inline=(not function_options) or not function_options.export_as_function,
    )
    assert isinstance(onx, ExportArtifact), f"Unexpected type {type(onx)} for onx."
    if filename:
        if verbose:
            print(f"[yobx.sklearn.to_onnx] saving model to {filename!r}")
        onx.save(filename)
    return onx
