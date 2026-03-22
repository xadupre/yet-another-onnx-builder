from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from onnx import ValueInfoProto
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils.validation import check_is_fitted
from .. import DEFAULT_TARGET_OPSET
from ..typing import ConvertOptionsProtocol
from ..container import ExportArtifact
from ..helpers.onnx_helper import np_dtype_to_tensor_dtype
from ..xbuilder import GraphBuilder, OptimizationOptions
from ..xbuilder.function_options import FunctionOptions
from .register import get_sklearn_converter, sklearn_exportable_methods
from .sklearn_helper import get_output_names


class ConvertOptions(ConvertOptionsProtocol):
    """
    Tunes the way every piece of a model is exported.

    :param decision_leaf: (:class:`bool`) — when ``True``, an extra output
        tensor is appended containing the leaf node index (int64) for each
        input sample.  Shapes follow the same convention as ``decision_path``.
    :param decision_path: (:class:`bool`) — when ``True``, an extra output
        tensor is appended containing the binary decision path string(s) for
        each input sample.  For single trees the shape is ``(N, 1)``; for
        ensembles ``(N, n_estimators)``.
    """

    OPTIONS = ["decision_leaf", "decision_path"]

    def __init__(
        self,
        decision_leaf: Union[bool, Set[str]] = False,
        decision_path: Union[bool, Set[str]] = False,
    ):
        self.decision_leaf = decision_leaf
        self.decision_path = decision_path

    def __repr__(self):
        rows = []
        for name in self.OPTIONS:
            rows.append(f"    {name}={getattr(self, name)},")
        text = "\n".join(rows)
        return f"{self.__class__.__name__}(\n{text}\n)"

    def has(self, option_name: str, piece: BaseEstimator) -> bool:  # type: ignore[bad-override]
        """Tells of options `option_name` applies on estimator `piece`."""
        assert hasattr(
            self, option_name
        ), f"Missing option {option_name!r}. Allowed {self.OPTIONS}."
        value = getattr(self, option_name)
        if not value:
            return False
        if value is True:
            return True
        raise NotImplementedError(
            f"Not implemented with {option_name!r} is not a boolean but {value!r}."
        )


def _extract_value_info_proto(vip: ValueInfoProto) -> Tuple[str, int, Optional[Tuple]]:
    """Extract ``(name, elem_type, shape)`` from a :class:`onnx.ValueInfoProto`.

    :param vip: an ONNX value-info descriptor
    :return: tuple ``(name, elem_type, shape)`` where *shape* is ``None`` when
        no shape information is present, and otherwise a tuple whose elements
        are ``int`` for static dimensions, a non-empty ``str`` for symbolic
        dimensions, and an auto-generated name such as ``"_unk_0_"`` for
        dimensions with no value and no symbolic name.
    """
    name = vip.name
    tt = vip.type.tensor_type
    elem_type = tt.elem_type
    if tt.HasField("shape"):
        shape_dims = [
            (dim.dim_param if dim.dim_param else (dim.dim_value or f"dim_{name}_{idim}"))
            for idim, dim in enumerate(tt.shape.dim)
        ]
        shape: Optional[Tuple] = tuple(shape_dims)
    else:
        shape = None
    return name, elem_type, shape


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
    :param args: dummy inputs; each element may be a numpy array **or** an
        :class:`onnx.ValueInfoProto` that explicitly describes the input
        tensor's name, element type and shape.  When a
        :class:`~onnx.ValueInfoProto` is provided no actual data is required,
        and the ``dynamic_shapes`` parameter is ignored for that input (the
        shape is taken directly from the proto).
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

    if input_names:
        if len(input_names) != len(args):
            raise ValueError(f"Length mismatch: {len(args)=} but input_names={input_names!r}")
    else:
        # Derive default input names; for ValueInfoProto use the embedded name.
        default_names = []
        for j, arg in enumerate(args):
            if isinstance(arg, ValueInfoProto):
                default_names.append(arg.name or (f"X{j}" if len(args) > 1 else "X"))
            else:
                default_names.append("X" if len(args) == 1 else f"X{j}")
        input_names = default_names
    for i, (name, arg) in enumerate(zip(input_names, args)):
        if isinstance(arg, ValueInfoProto):
            # Use name/type/shape directly from the ValueInfoProto.
            _, elem_type, shape = _extract_value_info_proto(arg)
            g.make_tensor_input(name, elem_type, shape, device=-1)
        else:
            if dynamic_shapes:
                ds = dynamic_shapes[i]
            else:
                ds = {0: "batch"}
            shape = list(arg.shape)  # type: ignore
            for axis, dim in ds.items():
                shape[axis] = dim  # type: ignore
            g.make_tensor_input(  # type: ignore
                name, np_dtype_to_tensor_dtype(arg.dtype), tuple(shape), device=-1  # type: ignore
            )  # type: ignore

    # Build the sts dict (shared state for converters). function_options, if set,
    # is passed explicitly to container converters below.
    sts: Dict[str, Any] = {}
    output_names = list(get_output_names(estimator))
    # Append extra output names requested by convert_options so that converters
    # can detect them via len(outputs) > extra_idx and emit the extra nodes.
    if convert_options:
        for _extra_opt in ConvertOptions.OPTIONS:
            if convert_options.has(_extra_opt, estimator):
                output_names.append(_extra_opt)

    is_container = isinstance(estimator, (Pipeline, ColumnTransformer, FeatureUnion))

    if function_options and function_options.export_as_function and not is_container:
        # Wrap the single top-level estimator as a local function.
        _wrap_step_as_function(
            g, function_options, estimator, list(input_names), output_names, fct, name="main"
        )
    elif is_container:
        fct(
            g,
            sts,
            output_names,
            estimator,
            *input_names,
            name="main",
            function_options=function_options,
        )
    else:
        fct(g, sts, output_names, estimator, *input_names, name="main")

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
            onx.save(filename)
        return onx
    onx = g.to_onnx(
        large_model=large_model,
        external_threshold=external_threshold,
        inline=(not function_options) or not function_options.export_as_function,
    )
    assert isinstance(onx, ExportArtifact), f"Unexpected type {type(onx)} for onx."
    if filename:
        onx.save(filename)
    return onx
