from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
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
from .convert_helper import wrap_step_as_function, wrap_step, default_ai_onnx_ml


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
    return_optimize_report: bool = False,
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
    :param return_optimize_report: if True, the returned
        :class:`~yobx.container.ExportArtifact` has its
        :attr:`~yobx.container.ExportArtifact.report` attribute populated with
        per-pattern optimization statistics
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
        dict_target_opset = {"": target_opset, "ai.onnx.ml": default_ai_onnx_ml(target_opset)}
    else:
        if not isinstance(target_opset, dict):
            raise TypeError(f"target_opset must be a dictionary or an integer not {target_opset}")
        dict_target_opset = target_opset.copy()
        if "" not in dict_target_opset:
            dict_target_opset[""] = 21
        if "ai.onnx.ml" not in dict_target_opset:
            dict_target_opset["ai.onnx.ml"] = default_ai_onnx_ml(dict_target_opset[""])

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
        out_names = wrap_step_as_function(
            g, function_options, estimator, list(input_names), output_names, fct, name="main"
        )
    else:
        out_names = wrap_step(
            g,
            sts,
            function_options,
            is_container,
            estimator,
            input_names,
            output_names,
            fct,
            "main",
        )

    assert isinstance(out_names, str) or (
        isinstance(out_names, tuple) and all(isinstance(o, str) for o in out_names)
    ), (
        f"estimator={cls}, {fct=}, type mismatch, {input_names=}, {out_names=}, "
        f"{is_container=}, {function_options=}, {estimator=}{g.get_debug_msg()}"
    )
    assert (
        output_names is None
        or (out_names == output_names[0] and len(output_names) == 1)
        or (out_names == tuple(output_names) and len(output_names) > 1)
    ), (
        f"estimator={cls}, {fct=}, output mismatch, {input_names=}, {output_names=}, "
        f"{out_names=}, {is_container=}, {function_options=}, {estimator=}{g.get_debug_msg()}"
    )
    if output_names is None:
        # Converter was called with outputs=None; collect returned name(s) into a tuple.
        output_names = (out_names,) if isinstance(out_names, str) else tuple(out_names)
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
            return_optimize_report=return_optimize_report,
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
