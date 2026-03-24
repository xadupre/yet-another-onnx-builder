from typing import Callable, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator
from ..xbuilder import GraphBuilder
from ..xbuilder.function_options import FunctionOptions
from .sklearn_helper import get_output_names


def default_ai_onnx_ml(main_opset: int) -> int:
    """Returns a default value for opset `ai.onnx.ml`."""
    if main_opset >= 21:
        return 5
    if main_opset >= 18:
        return 3
    if main_opset >= 6:
        return 2
    return 1


def wrap_step_as_function(
    g: GraphBuilder,
    fopts: FunctionOptions,
    estimator: BaseEstimator,
    input_names: List[str],
    output_names: Optional[List[str]],
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
    raw_output_names = get_output_names(estimator, g.convert_options, name)
    if not function_input_names or raw_output_names is None:
        # No inputs, or output count unknown (NoKnownOutputMixin): let the
        # converter (and its normalization wrapper) decide the output names.
        function_output_names = None
    else:
        function_output_names = [sub_g.unique_name(n) for n in raw_output_names]  # type: ignore

    # Run the converter inside the sub-builder (no function_options propagation).
    out_names = converter(
        sub_g, {}, function_output_names, estimator, *function_input_names, name=name
    )
    # Reconcile: if outputs were not pre-specified, use the names returned by the converter.
    if function_output_names is None:
        function_output_names = (out_names,) if isinstance(out_names, str) else tuple(out_names)  # type: ignore
    else:
        function_output_names = tuple(function_output_names)  # type: ignore

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
    if output_names:
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
    if renames:
        for tmp, final in renames:
            g.make_node("Identity", [tmp], [final], name=f"{name}_rename")
    else:
        output_names = actual_output_names

    # Propagate type/shape metadata from the sub-builder to the main graph.
    assert output_names is not None, "type checking"
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
    return function_output_names if len(function_output_names) > 1 else function_output_names[0]  # type: ignore


def wrap_step(
    g: GraphBuilder,
    sts,
    fopts: Optional[FunctionOptions],
    is_container: bool,
    estimator: BaseEstimator,
    input_names: List[str],
    output_names: Optional[List[str]],
    converter: Callable,
    name: str,
) -> Union[str, Tuple[str, ...]]:
    if is_container:
        return converter(
            g, sts, output_names, estimator, *input_names, name=name, function_options=fopts
        )
    # In that case, we need to concatenate input_names if there are multiple.
    if len(input_names) > 1:
        conc = g.op.Concat(*input_names, axis=1, name=f"{name}_concat")
        input_names = [conc]
    return converter(g, sts, output_names, estimator, *input_names, name=name)
