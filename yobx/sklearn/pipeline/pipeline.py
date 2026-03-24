from typing import Tuple, Dict, List, Optional, Union
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from ...typing import GraphBuilderExtendedProtocol
from ...xbuilder import FunctionOptions
from ..register import register_sklearn_converter, get_sklearn_converter
from ..sklearn_helper import get_output_names
from ..convert import _wrap_step_as_function


@register_sklearn_converter(Pipeline)
def sklearn_pipeline(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: Pipeline,
    X: str,
    name: str = "pipeline",
    function_options: Optional[FunctionOptions] = None,
) -> Union[str, Tuple[str, ...]]:
    """
    Converts a :class:`sklearn.pipeline.Pipeline` into ONNX using registered
    converters for each step in the pipeline.

    The converter iterates over the pipeline steps and chains each step's
    converter output into the next step's input.  Intermediate tensor names
    are generated with
    :meth:`GraphBuilder.unique_name <yobx.xbuilder.GraphBuilder.unique_name>`
    to avoid collisions.

    When ``sts`` contains a :class:`~yobx.xbuilder.FunctionOptions`
    under the ``_FUNCTION_OPTIONS_KEY`` key each step is exported as a
    separate ONNX local function instead of being inlined into the graph.

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`; also carries
        an optional :class:`~yobx.xbuilder.FunctionOptions` value under
        the key :data:`~yobx.sklearn.convert._FUNCTION_OPTIONS_KEY`
    :param outputs: desired output tensor names for the pipeline result
    :param estimator: a fitted :class:`sklearn.pipeline.Pipeline`
    :param X: name of the input tensor to the pipeline
    :param name: prefix used for names of nodes added for each pipeline step
    :param function_options: to export every step as a local function
    :return: name of the output tensor, or a tuple of output tensor names
    """
    assert isinstance(estimator, Pipeline), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    current_input = [X]
    for i, (step_name, step) in enumerate(estimator.steps):
        if i == len(estimator.steps) - 1:
            output_names = outputs
        else:
            output_names = list(get_output_names(step, g.convert_options))
            output_names = [g.unique_name(n) for n in output_names] if output_names else None
        fct = get_sklearn_converter(type(step))
        step_node_name = f"{name}__{step_name}"
        is_container = isinstance(step, (Pipeline, ColumnTransformer, FeatureUnion))
        if function_options and function_options.export_as_function and not is_container:
            with g.prefix_name_context(step_node_name):
                out_names = _wrap_step_as_function(
                    g,  # type: ignore
                    function_options,
                    step,
                    current_input,
                    output_names,
                    fct,
                    step_node_name,
                )
        elif is_container:
            with g.prefix_name_context(step_node_name):
                out_names = fct(
                    g,
                    sts,
                    output_names,
                    step,
                    *current_input,
                    name=step_node_name,
                    function_options=function_options,
                )
        else:
            with g.prefix_name_context(step_node_name):
                out_names = fct(g, sts, output_names, step, *current_input, name=step_node_name)
        if output_names is None:
            out_names = out_names
        current_input = output_names
    return current_input[0] if len(current_input) == 1 else tuple(current_input)
