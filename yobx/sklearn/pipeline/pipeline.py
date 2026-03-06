from typing import Tuple, Dict, List, Union
from sklearn.pipeline import Pipeline
from ...typing import GraphBuilderProtocol
from ..register import register_sklearn_converter, get_sklearn_converter
from ..sklearn_helper import get_output_names


@register_sklearn_converter(Pipeline)
def sklearn_pipeline(
    g: GraphBuilderProtocol,
    sts: Dict,
    outputs: List[str],
    estimator: Pipeline,
    X: str,
    name: str = "pipeline",
) -> Union[str, Tuple[str, ...]]:
    """
    Converts a :class:`sklearn.pipeline.Pipeline` into ONNX using registered
    converters for each step in the pipeline.

    The converter iterates over the pipeline steps and chains each step's
    converter output into the next step's input.  Intermediate tensor names
    are generated with
    :meth:`GraphBuilder.unique_name <yobx.xbuilder.GraphBuilder.unique_name>`
    to avoid collisions.

    :param g: the graph builder to add nodes to
    :param sts: shapes and types defined by :epkg:`scikit-learn`
    :param outputs: desired output tensor names for the pipeline result
    :param estimator: a fitted :class:`sklearn.pipeline.Pipeline`
    :param X: name of the input tensor to the pipeline
    :param name: prefix used for names of nodes added for each pipeline step
    :return: name of the output tensor, or a tuple of output tensor names
    """
    assert isinstance(estimator, Pipeline), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"
    current_input = [X]
    for i, (step_name, step) in enumerate(estimator.steps):
        if i == len(estimator.steps) - 1:
            output_names = outputs
        else:
            output_names = list(get_output_names(step))
            output_names = [g.unique_name(n) for n in output_names]
        fct = get_sklearn_converter(type(step))
        fct(g, sts, output_names, step, *current_input, name=f"{name}__{step_name}")
        current_input = output_names
    return current_input[0] if len(current_input) == 1 else tuple(current_input)
