from typing import Tuple, Dict, List, Union
from sklearn.pipeline import Pipeline
from ..register import register_sklearn_converter, get_sklearn_converter
from ...xbuilder import GraphBuilder


@register_sklearn_converter(Pipeline)
def sklearn_pipeline(
    g: GraphBuilder,
    sts: Dict,
    outputs: List[str],
    estimator: Pipeline,
    X: str,
    name: str = "pipeline",
) -> Union[str, Tuple[str, str]]:
    """
    Converts a :class:`class sklearn.preprocessing.StandardScaler` into ONNX.

    :param g: the graph builder to add nodes to
    :param sts: shapes defined by :epkg:`scikit-learn`
    :param estimator: a fitted ``LogisticRegression``
    :param outputs: desired names (scaled inputs)
    :param X: inputs
    :param name: prefix names for the added nodes
    :return: output
    """
    assert isinstance(estimator, Pipeline), f"Unexpected type {type(estimator)} for estimator."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"
    current_input = (X,)
    for step_name, step in estimator.steps:
        output_names = estimator.get_feature_names_out()
        fct = get_sklearn_converter(type(step))
        current_input = fct(
            g, sts, output_names, step, *current_input, name=f"{name}__{step_name}"
        )
    return current_input
