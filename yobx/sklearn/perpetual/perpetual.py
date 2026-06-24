"""
ONNX converters for :class:`perpetual.PerpetualClassifier`
and :class:`perpetual.PerpetualRegressor`.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Union
import onnx
from perpetual import PerpetualClassifier, PerpetualRegressor
from ...typing import GraphBuilderExtendedProtocol
from ...xbuilder import GraphBuilder
from ..register import register_sklearn_converter


def _add_perpetual_subgraph(
    g: GraphBuilderExtendedProtocol, outputs: List[str], estimator, X: str, name: str
) -> Union[str, List[str]]:
    with TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / f"{name}.onnx"
        estimator.save_as_onnx(str(onnx_path), name=name)
        model = onnx.load(str(onnx_path))
    sub_builder = GraphBuilder(model)
    for domain in tuple(sub_builder.opsets):
        if g.has_opset(domain):
            sub_builder.opsets[domain] = g.get_opset(domain)
    input_names = [i.name for i in model.graph.input]
    assert len(input_names) == 1, f"Unexpected number of inputs in perpetual model: {input_names}"
    return g.make_nodes(sub_builder, [X], outputs, prefix=f"{name}_")


@register_sklearn_converter(PerpetualClassifier)
def sklearn_perpetual_classifier(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator,
    X: str,
    name: str = "perpetual_classifier",
) -> Union[str, List[str]]:
    """Converts a :class:`perpetual.PerpetualClassifier` to ONNX."""
    return _add_perpetual_subgraph(g, outputs, estimator, X, name)


@register_sklearn_converter(PerpetualRegressor)
def sklearn_perpetual_regressor(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: List[str],
    estimator,
    X: str,
    name: str = "perpetual_regressor",
) -> Union[str, List[str]]:
    """Converts a :class:`perpetual.PerpetualRegressor` to ONNX."""
    return _add_perpetual_subgraph(g, outputs, estimator, X, name)
