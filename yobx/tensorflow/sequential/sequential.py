from typing import Dict, List, Tuple, Union
from tensorflow.keras import Sequential  # type: ignore[import]
from ..register import register_tensorflow_converter, get_tensorflow_converter
from ..tensorflow_helper import get_output_names
from ...xbuilder import GraphBuilder


@register_tensorflow_converter(Sequential)
def tensorflow_sequential(
    g: GraphBuilder,
    sts: Dict,
    outputs: List[str],
    model: Sequential,
    X: str,
    name: str = "sequential",
) -> Union[str, Tuple[str, ...]]:
    """
    Converts a :class:`tensorflow.keras.Sequential` model into ONNX.

    The converter iterates over the model's layers and chains each layer's
    converter output into the next layer's input.  Intermediate tensor names
    are generated with
    :meth:`GraphBuilder.unique_name <yobx.xbuilder.GraphBuilder.unique_name>`
    to avoid collisions.

    :param g: the graph builder to add nodes to
    :param sts: shapes dictionary (unused, kept for API consistency)
    :param outputs: desired output tensor names for the final layer
    :param model: a built Keras :class:`Sequential` model
    :param X: name of the input tensor
    :param name: prefix used for names of nodes added for each layer
    :return: name of the output tensor, or a tuple of output tensor names
    """
    assert isinstance(model, Sequential), f"Unexpected type {type(model)} for model."
    assert g.has_type(X), f"Missing type for {X!r}{g.get_debug_msg()}"

    current_input = [X]
    layers = model.layers
    for i, layer in enumerate(layers):
        if i == len(layers) - 1:
            layer_outputs = outputs
        else:
            intermediate_names = list(get_output_names(layer))
            layer_outputs = [g.unique_name(n) for n in intermediate_names]
        fct = get_tensorflow_converter(type(layer))
        fct(g, sts, layer_outputs, layer, *current_input, name=f"{name}__{layer.name}")
        current_input = layer_outputs

    return current_input[0] if len(current_input) == 1 else tuple(current_input)
