"""
Converts a :class:`sklearn.preprocessing.StandardScaler` into an ONNX graph.
"""
import numpy as np
from onnx import TensorProto
from ..xbuilder import GraphBuilder


def _add_standard_scaler_nodes(
    g: GraphBuilder,
    scaler,
    input_name: str,
    output_name: str,
    prefix: str = "",
) -> str:
    """
    Adds StandardScaler nodes to an existing :class:`GraphBuilder`.

    :param g: the graph builder to add nodes to
    :param scaler: a fitted ``StandardScaler``
    :param input_name: name of the input result
    :param output_name: desired name for the output result
    :param prefix: prefix for initializer names (avoids collisions in pipelines)
    :return: name of the output result
    """
    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    n_features = mean.shape[0]

    mean_name = g.make_initializer(f"{prefix}mean", mean)
    scale_name = g.make_initializer(f"{prefix}scale", scale)
    centered = g.op.Sub(input_name, mean_name, name=f"{prefix}sub_mean")
    normalized = g.op.Div(
        centered, scale_name, name=f"{prefix}div_scale", outputs=[output_name]
    )
    return normalized


def convert_standard_scaler(
    scaler,
    input_name: str = "X",
    output_name: str = "variable",
    opset: int = 18,
) -> GraphBuilder:
    """
    Converts a fitted :class:`sklearn.preprocessing.StandardScaler` into
    a :class:`GraphBuilder`.

    :param scaler: a fitted ``StandardScaler``
    :param input_name: name of the input
    :param output_name: name of the output
    :param opset: ONNX opset version
    :return: :class:`GraphBuilder` ready to be exported with ``to_onnx()``
    """
    n_features = scaler.mean_.shape[0]
    g = GraphBuilder(opset, ir_version=9)
    g.make_tensor_input(input_name, TensorProto.FLOAT, (None, n_features))
    out = _add_standard_scaler_nodes(g, scaler, input_name, output_name)
    g.make_tensor_output(out, TensorProto.FLOAT, (None, n_features), indexed=False)
    return g
