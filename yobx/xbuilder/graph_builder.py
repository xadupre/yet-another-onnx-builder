"""Exposes the native graph builder at the historical public import path.

Graph construction, shape inference and pattern optimization use onnx-light.
The former Python graph engine is deliberately not available as a fallback.
Unsupported legacy constructor options raise rather than selecting another engine.
"""

from ..builder.onnxlight import OnnxLightGraphBuilder as GraphBuilder
from ..builder.onnxlight import OnnxLightOptimizationOptions as OptimizationOptions
from ..typing import GraphBuilderTorchProtocol
from .function_options import FunctionOptions
from .infer_shapes_options import InferShapesOptions

TEMPLATE_TYPE = 999

__all__ = [
    "TEMPLATE_TYPE",
    "FunctionOptions",
    "GraphBuilder",
    "GraphBuilderTorchProtocol",
    "InferShapesOptions",
    "OptimizationOptions",
]
