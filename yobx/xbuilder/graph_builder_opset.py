"""Preserves the historical Opset import while using the native builder adapter."""

from ..builder.onnxlight.bridge_graph_builder import OnnxLightGraphBuilderOpset as Opset

__all__ = ["Opset"]
