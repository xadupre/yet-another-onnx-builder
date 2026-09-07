"""Exposes explicit native optimization options without Python-engine fallbacks."""

from ..builder.onnxlight import OnnxLightOptimizationOptions as OptimizationOptions

__all__ = ["OptimizationOptions"]
