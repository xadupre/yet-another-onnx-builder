"""
yet-another-onnx-builder converts models to ONNX format.
"""

__version__ = "0.1.0"

from .typing import GraphBuilderProtocol, GraphBuilderExtendedProtocol, OpsetProtocol, TensorProtocol

__all__ = ["GraphBuilderProtocol", "GraphBuilderExtendedProtocol", "OpsetProtocol", "TensorProtocol"]
