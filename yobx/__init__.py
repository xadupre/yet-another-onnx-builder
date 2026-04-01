"""
yet-another-onnx-builder converts models from any kind to ONNX format.
"""

__version__ = "0.1.0"

DEFAULT_TARGET_OPSET = 21

from .convert import to_onnx  # noqa: E402
