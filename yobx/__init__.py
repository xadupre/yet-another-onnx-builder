"""
yet-another-onnx-builder converts models from any kind to ONNX format.
"""

__version__ = "0.2.0"

from .convert import DEFAULT_TARGET_OPSET, guess_export, to_onnx  # noqa: E402
