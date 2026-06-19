"""
yet-another-onnx-builder converts models from any kind to ONNX format.
"""

__version__ = "0.1.0"

from . import _onnx_shim  # noqa: F401
from .convert import DEFAULT_TARGET_OPSET, to_onnx  # noqa: E402
