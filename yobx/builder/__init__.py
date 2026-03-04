"""
Builder sub-package.
"""

try:
    from .onnxscript import OnnxScriptGraphBuilder

    __all__ = ["OnnxScriptGraphBuilder"]
except ImportError:
    # onnxscript or onnx_ir not installed; bridge not available.
    __all__ = []  # type: ignore[assignment]
