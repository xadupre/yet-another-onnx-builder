"""
:mod:`yobx.litert` — LiteRT/TFLite → ONNX converter.

Converts a ``.tflite`` model into an :class:`onnx.ModelProto` using
a pure-Python :epkg:`TFLite` FlatBuffer parser and a registry of
op-level converters.

Usage::

    import numpy as np
    from yobx.litert import to_onnx

    X = np.random.rand(1, 4).astype(np.float32)
    onx = to_onnx("model.tflite", (X,))
"""

from .convert import to_onnx


def register_litert_converters() -> None:
    """Register all built-in LiteRT op converters.

    This function is idempotent — calling it more than once has no effect.
    It is called automatically by :func:`to_onnx` so you rarely need to
    invoke it directly.
    """
    from .register import LITERT_OP_CONVERTERS

    if LITERT_OP_CONVERTERS:
        return
    from .ops import register as _register_ops

    _register_ops()


__all__ = [
    "register_litert_converters",
    "to_onnx",
]
