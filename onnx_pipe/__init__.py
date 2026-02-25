"""onnx_pipe – pipe operator for building ONNX models.

Usage::

    from onnx_pipe import OnnxPipe, op
    import numpy as np

    # Chain single-operator pipes with |
    pipe = op("Abs") | op("Relu")
    model = pipe.to_onnx()

    # Wrap an existing ModelProto
    import onnx
    existing = onnx.load("my_model.onnx")
    pipe2 = OnnxPipe(existing) | op("Sigmoid")
"""

from ._op import op
from ._pipe import OnnxPipe

__all__ = ["OnnxPipe", "op"]
