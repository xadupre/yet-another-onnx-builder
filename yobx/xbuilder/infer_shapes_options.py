from __future__ import annotations
from enum import IntEnum


class InferShapesOptions(IntEnum):
    """
    Defines options when running shape inference on an existing model.
    Options ``NEW`` means shapes information is removed by running it again.
    """

    NONE = 0
    NEW = 1
    ONNX = 2
    DATA_PROP = 4
    BUILDER = 8
