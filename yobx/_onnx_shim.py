"""Exposes the mandatory native ONNX API provided by the onnx-light wheel."""

from onnx_light import onnx
from onnx_light.onnx import (
    checker,
    defs,
    external_data_helper,
    helper,
    inliner,
    numpy_helper,
    parser,
    printer,
    shape_inference,
)

__all__ = [
    "checker",
    "defs",
    "external_data_helper",
    "helper",
    "inliner",
    "numpy_helper",
    "onnx",
    "parser",
    "printer",
    "shape_inference",
]
