"""Compatibility shim: import onnx or onnx_light.onnx based on USE_OPTIM_ONNX.

When the environment variable ``USE_OPTIM_ONNX`` is set to ``1``, this module
re-exports everything from ``onnx_light.onnx`` (and its submodules) so that the
rest of the codebase never imports ``onnx`` directly.  Otherwise it falls back
to the standard ``onnx`` package.

Usage in other modules::

    from yobx._onnx_shim import onnx  # noqa: TID251

This single import line replaces ``import onnx``.
"""

from __future__ import annotations

import importlib
import os

_USE_OPTIM_ONNX = os.environ.get("USE_OPTIM_ONNX", "0") == "1"

if _USE_OPTIM_ONNX:
    onnx = importlib.import_module("onnx_light.onnx")
    # Ensure commonly used submodules are importable via attribute access.
    onnx.external_data_helper = importlib.import_module(  # type: ignore[attr-defined]
        "onnx_light.onnx.external_data_helper"
    )
    onnx.checker = importlib.import_module("onnx_light.onnx.checker")  # type: ignore[attr-defined]
    onnx.shape_inference = importlib.import_module(  # type: ignore[attr-defined]
        "onnx_light.onnx.shape_inference"
    )
    onnx.defs = importlib.import_module("onnx_light.onnx.defs")  # type: ignore[attr-defined]
    onnx.helper = importlib.import_module("onnx_light.onnx.helper")  # type: ignore[attr-defined]
    onnx.backend = importlib.import_module("onnx_light.onnx.backend")  # type: ignore[attr-defined]
    onnx.numpy_helper = importlib.import_module(  # type: ignore[attr-defined]
        "onnx_light.onnx.numpy_helper"
    )
else:
    onnx = importlib.import_module("onnx")  # type: ignore[no-redef]
    # Ensure submodules are loaded.
    importlib.import_module("onnx.external_data_helper")
    importlib.import_module("onnx.checker")
    importlib.import_module("onnx.shape_inference")
    importlib.import_module("onnx.defs")
