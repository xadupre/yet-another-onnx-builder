"""Compatibility shim: import onnx or onnx_light.onnx based on USE_ONNX_LIGHT.

When the environment variable ``USE_ONNX_LIGHT`` is set to ``1``, this module
re-exports everything from ``onnx_light.onnx`` (and its submodules) so that the
rest of the codebase never imports ``onnx`` directly.  Otherwise it falls back
to the standard ``onnx`` package.

Usage in other modules::

    from yobx._onnx_shim import onnx  # noqa: TID251

This single import line replaces ``import onnx``.  Commonly used submodules are
registered as attributes so that ``onnx.helper``, ``onnx.numpy_helper`` and the
others are always available regardless of the active backend.
"""

from __future__ import annotations

import importlib
import os

_USE_ONNX_LIGHT = os.environ.get("USE_ONNX_LIGHT", "0") == "1"

# Submodules registered as attributes on the exported ``onnx`` module so that
# ``onnx.<submodule>`` works for both backends.  Some submodules do not exist in
# every backend (for example ``onnx.mapping`` was removed from recent ``onnx``
# releases), so missing ones are skipped.
_SUBMODULES = (
    "external_data_helper",
    "checker",
    "shape_inference",
    "defs",
    "helper",
    "numpy_helper",
    "parser",
    "backend",
    "version_converter",
    "mapping",
    "compose",
    "inliner",
    "reference",
)

_BASE = "onnx_light.onnx" if _USE_ONNX_LIGHT else "onnx"

onnx = importlib.import_module(_BASE)

for _submodule in _SUBMODULES:
    try:
        _module = importlib.import_module(f"{_BASE}.{_submodule}")
    except ModuleNotFoundError:
        continue
    setattr(onnx, _submodule, _module)  # type: ignore[attr-defined]

# Re-export every public attribute of the active backend (proto classes such as
# ``TensorProto``, helpers such as ``load``/``save`` and the submodules above) so
# that ``from yobx._onnx_shim import TensorProto`` or ``... import helper as oh``
# work as drop-in replacements for the equivalent ``onnx`` imports.
for _name, _value in vars(onnx).items():
    if not _name.startswith("_"):
        globals().setdefault(_name, _value)
