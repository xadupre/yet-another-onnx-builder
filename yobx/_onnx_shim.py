"""Selects the onnx backend based on the ``USE_OPTIM_ONNX`` environment variable.

The canonical equivalence is ``onnx`` ↔ ``onnx_light.onnx``.  When
``USE_OPTIM_ONNX=1`` is set this module registers ``onnx_light.onnx`` as
``sys.modules["onnx"]`` and maps each known sub-module
(``onnx_light.onnx.helper``, ``onnx_light.onnx.defs``, …) into the
corresponding ``sys.modules["onnx.*"]`` slot.  Every subsequent
``import onnx`` or ``from onnx.X import Y`` statement throughout the codebase
then transparently uses the lighter implementation without any other file
needing to be changed.

Importing this module is a **no-op** when ``USE_OPTIM_ONNX`` is not set to
``"1"``.
"""

import importlib
import os
import sys
import types

# Sub-modules of ``onnx`` that are used across the codebase.
_ONNX_SUBMODULES = (
    "backend",
    "checker",
    "defs",
    "external_data_helper",
    "helper",
    "inliner",
    "model_container",
    "numpy_helper",
    "reference",
    "shape_inference",
)


def _apply_onnx_light() -> None:
    """Registers ``onnx_light.onnx`` as ``sys.modules["onnx"]`` when
    ``USE_OPTIM_ONNX=1`` is present in the process environment.

    The function also maps each known ``onnx`` sub-module to its
    ``onnx_light.onnx.*`` counterpart so that dotted imports such as
    ``from onnx.helper import make_node`` continue to resolve correctly.

    Raises :exc:`ImportError` if ``USE_OPTIM_ONNX=1`` but the ``onnx_light``
    package is not installed.
    """
    if os.environ.get("USE_OPTIM_ONNX") != "1":
        return

    # ``onnx_light.onnx`` is the light-weight drop-in for the ``onnx`` package.
    try:
        onnx_mod: types.ModuleType = importlib.import_module("onnx_light.onnx")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "USE_OPTIM_ONNX=1 requires the 'onnx_light' package to be installed. "
            "Install it or unset USE_OPTIM_ONNX to use the standard 'onnx' package."
        ) from exc

    # Register the top-level replacement.
    sys.modules["onnx"] = onnx_mod

    # Register every known sub-module so that dotted imports work.
    for name in _ONNX_SUBMODULES:
        try:
            sub: types.ModuleType = importlib.import_module(f"onnx_light.onnx.{name}")
        except ModuleNotFoundError:
            # onnx_light may not implement every sub-module — skip gracefully.
            continue
        sys.modules[f"onnx.{name}"] = sub
        # Also expose as an attribute on the top-level module so that
        # ``import onnx; onnx.helper.make_node(…)`` works correctly.
        setattr(onnx_mod, name, sub)


_apply_onnx_light()
