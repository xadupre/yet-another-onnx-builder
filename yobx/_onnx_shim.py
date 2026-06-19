"""Selects the onnx backend based on the ``USE_OPTIM_ONNX`` environment variable.

When ``USE_OPTIM_ONNX=1`` is set, this module replaces the ``onnx`` package in
:data:`sys.modules` with ``onnx_light`` so that every subsequent
``import onnx`` statement throughout the codebase transparently uses the
lighter implementation.  The substitution relies on Python's module-import
machinery: once ``sys.modules["onnx"]`` is set to *onnx_light*, Python uses
*onnx_light*'s ``__path__`` when resolving dotted imports such as
``from onnx.helper import make_node``, so no other files need to be changed.

Importing this module is a **no-op** when ``USE_OPTIM_ONNX`` is not set to
``"1"`` or when ``onnx_light`` is not installed.
"""

import os
import sys


def _apply_onnx_light() -> None:
    """Replaces ``onnx`` with ``onnx_light`` in :data:`sys.modules` when
    ``USE_OPTIM_ONNX=1`` is present in the process environment.

    Raises :exc:`ImportError` if ``USE_OPTIM_ONNX=1`` but the ``onnx_light``
    package is not installed.
    """
    if os.environ.get("USE_OPTIM_ONNX") != "1":
        return

    try:
        import onnx_light  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "USE_OPTIM_ONNX=1 requires the 'onnx_light' package to be installed. "
            "Install it or unset USE_OPTIM_ONNX to use the standard 'onnx' package."
        ) from exc

    # Register onnx_light as the canonical onnx module.  Python's import
    # machinery will then use onnx_light.__path__ when resolving any
    # subsequent sub-module imports (e.g. ``onnx.helper``, ``onnx.defs``),
    # so no further changes are needed in the rest of the codebase.
    sys.modules["onnx"] = onnx_light


_apply_onnx_light()
