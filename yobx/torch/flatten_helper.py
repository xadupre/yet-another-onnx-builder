"""Backward-compatibility shim. Use :mod:`yobx.torch.flatten` instead."""
from __future__ import annotations

from .flatten import (  # noqa: F401
    PATCH_OF_PATCHES,
    flattening_functions,
    make_flattening_function_for_dataclass,
    register_cache_flattening,
    register_class_flattening,
    register_flattening_functions,
    replacement_before_exporting,
    unregister_cache_flattening,
    unregister_class_flattening,
)
