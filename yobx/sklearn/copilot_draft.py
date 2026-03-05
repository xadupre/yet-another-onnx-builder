"""
Compatibility shim - the implementation has moved to
:mod:`yobx.helpers.copilot`.  This module re-exports everything so that
existing imports continue to work without modification.
"""

# Re-export all public symbols from the canonical location.
from yobx.helpers.copilot import (  # noqa: F401
    _build_converter_prompt,
    _call_copilot_api,
    _extract_python_code,
    _get_copilot_token,
    _infer_submodule,
    draft_converter_with_copilot,
)
