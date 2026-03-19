"""Helper utilities for the :mod:`xla_call_module` StableHLO→ONNX converter.

This module contains low-level extraction utilities (bracket/brace balancing,
function-body extraction, dtype/reduce-op maps) that have no dependency on
TensorFlow, JAX, or any ONNX graph-builder state.

Functions whose names contain ``parse`` (e.g. :func:`~xla_call_module_parsing._parse_tensor_type`,
:func:`~xla_call_module_parsing._parse_dense_value`, :func:`~xla_call_module_parsing._parse_body`)
live in :mod:`xla_call_module_parsing` instead.
"""
import re
from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Dtype and reduce-op maps
# ---------------------------------------------------------------------------

# Mapping from MLIR element-type suffix to numpy dtype.  Sorted longest-first
# by key length to avoid partial-match issues (e.g. 'f16' before 'f1').
_MLIR_DTYPE_MAP: Dict[str, type] = {
    "bf16": np.float32,  # bfloat16 approximated as float32
    "f64": np.float64,
    "f32": np.float32,
    "f16": np.float16,
    "i64": np.int64,
    "i32": np.int32,
    "i16": np.int16,
    "i8": np.int8,
    "i1": np.bool_,
    "ui64": np.uint64,
    "ui32": np.uint32,
    "ui8": np.uint8,
}

# Mapping from StableHLO reduce op name to the ONNX op key used by _parse_body.
_REDUCE_OP_MAP: Dict[str, str] = {
    "maximum": "reduce_max",
    "add": "reduce_sum",
    "minimum": "reduce_min",
}


# ---------------------------------------------------------------------------
# Bracket / brace extraction utility
# ---------------------------------------------------------------------------


def _extract_balanced_parens(text: str, start: int, open_char: str = "(") -> str:
    """
    Extract the content between the opening bracket at *start* and its matching
    closing bracket, properly handling nested brackets.

    *start* must point at the *open_char* character.  Returns the text *inside*
    the outermost bracket pair (the delimiters themselves are not included).
    Returns an empty string if *start* does not point at *open_char*.

    The *open_char* / close pair defaults to ``"("`` / ``")"``; pass
    ``open_char="{"`` to extract curly-brace blocks.
    """
    close_char = ")" if open_char == "(" else "}"
    if start >= len(text) or text[start] != open_char:
        return ""
    depth = 0
    pos = start
    while pos < len(text):
        ch = text[pos]
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start + 1 : pos]
        pos += 1
    return text[start + 1 :]  # unterminated – return rest of string


# ---------------------------------------------------------------------------
# Private-function body extractor
# ---------------------------------------------------------------------------


def _extract_function_body(mlir_string: str, func_name: str) -> Optional[str]:
    """Return the body text of a named function in *mlir_string*.

    Handles both ``public`` and ``private`` function definitions.
    Returns ``None`` if the function is not found.
    """
    # Match 'func.func [public|private] @NAME('
    pattern = rf"func\.func\s+(?:public|private)\s+@{re.escape(func_name)}\s*\("
    m = re.search(pattern, mlir_string)
    if not m:
        return None
    paren_start = m.end() - 1  # position of '('
    sig_text = _extract_balanced_parens(mlir_string, paren_start)
    sig_end = paren_start + len(sig_text) + 2  # past closing ')'
    body_start = mlir_string.find("{", sig_end)
    if body_start == -1:
        return None
    # Skip '{' that are part of attribute/type annotations (e.g.
    # ``{jax.result_info = ...}`` in the return-type annotation).  The actual
    # function body ``{`` appears after either a ``)``, ``>``, or at the very
    # end of the return-type declaration.
    while body_start != -1:
        preceding = mlir_string[max(0, body_start - 80) : body_start]
        # The function-body opener follows the closing of the return type,
        # which ends with either ')' (annotated tensor) or '>' (plain tensor).
        if re.search(r"[)>]\s*$", preceding):
            break
        body_start = mlir_string.find("{", body_start + 1)
    if body_start == -1:
        return None
    return _extract_balanced_parens(mlir_string, body_start, open_char="{")

