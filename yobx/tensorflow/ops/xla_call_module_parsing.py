"""StableHLO MLIR → layer-dict parser for the XlaCallModule converter.

:func:`parse_mlir` is the public entry point.  All parsing utilities it
depends on live in :mod:`xla_call_module_helper`.
"""
import re
from typing import List

from .xla_call_module_helper import _extract_balanced_parens, _parse_body


def _get_layer_pos(layer: dict, scan_text: str) -> int:
    """Return the character offset of this layer's result id in *scan_text*.

    Used as the sort key to restore execution order after all patterns have
    been collected.  Returns ``len(scan_text)`` (i.e., sorts last) when the
    id cannot be located.
    """
    rid = layer.get("id", "")
    if rid and rid != "?":
        m = re.search(re.escape(rid) + r"\b", scan_text)
        if m:
            return m.start()
    return len(scan_text)


def parse_mlir(mlir_string: str) -> List[dict]:
    """Parse a StableHLO MLIR module text into a list of layer dicts.

    Each dict has at minimum the keys ``id``, ``op``, ``operands``, ``shape``,
    and ``loc``.  Recognised ``op`` values include:

    * ``"Input"`` – a function argument (tensor input).
    * ``"return"`` – function return.
    * ``"constant"`` – ``stablehlo.constant``; has extra key ``dense_content``.
    * ``"dot_general"`` – matrix multiply; has ``lhs_contracting`` /
      ``rhs_contracting``.
    * ``"broadcast_in_dim"`` / ``"dynamic_broadcast_in_dim"`` – pass-through
      broadcast (ONNX handles implicit broadcasting).
    * ``"call"`` – call to a private helper function; has extra key ``func``.
    * ``"reduce_max"`` / ``"reduce_sum"`` – reduction with keepdims; has
      ``axes``.
    * ``"skip"`` – shape-only op (reshape of integer tensors, concatenate of
      integer tensors, get_dimension_size); should be ignored by the converter.
    * Any other name – direct StableHLO→ONNX op name (e.g. ``"sine"``,
      ``"add"``, ``"compare_GT"``).
    """
    results = []

    # 1. Capture Function Arguments (The Initial Inputs)
    #
    # jax2tf with dynamic shapes emits a @main function followed by a private
    # @_wrapped_jax_export_main helper.  The helper carries both the actual
    # tensor inputs *and* extra scalar dimension-variable arguments annotated
    # with ``jax.global_constant``.  We only want the tensor inputs that
    # correspond to entries in op.inputs.
    #
    # Strategy: extract inputs only from the *public* @main function signature.
    # If there is no explicit @main function, fall back to scanning the whole
    # text as before.
    arg_header_pattern = r"(%arg\d+)\s*:\s*(tensor<[^>]+>)"

    public_func_match = re.search(r"func\.func\s+public\s+@\w+\s*\(", mlir_string)
    if public_func_match:
        paren_start = public_func_match.end() - 1  # position of '('
        header_text = _extract_balanced_parens(mlir_string, paren_start)
    else:
        header_text = mlir_string

    seen_arg_ids: set = set()
    for arg_id, shape in re.findall(arg_header_pattern, header_text):
        if arg_id in seen_arg_ids:
            continue
        seen_arg_ids.add(arg_id)
        results.append(
            {"id": arg_id, "op": "Input", "operands": tuple(), "shape": shape, "loc": "header"}
        )

    # 2. Determine which text region to scan for computation ops.
    #
    # When jax2tf emits a @_wrapped_jax_export_main helper, the actual
    # computation lives there.  The @main body only has boilerplate (shape
    # assertions, dimension-size reads, the call instruction, and a return).
    # We must:
    #   (a) parse ops from the wrapped function body, and
    #   (b) remap the wrapped function's %argN references back to the @main
    #       %argM names so that the result lookup works correctly.
    #
    # If no helper function exists, scan the whole MLIR text as before.

    # Build an alias map from the @_wrapped helper's arg names back to the
    # @main arg names so that op references in the helper resolve correctly.
    # The @main body contains exactly one ``call @_wrapped…(args…)`` instruction
    # whose positional arguments tell us the mapping:
    #   call @_wrapped_jax_export_main(%dim0, %mainArg0, %mainArg1, …)
    # We map wrapped_arg_id → main_arg_id for each position that is a @main arg.
    arg_alias: dict = {}

    # Try to extract just the body of @_wrapped_jax_export_main if it exists.
    wrapped_func_match = re.search(
        r"func\.func\s+private\s+@_wrapped_jax_export_main\s*\(", mlir_string
    )
    if wrapped_func_match:
        # Extract full wrapped function signature (handles nested parens from
        # ``loc(...)`` attribute annotations).
        paren_start = wrapped_func_match.end() - 1  # position of '('
        wrapped_sig_text = _extract_balanced_parens(mlir_string, paren_start)
        # Find body start (``{``) after the signature closing ')'
        sig_end = paren_start + len(wrapped_sig_text) + 2  # +2 for the '(' and ')'
        body_start = mlir_string.find("{", sig_end)
        if body_start != -1:
            # Skip ``{`` that are part of attribute annotations like
            # ``{jax.result_info = ...}`` — these appear before the actual
            # function body.  The function body ``{`` is preceded by a ')'
            # (closing the return-type annotation).
            while body_start != -1:
                # Check that this '{' is the function-body opener: it should
                # come right after ') {' or '}) {' on the same/adjacent line.
                preceding = mlir_string[max(0, body_start - 50) : body_start]
                if re.search(r"\)\s*$", preceding):
                    break
                body_start = mlir_string.find("{", body_start + 1)
        if body_start != -1:
            scan_text = _extract_balanced_parens(mlir_string, body_start, open_char="{")
        else:
            scan_text = mlir_string

        # Extract wrapped args, skipping jax.global_constant scalar args.
        wrapped_params = re.findall(arg_header_pattern, wrapped_sig_text)

        call_match = re.search(r"call\s+@_wrapped_jax_export_main\s*\(([^)]*)\)", mlir_string)
        if call_match:
            call_args = [a.strip() for a in call_match.group(1).split(",")]
            for i, (w_arg_id, _) in enumerate(wrapped_params):
                if i < len(call_args):
                    main_ref = call_args[i].strip()
                    if main_ref in seen_arg_ids:
                        # This wrapped arg corresponds to a real @main input.
                        arg_alias[w_arg_id] = main_ref
    else:
        # No @_wrapped_jax_export_main: extract just the @main function body so
        # that private helper functions (e.g. @relu) are NOT parsed at the
        # top level — they will be inlined when a ``call`` op is encountered.
        if public_func_match:
            paren_start_main = public_func_match.end() - 1
            sig_text_main = _extract_balanced_parens(mlir_string, paren_start_main)
            sig_end_main = paren_start_main + len(sig_text_main) + 2
            body_start_main = mlir_string.find("{", sig_end_main)
            while body_start_main != -1:
                preceding = mlir_string[max(0, body_start_main - 50) : body_start_main]
                if re.search(r"\)\s*$", preceding):
                    break
                body_start_main = mlir_string.find("{", body_start_main + 1)
            if body_start_main != -1:
                scan_text = _extract_balanced_parens(
                    mlir_string, body_start_main, open_char="{"
                )
            else:
                scan_text = mlir_string
        else:
            scan_text = mlir_string

    # 3. Parse the scan_text using all supported patterns, then sort.
    results += _parse_body(scan_text, arg_alias)

    # Sort all non-Input, non-return layers by their position in scan_text so
    # that compare and regular ops are interleaved in the correct execution
    # order.  Input layers captured from the header are always first.
    input_layers = [la for la in results if la["op"] == "Input"]
    compute_layers = [la for la in results if la["op"] not in ("Input", "return")]
    return_layers = [la for la in results if la["op"] == "return"]

    compute_layers.sort(key=lambda la: _get_layer_pos(la, scan_text))
    return input_layers + compute_layers + return_layers
