import re
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol
from .jax_ops import _MAPPING_JAX_ONNX, _COMPOSITE_JAX_OPS, get_jax_cvt  # noqa: F401
from .xla_call_module_helper import (  # noqa: F401
    _MLIR_DTYPE_MAP,
    _extract_balanced_parens,
    _extract_function_body,
    _get_function_param_ids,
    _get_function_param_info,
    _parse_body,
    _parse_dense_value,
    _parse_tensor_type,
)


# ---------------------------------------------------------------------------
# Layer-list parser (extended)
# ---------------------------------------------------------------------------


@register_tf_op_converter("XlaCallModule")
def convert_exp(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """Convert a StableHLO ``XlaCallModule`` operation to ONNX.

    Supports the following StableHLO ops: unary/binary elementwise,
    ``dot_general`` (matrix multiply), ``broadcast_in_dim``,
    ``dynamic_broadcast_in_dim``, ``constant``, ``reduce`` (max/sum),
    ``call`` (to private functions), and comparison ops.
    """
    import jax.extend.mlir

    hlo_module = op.get_attr("module")
    decoded_module = jax.extend.mlir.deserialize_portable_artifact(hlo_module)
    layers = parse_mlir(decoded_module)
    results: Dict[str, str] = {}

    def _process_layers(layer_list: List[dict], local_results: Dict[str, str]) -> None:
        """Process a list of layers, updating *local_results* in-place.

        Returns early (after the ``return`` layer) so callers can detect the
        return value.
        """
        for layer in layer_list:
            op_type = layer["op"]

            if op_type == "Input":
                continue  # inputs already in local_results

            if op_type == "return":
                return

            if op_type == "skip":
                continue

            if op_type == "constant":
                # Create an ONNX initializer from the dense attribute.
                const_name = layer["id"]
                type_str = layer.get("shape", "")
                dense_content = layer.get("dense_content", "")
                shape, dtype = _parse_tensor_type(type_str)
                if shape is None or dtype is None:
                    # Fall back to float32 scalar
                    shape, dtype = (), np.float32
                try:
                    np_val = _parse_dense_value(dense_content, shape, dtype)
                except (ValueError, Exception) as exc:
                    warnings.warn(
                        f"XlaCallModule: failed to parse dense constant "
                        f"{const_name!r}: {exc}. Using zero fallback.",
                        stacklevel=2,
                    )
                    np_val = np.array(0, dtype=dtype)
                # Generate a unique name if the candidate is already taken
                # (e.g. when inlining a private function whose constants share
                # IDs with the outer-scope constants).
                candidate = const_name
                _dedup_idx = 0
                while g.has_name(candidate):
                    _dedup_idx += 1
                    candidate = f"{const_name}_{_dedup_idx}_dup"
                const_name = candidate
                g.make_initializer(const_name, np_val, source="XlaCallModule.constant")
                local_results[layer["id"]] = const_name
                continue

            if op_type in ("broadcast_in_dim", "dynamic_broadcast_in_dim"):
                # ONNX arithmetic ops support implicit broadcasting; pass through.
                operand_id = layer["operands"][0]
                if operand_id in local_results:
                    local_results[layer["id"]] = local_results[operand_id]
                continue

            if op_type == "dot_general":
                lhs_id, rhs_id = layer["operands"]
                if lhs_id not in local_results or rhs_id not in local_results:
                    continue
                res = g.op.MatMul(
                    local_results[lhs_id], local_results[rhs_id], name="XlaCallModule"
                )
                local_results[layer["id"]] = res if isinstance(res, str) else res[0]
                continue

            if op_type in ("reduce_max", "reduce_sum"):
                inp_id = layer["operands"][0]
                if inp_id not in local_results:
                    continue
                axes = layer.get("axes", [])
                np_axes = np.array(axes, dtype=np.int64)
                if op_type == "reduce_max":
                    res = g.op.ReduceMax(
                        local_results[inp_id], np_axes, keepdims=1, name="XlaCallModule"
                    )
                else:
                    res = g.op.ReduceSum(
                        local_results[inp_id], np_axes, keepdims=1, name="XlaCallModule"
                    )
                local_results[layer["id"]] = res if isinstance(res, str) else res[0]
                continue

            if op_type == "call":
                func_name = layer.get("func", "")
                call_args = layer["operands"]

                # Parse and inline the private function.
                # Use _get_function_param_info to correctly align tensor params
                # with their corresponding call arguments, skipping shape-only
                # (integer) parameters that appear in dynamic-shape functions.
                func_param_info = _get_function_param_info(decoded_module, func_name)
                func_body = _extract_function_body(decoded_module, func_name)
                if func_body is None:
                    continue

                # Build a local scope mapping function param IDs to caller args.
                inline_results: Dict[str, str] = {}
                for (param_id, is_shape_only), call_arg_id in zip(
                    func_param_info, call_args
                ):
                    if is_shape_only:
                        continue
                    if call_arg_id in local_results:
                        inline_results[param_id] = local_results[call_arg_id]

                # Parse the function body (scan only the body, with no arg alias).
                func_layers = _parse_body(func_body, {})

                # Process layers; track the return value.
                ret_val: Optional[str] = None
                for sub_layer in func_layers:
                    sub_op = sub_layer["op"]
                    if sub_op == "Input":
                        continue
                    if sub_op == "return":
                        operands = sub_layer["operands"]
                        if operands and operands[0] in inline_results:
                            ret_val = inline_results[operands[0]]
                        break
                    # Reuse _process_layers for the sub-layer (single-item list).
                    _process_layers([sub_layer], inline_results)

                if ret_val is not None:
                    local_results[layer["id"]] = ret_val
                continue

            # Generic elementwise op via get_jax_cvt.
            fct = get_jax_cvt(decoded_module, g, op_type)
            args_list = []
            for a in layer["operands"]:
                if a not in local_results:
                    break
                args_list.append(local_results[a])
            else:
                res = fct(*args_list, name="XlaCallModule")
                if isinstance(res, str):
                    res = (res,)
                if res:
                    local_results[layer["id"]] = res[0]

    for layer in layers:
        if layer["op"] == "Input":
            if layer["id"] not in results:
                results[layer["id"]] = op.inputs[len(results)].name
            continue

        if layer["op"] == "return":
            if len(layer["operands"]) == 1:
                return g.op.Identity(
                    results[layer["operands"][0]], outputs=outputs, name="XlaCallModule"
                )
            return tuple(
                g.op.Identity(results[a], outputs=outputs[i : i + 1])
                for i, a in enumerate(layer["operands"])
            )

        _process_layers([layer], results)

    raise NotImplementedError(
        f"Unable to convert XlaCallModule with the following assembly"
        f"\n{layers}\n{decoded_module}{g.get_debug_msg()}"
    )


def parse_mlir(mlir_string):
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

    def _layer_pos(layer: dict) -> int:
        """Return the character offset of this layer's result id in scan_text."""
        rid = layer.get("id", "")
        if rid and rid != "?":
            m = re.search(re.escape(rid) + r"\b", scan_text)
            if m:
                return m.start()
        return len(scan_text)

    compute_layers.sort(key=_layer_pos)
    return input_layers + compute_layers + return_layers
