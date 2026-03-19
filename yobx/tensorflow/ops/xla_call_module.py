import re
import struct
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol
from .jax_ops import _MAPPING_JAX_ONNX, _COMPOSITE_JAX_OPS, get_jax_cvt  # noqa: F401

# ---------------------------------------------------------------------------
# Tensor-type and dense-value helpers
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


def _parse_tensor_type(type_str: str) -> Tuple[Optional[tuple], Optional[type]]:
    """Parse a MLIR tensor type string into ``(shape, dtype)``.

    Returns ``(None, None)`` when parsing fails.

    Examples::

        _parse_tensor_type("tensor<8x16xf32>")   -> ((8, 16), np.float32)
        _parse_tensor_type("tensor<f32>")         -> ((), np.float32)
        _parse_tensor_type("tensor<?x10xf32>")    -> ((-1, 10), np.float32)
        _parse_tensor_type("tensor<i32>")         -> ((), np.int32)
        _parse_tensor_type("tensor<1xi32>")       -> ((1,), np.int32)
    """
    m = re.match(r"tensor<([^>]+)>", type_str.strip())
    if not m:
        return None, None
    inner = m.group(1)
    for dtype_str in sorted(_MLIR_DTYPE_MAP, key=len, reverse=True):
        if inner == dtype_str:
            return (), _MLIR_DTYPE_MAP[dtype_str]
        if inner.endswith("x" + dtype_str):
            shape_part = inner[: -(len(dtype_str) + 1)]
            shape = tuple(int(d) if d != "?" else -1 for d in shape_part.split("x"))
            return shape, _MLIR_DTYPE_MAP[dtype_str]
    return None, None


def _parse_dense_value(dense_content: str, shape: tuple, dtype: type) -> np.ndarray:
    """Parse the content of a ``dense<VALUE>`` attribute into a numpy array.

    :param dense_content: text between ``<`` and ``>`` in ``dense<...>``.
    :param shape: tensor shape tuple (may contain ``-1`` for dynamic dims).
    :param dtype: numpy element dtype.
    """
    dense_content = dense_content.strip()
    static_shape = tuple(d for d in shape if d != -1)

    if dense_content.startswith('"'):
        # Hex-encoded binary blob: "0xABCDEF…"
        inner = dense_content.strip('"')
        if inner.startswith(("0x", "0X")):
            inner = inner[2:]
        raw = bytes.fromhex(inner)
        arr = np.frombuffer(raw, dtype=dtype).copy()
        if static_shape:
            arr = arr.reshape(static_shape)
        return arr

    if dense_content.startswith(("0x", "0X")):
        # Single-element hex scalar: e.g. 0xFF800000 (-inf for f32).
        # In StableHLO text, the hex is the big-endian bit pattern.
        hex_str = dense_content[2:]
        nbytes = np.dtype(dtype).itemsize
        hex_str_padded = hex_str.zfill(nbytes * 2)
        raw = bytes.fromhex(hex_str_padded)
        # Decode as big-endian value of the appropriate type.
        # ">e" is the struct format code for big-endian IEEE 754 float16
        # (supported since Python 3.6 / struct docs).
        _be_fmt = {1: ">b", 2: ">e", 4: ">f", 8: ">d"}
        fmt = _be_fmt.get(nbytes)
        if fmt is None or dtype in (np.int32, np.int64, np.uint32, np.uint64, np.bool_):
            # For integer dtypes use numpy's dtype-aware conversion to handle
            # two's-complement sign correctly.
            scalar = np.frombuffer(raw[::-1], dtype=dtype)[0]
        else:
            scalar = struct.unpack(fmt, raw)[0]
        if static_shape:
            return np.full(static_shape, scalar, dtype=dtype)
        return np.array(scalar, dtype=dtype)

    if dense_content.startswith("["):
        # Nested list: [[v1, v2], [v3, v4]]
        import ast

        data = ast.literal_eval(dense_content)
        return np.array(data, dtype=dtype)

    # Single scalar value (decimal or integer)
    try:
        val = float(dense_content)
    except ValueError:
        val = int(dense_content)
    if static_shape:
        return np.full(static_shape, val, dtype=dtype)
    return np.array(val, dtype=dtype)


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


def _get_function_param_ids(mlir_string: str, func_name: str) -> List[str]:
    """Return the ``%argN`` parameter IDs of a named function (tensors only).

    Integer-typed parameters (e.g. ``tensor<i32>``) are typically shape
    constants injected by jax2tf and are excluded from the list.
    """
    pattern = rf"func\.func\s+(?:public|private)\s+@{re.escape(func_name)}\s*\("
    m = re.search(pattern, mlir_string)
    if not m:
        return []
    paren_start = m.end() - 1
    sig_text = _extract_balanced_parens(mlir_string, paren_start)
    arg_header_pattern = r"(%arg\d+)\s*:\s*(tensor<[^>]+>)"
    params = []
    for arg_id, type_str in re.findall(arg_header_pattern, sig_text):
        shape, dtype = _parse_tensor_type(type_str)
        if shape is None:
            params.append(arg_id)
            continue
        # Skip pure shape/integer constant args (jax.global_constant)
        # Detect: integer dtype AND scalar or 1-element shape
        is_int_scalar = dtype in (np.int32, np.int64, np.uint32, np.uint64) and len(shape) <= 1
        if not is_int_scalar:
            params.append(arg_id)
    return params


def _get_function_param_info(mlir_string: str, func_name: str) -> List[Tuple]:
    """Return ``[(param_id, is_shape_only), ...]`` for a named function.

    Shape-only parameters are integer scalars/vectors injected by jax2tf
    for dynamic-shape bookkeeping.  They should be skipped when building the
    argument map for inline calls.
    """
    pattern = rf"func\.func\s+(?:public|private)\s+@{re.escape(func_name)}\s*\("
    m = re.search(pattern, mlir_string)
    if not m:
        return []
    paren_start = m.end() - 1
    sig_text = _extract_balanced_parens(mlir_string, paren_start)
    arg_header_pattern = r"(%arg\d+)\s*:\s*(tensor<[^>]+>)"
    info: List[Tuple] = []
    for arg_id, type_str in re.findall(arg_header_pattern, sig_text):
        shape, dtype = _parse_tensor_type(type_str)
        if shape is None:
            info.append((arg_id, False))
            continue
        # jax2tf injects integer scalar/vector parameters as shape bookkeeping
        # variables annotated with ``jax.global_constant``.  These are:
        # * dimension-variable scalars: tensor<i32>  (shape = ())
        # * single-element shape tensors: tensor<1xi32>  (shape = (1,))
        # Both have len(shape) <= 1 and integer dtype, so we identify them as
        # shape-only (True) and skip them when building the inline arg map.
        is_shape_only = (
            dtype in (np.int32, np.int64, np.uint32, np.uint64) and len(shape) <= 1
        )
        info.append((arg_id, is_shape_only))
    return info


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
                except (ValueError, struct.error, Exception) as exc:
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
            try:
                fct = get_jax_cvt(decoded_module, g, op_type)
            except RuntimeError:
                warnings.warn(
                    f"XlaCallModule: unsupported StableHLO op {op_type!r} "
                    f"(layer id={layer.get('id')!r}); skipping.",
                    stacklevel=2,
                )
                continue
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


def _parse_body(scan_text: str, arg_alias: dict) -> list:
    """Parse a single StableHLO function body into a list of layer dicts.

    This is the shared parsing kernel used by both :func:`parse_mlir` (for the
    main / wrapped function body) and by the ``call``-inlining code in
    :func:`convert_exp` (for private helper function bodies).

    :param scan_text: the text between ``{`` and ``}`` of the function body.
    :param arg_alias: mapping from local ``%argN`` ids to caller-scope ids
        (used when parsing the ``@_wrapped_jax_export_main`` body).
    :returns: list of layer dicts (unsorted; caller is responsible for sorting
        if needed).
    """
    layers: list = []
    all_special_spans: set = set()

    # ------------------------------------------------------------------
    # 3a. stablehlo.constant
    # ------------------------------------------------------------------
    # Matches: %id = stablehlo.constant dense<VALUE> : tensor<...> loc(...)
    # The dense value can be:
    #   * "0xHEXBYTES"        – raw binary
    #   * 0xFF800000           – single hex scalar
    #   * 0.000000e+00         – floating-point scalar
    #   * [[v1, v2], ...]      – nested list
    const_pattern = (
        r'(%[\w]+)\s*=\s*stablehlo\.constant\s+'
        r'(dense<(?:"[^"]*"|[^>]*)>)'  # dense<...>
        r'\s*:\s*(tensor<[^>]+>)'  # : tensor<...>
        r'[^\n]*loc\(([^)]*)\)'
    )
    for m in re.finditer(const_pattern, scan_text):
        all_special_spans.update(range(m.start(), m.end()))
        res_id = arg_alias.get(m.group(1), m.group(1))
        dense_attr = m.group(2)  # dense<...>
        # Strip the dense< prefix and > suffix.
        dense_content = dense_attr[len("dense<") : -1]
        layers.append(
            {
                "id": res_id,
                "op": "constant",
                "operands": [],
                "shape": m.group(3),
                "loc": m.group(4),
                "dense_content": dense_content,
            }
        )

    # ------------------------------------------------------------------
    # 3b. stablehlo.dot_general
    # ------------------------------------------------------------------
    dot_pattern = (
        r'(%\w+)\s*=\s*stablehlo\.dot_general\s+'
        r'(%\w+)\s*,\s*(%\w+)\s*,\s*'
        r'contracting_dims\s*=\s*\[([^\]]*)\]\s*x\s*\[([^\]]*)\]'
        r'[^:]*:[^-]*->\s*(tensor<[^>]+>)'
        r'[^\n]*loc\(([^)]*)\)'
    )
    for m in re.finditer(dot_pattern, scan_text):
        all_special_spans.update(range(m.start(), m.end()))
        res_id = m.group(1)
        lhs = arg_alias.get(m.group(2), m.group(2))
        rhs = arg_alias.get(m.group(3), m.group(3))
        lhs_dims = [int(d.strip()) for d in m.group(4).split(",") if d.strip()]
        rhs_dims = [int(d.strip()) for d in m.group(5).split(",") if d.strip()]
        layers.append(
            {
                "id": res_id,
                "op": "dot_general",
                "operands": [lhs, rhs],
                "shape": m.group(6),
                "loc": m.group(7),
                "lhs_contracting": lhs_dims,
                "rhs_contracting": rhs_dims,
            }
        )

    # ------------------------------------------------------------------
    # 3c. stablehlo.broadcast_in_dim  (static shapes)
    # ------------------------------------------------------------------
    bcast_pattern = (
        r'(%\w+)\s*=\s*stablehlo\.broadcast_in_dim\s+'
        r'(%\w+)\s*,\s*dims\s*=\s*\[([^\]]*)\]'
        r'\s*:\s*\([^)]+\)\s*->\s*(tensor<[^>]+>)'
        r'[^\n]*loc\(([^)]*)\)'
    )
    for m in re.finditer(bcast_pattern, scan_text):
        all_special_spans.update(range(m.start(), m.end()))
        res_id = m.group(1)
        operand = arg_alias.get(m.group(2), m.group(2))
        dims = [int(d.strip()) for d in m.group(3).split(",") if d.strip()]
        layers.append(
            {
                "id": res_id,
                "op": "broadcast_in_dim",
                "operands": [operand],
                "shape": m.group(4),
                "loc": m.group(5),
                "dims": dims,
            }
        )

    # ------------------------------------------------------------------
    # 3d. stablehlo.dynamic_broadcast_in_dim
    # ------------------------------------------------------------------
    dyn_bcast_pattern = (
        r'(%\w+)\s*=\s*stablehlo\.dynamic_broadcast_in_dim\s+'
        r'(%\w+)\s*,\s*(%\w+)\s*,\s*dims\s*=\s*\[([^\]]*)\]'
        r'\s*:\s*\([^)]+\)\s*->\s*(tensor<[^>]+>)'
        r'[^\n]*loc\(([^)]*)\)'
    )
    for m in re.finditer(dyn_bcast_pattern, scan_text):
        all_special_spans.update(range(m.start(), m.end()))
        res_id = m.group(1)
        # Only keep the tensor operand; the shape tensor is not needed.
        operand = arg_alias.get(m.group(2), m.group(2))
        dims = [int(d.strip()) for d in m.group(4).split(",") if d.strip()]
        layers.append(
            {
                "id": res_id,
                "op": "dynamic_broadcast_in_dim",
                "operands": [operand],
                "shape": m.group(5),
                "loc": m.group(6),
                "dims": dims,
            }
        )

    # ------------------------------------------------------------------
    # 3e. call @funcname(args) → call layer
    # ------------------------------------------------------------------
    call_pattern = (
        r'(%\w+)\s*=\s*call\s+@(\w+)\s*\(([^)]*)\)'
        r'\s*:\s*\([^)]*\)\s*->\s*(tensor<[^>]+>)'
        r'[^\n]*loc\(([^)]*)\)'
    )
    for m in re.finditer(call_pattern, scan_text):
        # Do NOT intercept @_wrapped_jax_export_main calls; they are handled
        # by the outer parse_mlir logic via the wrapped-function body.
        if m.group(2) == "_wrapped_jax_export_main":
            all_special_spans.update(range(m.start(), m.end()))
            continue
        all_special_spans.update(range(m.start(), m.end()))
        res_id = m.group(1)
        func_name = m.group(2)
        raw_args = [a.strip() for a in m.group(3).split(",") if a.strip()]
        args = [arg_alias.get(a, a) for a in raw_args]
        layers.append(
            {
                "id": res_id,
                "op": "call",
                "operands": args,
                "shape": m.group(4),
                "loc": m.group(5),
                "func": func_name,
            }
        )

    # ------------------------------------------------------------------
    # 3f. stablehlo.reduce  (max / sum)
    # ------------------------------------------------------------------
    # Pattern: %0 = stablehlo.reduce(%arg init: %init) applies stablehlo.OP
    #          across dimensions = [N] : (...) -> tensor<...> loc(...)
    reduce_pattern = (
        r'(%\w+)\s*=\s*stablehlo\.reduce\s*\('
        r'(%\w+)\s+init:\s+(%\w+)\s*\)\s*applies\s+stablehlo\.(\w+)\s+'
        r'across\s+dimensions\s*=\s*\[([^\]]*)\]'
        r'[^:]*:[^-]*->\s*(tensor<[^>]+>)'
        r'[^\n]*loc\(([^)]*)\)'
    )
    _REDUCE_OP_MAP = {"maximum": "reduce_max", "add": "reduce_sum", "minimum": "reduce_min"}
    for m in re.finditer(reduce_pattern, scan_text):
        all_special_spans.update(range(m.start(), m.end()))
        res_id = m.group(1)
        operand = arg_alias.get(m.group(2), m.group(2))
        reduce_op = m.group(4)
        axes = [int(d.strip()) for d in m.group(5).split(",") if d.strip()]
        onnx_op = _REDUCE_OP_MAP.get(reduce_op, f"reduce_{reduce_op}")
        layers.append(
            {
                "id": res_id,
                "op": onnx_op,
                "operands": [operand],
                "shape": m.group(6),
                "loc": m.group(7),
                "axes": axes,
            }
        )

    # ------------------------------------------------------------------
    # 3g. Shape-only ops: reshape, concatenate, get_dimension_size,
    #     custom_call @shape_assertion  → "skip"
    # ------------------------------------------------------------------
    # These ops manipulate integer shape tensors produced by
    # ``stablehlo.get_dimension_size`` and are only needed for the
    # ``dynamic_broadcast_in_dim`` shape argument, which we elide.
    skip_patterns = [
        # stablehlo.reshape of integer-typed tensors (shape-only ops).
        # We match only when both input and output types end with an integer
        # dtype suffix (i8, i16, i32, i64, ui8, ui32, ui64) to avoid
        # accidentally skipping reshapes of float tensors.
        (
            r'(%\w+)\s*=\s*stablehlo\.reshape\s+(%\w+)'
            r'\s*:\s*\(tensor<[^>]*x?(?:i|ui)\d+>\)\s*->\s*tensor<[^>]*x?(?:i|ui)\d+>'
            r'[^\n]*loc\(([^)]*)\)'
        ),
        # stablehlo.concatenate of integer tensors
        (
            r'(%\w+)\s*=\s*stablehlo\.concatenate\s+(%\w+)[^:]*'
            r':\s*\(tensor<[^>]*x?(?:i|ui)\d+>[^)]*\)\s*->\s*tensor<[^>]*x?(?:i|ui)\d+>'
            r'[^\n]*loc\(([^)]*)\)'
        ),
        # stablehlo.get_dimension_size
        (
            r'(%\w+)\s*=\s*stablehlo\.get_dimension_size\s+%\w+[^\n]*'
            r'loc\(([^)]*)\)'
        ),
    ]
    for pattern in skip_patterns:
        for m in re.finditer(pattern, scan_text):
            all_special_spans.update(range(m.start(), m.end()))
            res_id = arg_alias.get(m.group(1), m.group(1))
            layers.append(
                {"id": res_id, "op": "skip", "operands": [], "shape": "", "loc": ""}
            )

    # stablehlo.custom_call (no result; side-effect only)
    custom_call_pat = r'stablehlo\.custom_call\s+@\w+[^\n]*loc\([^)]*\)'
    for m in re.finditer(custom_call_pat, scan_text):
        all_special_spans.update(range(m.start(), m.end()))

    # ------------------------------------------------------------------
    # 3h. stablehlo.compare  (existing logic, extended with span tracking)
    # ------------------------------------------------------------------
    compare_pattern = (
        r"(%\w+)\s*=\s*stablehlo\.compare\s+(\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)"
        r"[^:]*:\s*(?:.*?->\s*)?(tensor<[^>]+>).*?loc\((.*?)\)"
    )
    compare_spans: set = set()
    for match in re.finditer(compare_pattern, scan_text):
        compare_spans.update(range(match.start(), match.end()))
        all_special_spans.update(range(match.start(), match.end()))
        res_id, direction, op1, op2, shape, location = match.groups()
        op1 = arg_alias.get(op1, op1)
        op2 = arg_alias.get(op2, op2)
        layers.append(
            {
                "id": res_id,
                "op": f"compare_{direction}",
                "operands": [op1, op2],
                "shape": shape,
                "loc": location,
            }
        )

    # ------------------------------------------------------------------
    # 3i. General op pattern (existing logic, skipping already-handled spans)
    # ------------------------------------------------------------------
    op_pattern = (
        r"(?:(%?\w+)\s*=\s*)?\"?([\w\.]+)\"?\s*(%[\w\s,%]+)?"
        r"\s*:\s*(?:.*?->\s*)?(tensor<[^>]+>).*?loc\((.*?)\)"
    )
    for match in re.finditer(op_pattern, scan_text):
        if match.start() in all_special_spans:
            continue

        res_id, op_name, operands, shape, location = match.groups()

        if operands:
            raw_operands = operands.strip().split(",")
            clean_operands = [arg_alias.get(o.strip(), o.strip()) for o in raw_operands]
        else:
            clean_operands = []
        clean_op = op_name.replace("stablehlo.", "")
        if not clean_operands:
            continue

        layers.append(
            {
                "id": res_id or "?",
                "op": clean_op,
                "operands": clean_operands,
                "shape": shape,
                "loc": location,
            }
        )
        if op_name == "return":
            break

    return layers
