import re
from typing import Any, Dict, List
import tensorflow as tf
from ..register import register_tf_op_converter
from ...typing import GraphBuilderExtendedProtocol


@register_tf_op_converter("XlaCallModule")
def convert_exp(
    g: GraphBuilderExtendedProtocol, sts: Dict[str, Any], outputs: List[str], op: tf.Operation
) -> str:
    """Not fully implemented."""
    import jax.extend.mlir

    hlo_module = op.get_attr("module")
    decoded_module = jax.extend.mlir.deserialize_portable_artifact(hlo_module)
    layers = parse_mlir(decoded_module)
    results = {}
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

        fct = get_jax_cvt(decoded_module, g, layer["op"])
        args = [results[a] for a in layer["operands"]]
        res = fct(*args, name="XlaCallModule")
        if isinstance(res, str):
            res = (res,)
        assert len(res) == 1, f"Not yet implemented for layer={layer}{g.get_debug_msg()}"
        results[layer["id"]] = res[0]

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

    public_func_match = re.search(
        r"func\.func\s+public\s+@\w+\s*\(", mlir_string
    )
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

        call_match = re.search(
            r"call\s+@_wrapped_jax_export_main\s*\(([^)]*)\)", mlir_string
        )
        if call_match:
            call_args = [a.strip() for a in call_match.group(1).split(",")]
            for i, (w_arg_id, _) in enumerate(wrapped_params):
                if i < len(call_args):
                    main_ref = call_args[i].strip()
                    if main_ref in seen_arg_ids:
                        # This wrapped arg corresponds to a real @main input.
                        arg_alias[w_arg_id] = main_ref
    else:
        scan_text = mlir_string

    # 3. Comprehensive Pattern for Ops and Returns
    # Group 1: Result ID (%0)
    # Group 2: Op Name (stablehlo.sine)
    # Group 3: Operands (%arg0, or %0, %1)
    # Group 4: Shape (tensor<5x4xf32>)
    # Group 5: Location (#loc13)
    op_pattern = (
        r"(?:(%?\w+)\s*=\s*)?\"?([\w\.]+)\"?\s*(%[\w\s,%]+)?"
        r"\s*:\s*(?:.*?->\s*)?(tensor<[^>]+>).*?loc\((.*?)\)"
    )

    for match in re.finditer(op_pattern, scan_text):
        res_id, op_name, operands, shape, location = match.groups()

        # Clean up operands (remove extra whitespace/newlines) and apply the
        # alias map so references to @_wrapped args resolve to @main arg names.
        if operands:
            raw_operands = operands.strip().split(",")
            clean_operands = [arg_alias.get(o.strip(), o.strip()) for o in raw_operands]
        else:
            clean_operands = []
        clean_op = op_name.replace("stablehlo.", "")
        if not clean_operands:
            continue

        obs = {
            "id": res_id or "?",
            "op": clean_op,
            "operands": clean_operands,
            "shape": shape,
            "loc": location,
        }
        results.append(obs)
        if op_name == "return":
            break

    return results


# Mapping from StableHLO unary op names (after stripping the ``stablehlo.``
# prefix) to their direct ONNX-op-name equivalents.  Only truly 1-to-1 ops
# belong here; composite ops are handled in :func:`get_jax_cvt` below.
_MAPPING_JAX_ONNX: dict = {
    # Magnitude / rounding
    "abs": "Abs",
    "ceil": "Ceil",
    "floor": "Floor",
    "negate": "Neg",
    "round_nearest_even": "Round",
    "sign": "Sign",
    # Exponential / logarithm
    "exponential": "Exp",
    "log": "Log",
    # Trigonometric
    "cosine": "Cos",
    "sine": "Sin",
    # Hyperbolic
    "tanh": "Tanh",
    # Activation
    "logistic": "Sigmoid",
    # Square-root
    "sqrt": "Sqrt",
    # Bitwise / logical
    "not": "Not",
}


def _make_rsqrt(g: GraphBuilderExtendedProtocol):
    """Return a callable for ``stablehlo.rsqrt`` → ``Reciprocal(Sqrt(x))``."""

    def _rsqrt(*args, **kwargs):
        name = kwargs.pop("name", "rsqrt")
        outputs = kwargs.pop("outputs", None)
        (x,) = args
        sqrt = g.op.Sqrt(x, name=f"{name}_sqrt")
        kw = {"name": name}
        if outputs is not None:
            kw["outputs"] = outputs
        return g.op.Reciprocal(sqrt, **kw)

    return _rsqrt


def _make_log_plus_one(g: GraphBuilderExtendedProtocol):
    """Return a callable for ``stablehlo.log_plus_one`` → ``Log(Add(x, 1))``."""
    import numpy as np

    def _log1p(*args, **kwargs):
        name = kwargs.pop("name", "log_plus_one")
        outputs = kwargs.pop("outputs", None)
        (x,) = args
        one = np.array(1, dtype=np.float32)
        xp1 = g.op.Add(x, one, name=f"{name}_add")
        kw = {"name": name}
        if outputs is not None:
            kw["outputs"] = outputs
        return g.op.Log(xp1, **kw)

    return _log1p


def _make_exponential_minus_one(g: GraphBuilderExtendedProtocol):
    """Return a callable for ``stablehlo.exponential_minus_one`` → ``Sub(Exp(x), 1)``."""
    import numpy as np

    def _expm1(*args, **kwargs):
        name = kwargs.pop("name", "exponential_minus_one")
        outputs = kwargs.pop("outputs", None)
        (x,) = args
        one = np.array(1, dtype=np.float32)
        exp_x = g.op.Exp(x, name=f"{name}_exp")
        kw = {"name": name}
        if outputs is not None:
            kw["outputs"] = outputs
        return g.op.Sub(exp_x, one, **kw)

    return _expm1


# Factory functions for composite ops that cannot be expressed as a single
# ONNX op.  Each factory receives the :class:`GraphBuilderExtendedProtocol`
# instance and returns a callable with the same signature as a simple
# ``g.op.<OpName>`` call.
_COMPOSITE_JAX_OPS: dict = {
    "rsqrt": _make_rsqrt,
    "log_plus_one": _make_log_plus_one,
    "exponential_minus_one": _make_exponential_minus_one,
}


def get_jax_cvt(assembly_code: str, g: GraphBuilderExtendedProtocol, jax_type: str):
    if jax_type in _MAPPING_JAX_ONNX:
        return lambda *args, **kwargs: getattr(g.op, _MAPPING_JAX_ONNX[jax_type])(*args, **kwargs)
    if jax_type in _COMPOSITE_JAX_OPS:
        return _COMPOSITE_JAX_OPS[jax_type](g)
    raise RuntimeError(
        f"Unable to handle jax operator {jax_type!r} in\n{assembly_code}{g.get_debug_msg()}"
    )
