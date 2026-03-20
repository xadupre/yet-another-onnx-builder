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
)
from .xla_call_module_parsing import (  # noqa: F401
    _get_function_param_ids,
    _get_function_param_info,
    _parse_body,
    _parse_dense_value,
    _parse_tensor_type,
    parse_mlir,
)


def _process_constant_layer(
    layer: dict, local_results: Dict[str, str], g: GraphBuilderExtendedProtocol
) -> None:
    """Create an ONNX initializer for a ``constant`` StableHLO layer.

    Updates *local_results* in-place, mapping the layer's result id to the
    generated initializer name.

    :param layer: layer dict with keys ``id``, ``shape``, and ``dense_content``.
    :param local_results: mutable id→tensor-name mapping for the current scope.
    :param g: ONNX graph builder.
    """
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
    g.make_initializer(const_name, np_val)
    local_results[layer["id"]] = const_name


def _process_broadcast_layer(layer: dict, local_results: Dict[str, str]) -> None:
    """Handle a ``broadcast_in_dim`` or ``dynamic_broadcast_in_dim`` layer.

    ONNX arithmetic ops support implicit broadcasting, so we simply pass the
    operand tensor through unchanged.

    :param layer: layer dict with keys ``id`` and ``operands``.
    :param local_results: mutable id→tensor-name mapping for the current scope.
    """
    operand_id = layer["operands"][0]
    if operand_id in local_results:
        local_results[layer["id"]] = local_results[operand_id]


def _process_dot_general_layer(
    layer: dict, local_results: Dict[str, str], g: GraphBuilderExtendedProtocol
) -> None:
    """Emit a MatMul ONNX op for a ``dot_general`` StableHLO layer.

    :param layer: layer dict with keys ``id`` and ``operands`` (lhs, rhs).
    :param local_results: mutable id→tensor-name mapping for the current scope.
    :param g: ONNX graph builder.
    """
    lhs_id, rhs_id = layer["operands"]
    if lhs_id not in local_results or rhs_id not in local_results:
        return
    res = g.op.MatMul(local_results[lhs_id], local_results[rhs_id], name="XlaCallModule")
    local_results[layer["id"]] = res if isinstance(res, str) else res[0]


def _process_reduce_layer(
    layer: dict, local_results: Dict[str, str], g: GraphBuilderExtendedProtocol
) -> None:
    """Emit a ReduceMax or ReduceSum ONNX op for a ``reduce_*`` StableHLO layer.

    :param layer: layer dict with keys ``id``, ``op``, ``operands``, and ``axes``.
    :param local_results: mutable id→tensor-name mapping for the current scope.
    :param g: ONNX graph builder.
    """
    inp_id = layer["operands"][0]
    if inp_id not in local_results:
        return
    axes = layer.get("axes", [])
    np_axes = np.array(axes, dtype=np.int64)
    if layer["op"] == "reduce_max":
        res = g.op.ReduceMax(local_results[inp_id], np_axes, keepdims=1, name="XlaCallModule")
    else:
        res = g.op.ReduceSum(local_results[inp_id], np_axes, keepdims=1, name="XlaCallModule")
    local_results[layer["id"]] = res if isinstance(res, str) else res[0]


def _process_call_layer(
    layer: dict,
    local_results: Dict[str, str],
    g: GraphBuilderExtendedProtocol,
    decoded_module: str,
) -> None:
    """Inline a ``call`` StableHLO layer by parsing and executing its body.

    Looks up the named private function in *decoded_module*, maps caller
    arguments to the function's parameter IDs (skipping shape-only args), and
    processes the function body recursively via :func:`_process_layers`.

    :param layer: layer dict with keys ``id``, ``operands``, and ``func``.
    :param local_results: mutable id→tensor-name mapping for the current scope.
    :param g: ONNX graph builder.
    :param decoded_module: raw MLIR text used to look up the called function.
    """
    func_name = layer.get("func", "")
    call_args = layer["operands"]

    # Parse and inline the private function.
    # Use _get_function_param_info to correctly align tensor params
    # with their corresponding call arguments, skipping shape-only
    # (integer) parameters that appear in dynamic-shape functions.
    func_param_info = _get_function_param_info(decoded_module, func_name)
    func_body = _extract_function_body(decoded_module, func_name)
    if func_body is None:
        return

    # Build a local scope mapping function param IDs to caller args.
    inline_results: Dict[str, str] = {}
    for (param_id, is_shape_only), call_arg_id in zip(func_param_info, call_args):
        if is_shape_only:
            continue
        if call_arg_id in local_results:
            inline_results[param_id] = local_results[call_arg_id]

    # Parse the function body (scan only the body, with no arg alias).
    func_layers = _parse_body(func_body, {})

    # Process function body layers (stops at return layer).
    _process_layers(func_layers, inline_results, g, decoded_module)

    # Retrieve the return value from the return layer.
    ret_val: Optional[str] = None
    for sub_layer in func_layers:
        if sub_layer["op"] == "return":
            operands = sub_layer["operands"]
            if operands and operands[0] in inline_results:
                ret_val = inline_results[operands[0]]
            break

    if ret_val is not None:
        local_results[layer["id"]] = ret_val


def _process_layers(
    layer_list: List[dict],
    local_results: Dict[str, str],
    g: GraphBuilderExtendedProtocol,
    decoded_module: str,
) -> None:
    """Process a list of StableHLO layers, emitting ONNX ops into *g*.

    Updates *local_results* in-place mapping StableHLO result ids to ONNX
    tensor names.  Returns early when a ``return`` layer is encountered.

    :param layer_list: list of layer dicts produced by :func:`parse_mlir`.
    :param local_results: mutable id→tensor-name mapping for the current scope.
    :param g: ONNX graph builder.
    :param decoded_module: raw MLIR text (needed to inline ``call`` targets).
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
            _process_constant_layer(layer, local_results, g)
            continue

        if op_type in ("broadcast_in_dim", "dynamic_broadcast_in_dim"):
            _process_broadcast_layer(layer, local_results)
            continue

        if op_type == "dot_general":
            _process_dot_general_layer(layer, local_results, g)
            continue

        if op_type in ("reduce_max", "reduce_sum"):
            _process_reduce_layer(layer, local_results, g)
            continue

        if op_type == "call":
            _process_call_layer(layer, local_results, g, decoded_module)
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

        _process_layers([layer], results, g, decoded_module)

    raise NotImplementedError(
        f"Unable to convert XlaCallModule with the following assembly"
        f"\n{layers}\n{decoded_module}{g.get_debug_msg()}"
    )
