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


def parse_mlir(mlir_string):
    results = []

    # 1. Capture Function Arguments (The Initial Inputs)
    arg_header_pattern = r"(%arg\d+)\s*:\s*(tensor<[^>]+>)"
    for arg_id, shape in re.findall(arg_header_pattern, mlir_string):
        results.append(
            {"id": arg_id, "op": "Input", "operands": tuple(), "shape": shape, "loc": "header"}
        )

    # 2. Comprehensive Pattern for Ops and Returns
    # Group 1: Result ID (%0)
    # Group 2: Op Name (stablehlo.sine)
    # Group 3: Operands (%arg0, or %0, %1)
    # Group 4: Shape (tensor<5x4xf32>)
    # Group 5: Location (#loc13)
    op_pattern = (
        r"(?:(%?\w+)\s*=\s*)?\"?([\w\.]+)\"?\s*(%[\w\s,%]+)?"
        r"\s*:\s*(?:.*?->\s*)?(tensor<[^>]+>).*?loc\((.*?)\)"
    )

    for match in re.finditer(op_pattern, mlir_string):
        res_id, op_name, operands, shape, location = match.groups()

        # Clean up operands (remove extra whitespace/newlines)
        clean_operands = operands.strip().split(",") if operands else tuple()
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


_MAPPING_JAX_ONNX = {"sine": "Sin"}


def get_jax_cvt(assembly_code: str, g: GraphBuilderExtendedProtocol, jax_type: str):
    if jax_type in _MAPPING_JAX_ONNX:
        return lambda *args, **kwargs: getattr(g.op, _MAPPING_JAX_ONNX[jax_type])(*args, **kwargs)
    raise RuntimeError(
        f"Unable to handle jax operator {jax_type!r} in\n{assembly_code}{g.get_debug_msg()}"
    )
