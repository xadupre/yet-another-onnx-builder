"""
Converter for the TF ``BiasAdd`` op → ONNX ``Add``.
"""

from ..register import register_tf_op_converter
from ..tensorflow_helper import sanitize_name
from ...xbuilder import GraphBuilder


@register_tf_op_converter("BiasAdd")
def convert_bias_add(
    g: GraphBuilder, sts: dict, outputs: list, op, verbose: int = 0
) -> None:
    """Converts TF ``BiasAdd`` to ONNX ``Add``.

    TF's ``BiasAdd`` is semantically equivalent to adding a 1-D bias along the
    last dimension of the input, which maps directly to ONNX ``Add`` with
    numpy-style broadcasting.
    """
    a = sts.get(op.inputs[0].name)
    b = sts.get(op.inputs[1].name)
    if a is None or b is None:
        if verbose:
            print(f"[BiasAdd] missing input(s) for op {op.name!r}")
        return

    result = g.op.Add(a, b, outputs=outputs[:1], name=sanitize_name(op.name))
    assert isinstance(result, str)
    sts[op.outputs[0].name] = result
