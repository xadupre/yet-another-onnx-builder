"""Helper function to create a single-operator OnnxPipe."""

from __future__ import annotations

from typing import Any

import onnx
from onnx import TensorProto, helper

from ._pipe import OnnxPipe


def op(
    op_type: str,
    *,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    input_type: int = TensorProto.FLOAT,
    output_type: int = TensorProto.FLOAT,
    domain: str = "",
    opset_version: int = 20,
    **kwargs: Any,
) -> OnnxPipe:
    """Create an :class:`~onnx_pipe.OnnxPipe` from a single ONNX operator.

    Args:
        op_type: ONNX operator name (e.g. ``"Abs"``, ``"Relu"``).
        input_names: Names for the graph inputs.  Defaults to ``["X"]``.
        output_names: Names for the graph outputs.  Defaults to ``["Y"]``.
        input_type: ONNX data type for inputs.  Defaults to ``FLOAT``.
        output_type: ONNX data type for outputs.  Defaults to ``FLOAT``.
        domain: Operator domain.  Defaults to the default ONNX domain.
        opset_version: Opset version for the generated model.  Defaults to 20.
        **kwargs: Additional attributes forwarded to
            :func:`onnx.helper.make_node`.

    Returns:
        An :class:`~onnx_pipe.OnnxPipe` wrapping the single-operator model.

    Example::

        from onnx_pipe import op
        import numpy as np

        pipe = op("Abs") | op("Relu")
        model = pipe.to_onnx()
    """
    if input_names is None:
        input_names = ["X"]
    if output_names is None:
        output_names = ["Y"]

    inputs = [
        helper.make_tensor_value_info(name, input_type, [None])
        for name in input_names
    ]
    outputs = [
        helper.make_tensor_value_info(name, output_type, [None])
        for name in output_names
    ]

    node = helper.make_node(op_type, input_names, output_names, domain=domain, **kwargs)
    graph = helper.make_graph([node], f"{op_type}_graph", inputs, outputs)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid(domain, opset_version)],
    )
    return OnnxPipe(model)
