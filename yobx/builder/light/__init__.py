"""
Light API for building ONNX graphs using a fluent, chained syntax.

Inspired by the ``light_api`` module in
`onnx-array-api <https://github.com/sdpython/onnx-array-api>`_.

Typical usage::

    from yobx.builder.light import start

    # Single-input model: Y = Neg(X)
    onx = start().vin("X").Neg().rename("Y").vout().to_onnx()

    # Two-input model: Z = Add(X, Y)
    onx = (
        start()
        .vin("X")
        .vin("Y")
        .bring("X", "Y")
        .Add()
        .rename("Z")
        .vout()
        .to_onnx()
    )
"""

from typing import Dict, Optional
from ._graph import OnnxGraph, ProtoType
from ._var import Var, Vars


def start(
    opset: Optional[int] = None,
    opsets: Optional[Dict[str, int]] = None,
    ir_version: Optional[int] = None,
) -> OnnxGraph:
    """
    Creates a new :class:`OnnxGraph` to start building an ONNX model.

    :param opset: main opset version
    :param opsets: additional opsets as ``{domain: version}``
    :param ir_version: ONNX IR version
    :return: :class:`OnnxGraph`

    Example::

        from yobx.builder.light import start

        onx = start().vin("X").Neg().rename("Y").vout().to_onnx()
    """
    return OnnxGraph(opset=opset, opsets=opsets, ir_version=ir_version)


def g() -> OnnxGraph:
    """
    Creates a new :class:`OnnxGraph` for use as a subgraph (e.g. inside ``If``).

    :return: :class:`OnnxGraph` with ``proto_type=ProtoType.GRAPH``
    """
    return OnnxGraph(proto_type=ProtoType.GRAPH)


__all__ = ["start", "g", "OnnxGraph", "ProtoType", "Var", "Vars"]
