from __future__ import annotations
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


__all__ = ["OnnxGraph", "ProtoType", "Var", "Vars", "g", "start"]
