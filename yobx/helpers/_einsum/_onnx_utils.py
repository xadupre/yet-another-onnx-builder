"""
Minimal ONNX graph utilities for the einsum decomposition package.
"""

from typing import Set
import onnx


def onnx_remove_node_unused(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Removes nodes from *model* whose outputs are never consumed (dead code).

    Walks the graph backwards from the declared outputs and marks every node
    that contributes to at least one output.  Nodes that are not marked are
    dropped.  Initializers that are no longer referenced are also removed.

    :param model: input ONNX model
    :return: new :class:`onnx.ModelProto` without unreachable nodes
    """
    graph = model.graph

    # Collect names that are graph outputs or consumed by at least one node.
    needed: Set[str] = set()
    for out in graph.output:
        needed.add(out.name)

    # Iterate backwards until stable.
    changed = True
    while changed:
        changed = False
        for node in graph.node:
            if any(o in needed for o in node.output):
                for inp in node.input:
                    if inp and inp not in needed:
                        needed.add(inp)
                        changed = True

    # Keep nodes whose outputs are (partially) needed.
    kept_nodes = [n for n in graph.node if any(o in needed for o in n.output)]

    # Keep initializers still referenced.
    kept_inits = [i for i in graph.initializer if i.name in needed]

    new_graph = onnx.helper.make_graph(
        kept_nodes, graph.name, list(graph.input), list(graph.output), initializer=kept_inits
    )

    new_model = onnx.helper.make_model(
        new_graph,
        ir_version=model.ir_version,
        opset_imports=list(model.opset_import),
        producer_name=model.producer_name,
        producer_version=model.producer_version,
    )
    return new_model
