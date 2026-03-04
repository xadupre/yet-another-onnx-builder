from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization, _get_lineno

if TYPE_CHECKING:
    from ...xbuilder.graph_builder import GraphBuilder
    from ..graph_builder_optim import GraphBuilderPatternOptimization



class ShapeBasedShapeShapeAddPattern(PatternOptimization):
    """
    Tries to find another way to get a dimension obtained with the addition of two.
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Add" or node.domain != "":
            return self.none()
        shape1 = g.node_before(node.input[0])
        if shape1 is None or shape1.op_type != "Shape" or shape1.domain != "":
            return self.none(node, _get_lineno())
        shape2 = g.node_before(node.input[1])
        if shape2 is None or shape2.op_type != "Shape" or shape2.domain != "":
            return self.none(node, _get_lineno())
        # ishape1 = g.get_shape_renamed(shape1.input[0])
        # ishape2 = g.get_shape_renamed(shape2.input[0])
        # value1 = g.builder.value_as_shape(node.input[0])
        # value2 = g.builder.value_as_shape(node.input[1])
        # input_shapes = [g.get_shape_renamed(i) for i in g.builder.input_names]
        # g.builder._known_value_shape
        # g.builder.constraints_)
        # g.builder.replacements_dimensions_
        return self.none(node, _get_lineno())
        # return MatchResult(self, [shape1, shape2, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        shape1_node: NodeProto,
        shape2_node: NodeProto,
        add_node: NodeProto,
    ) -> List[NodeProto]:
        raise NotImplementedError(f"{self.___class__.__name__} is not implemented yet.")
