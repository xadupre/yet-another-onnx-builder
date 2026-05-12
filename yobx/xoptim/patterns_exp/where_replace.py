import inspect
from typing import List, Optional
from onnx import NodeProto, TensorProto
from ..patterns_api import MatchResult, PatternOptimization


class ReplaceZeroPattern(PatternOptimization):
    """
    Replaces Where(bool(X), value, X) into ReplaceZero(X, by=by).

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(UNKNOWNDIM, UNKNOWNDIM1)"])

            Cast_0[["Cast(., to=BOOL)"]]
            Where_1[["Where(., [5.67], .)"]]

            I_X -->|"FLOAT(UNKNOWNDIM, UNKNOWNDIM1)"| Cast_0
            Cast_0 -->|"BOOL(UNKNOWNDIM, UNKNOWNDIM1)"| Where_1
            I_X -->|"FLOAT(UNKNOWNDIM, UNKNOWNDIM1)"| Where_1

            O_Y(["Y FLOAT(UNKNOWNDIM2, UNKNOWNDIM3)"])
            Where_1 --> O_Y

            class I_X,O_Y ioNode
            class Cast_0,Where_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(UNKNOWNDIM, UNKNOWNDIM1)"])

            ReplaceZero_0[["yaourt.ortops.fused_kernel.cuda.ReplaceZero(.)"]]

            I_X -->|"FLOAT(UNKNOWNDIM, UNKNOWNDIM1)"| ReplaceZero_0

            O_Y(["Y FLOAT(UNKNOWNDIM2, UNKNOWNDIM3)"])
            ReplaceZero_0 --> O_Y

            class I_X,O_Y ioNode
            class ReplaceZero_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type != "Where" or node.domain != "":
            return self.none()

        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        cast_node = g.node_before(node.input[0])
        if cast_node is None or cast_node.op_type != "Cast" or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        to = g.get_attribute(cast_node, "to").i
        if to != TensorProto.BOOL:
            return self.none(node, inspect.currentframe().f_lineno)

        if node.input[2] != cast_node.input[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [cast_node, node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", cast_node: NodeProto, where_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        cst = g.get_constant_scalar(where_node.input[1])
        new_node = g.make_node(
            "ReplaceZero",
            cast_node.input,
            where_node.output,
            by=cst,
            equal=False,
            name=f"{self.__class__.__name__}--{where_node.name}",
            domain="yaourt.ortops.fused_kernel.cuda",
        )
        return [new_node]
