import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class ConvBiasNullPattern(PatternOptimization):
    """
    Checks that a Conv has a null bias.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(512, 3, 64, 64)"])
            I_W(["W FLOAT(64, 3, 4, 4)"])
            i_B2["B2 FLOAT(64)"]

            Conv_0[["Conv(., ., .)"]]

            I_X -->|"FLOAT(512, 3, 64, 64)"| Conv_0
            I_W -->|"FLOAT(64, 3, 4, 4)"| Conv_0
            i_B2 -->|"FLOAT(64)"| Conv_0

            O_Y(["Y FLOAT(512, 64, 32, 32)"])
            Conv_0 --> O_Y

            class I_X,I_W,O_Y ioNode
            class i_B2 initNode
            class Conv_0 opNode
    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(512, 3, 64, 64)"])
            I_W(["W FLOAT(64, 3, 4, 4)"])

            Conv_0[["Conv(., .)"]]

            I_X -->|"FLOAT(512, 3, 64, 64)"| Conv_0
            I_W -->|"FLOAT(64, 3, 4, 4)"| Conv_0

            O_Y(["Y FLOAT(512, 64, 32, 32)"])
            Conv_0 --> O_Y

            class I_X,I_W,O_Y ioNode
            class Conv_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Conv" or node.domain != "":
            return self.none()
        if len(node.input) < 3:
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant(node.input[2]):
            return self.none(node, inspect.currentframe().f_lineno)

        cst = g.get_computed_constant(node.input[2])
        if cst is None or cst.min() != 0 or cst.max() != 0:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(self, g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
        new_node = g.make_node(
            "Conv",
            node.input[:2],
            node.output,
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        new_node.attribute.extend(node.attribute)
        return [new_node]
