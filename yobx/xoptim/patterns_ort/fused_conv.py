import inspect
from typing import List, Optional
from onnx import NodeProto, TensorProto
from ..patterns_api import MatchResult, PatternOptimization


class FusedConvPattern(PatternOptimization):
    """
    Replaces the Conv + Relu into FusedConv.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_W(["W FLOAT(8, 8, 3, 3)"])
            I_X(["X FLOAT(1, 8, 6, 6)"])
            I_B(["B FLOAT(8)"])

            Conv_0[["Conv(., ., .)"]]
            Relu_1[["Relu(.)"]]

            I_X -->|"FLOAT(1, 8, 6, 6)"| Conv_0
            I_W -->|"FLOAT(8, 8, 3, 3)"| Conv_0
            I_B -->|"FLOAT(8)"| Conv_0
            Conv_0 -->|"FLOAT(1, 8, 6, 6)"| Relu_1

            O_Y(["Y FLOAT(1, 8, 6, 6)"])
            Relu_1 --> O_Y

            class I_W,I_X,I_B,O_Y ioNode
            class Conv_0,Relu_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_W(["W FLOAT(8, 8, 3, 3)"])
            I_X(["X FLOAT(1, 8, 6, 6)"])
            I_B(["B FLOAT(8)"])

            FusedConv_0[["com.microsoft.FusedConv(., ., .)"]]

            I_X -->|"FLOAT(1, 8, 6, 6)"| FusedConv_0
            I_W -->|"FLOAT(8, 8, 3, 3)"| FusedConv_0
            I_B -->|"FLOAT(8)"| FusedConv_0

            O_Y(["Y FLOAT(1, 8, 6, 6)"])
            FusedConv_0 --> O_Y

            class I_W,I_X,I_B,O_Y ioNode
            class FusedConv_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Conv" or node.domain != "":
            return self.none()

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        op_type = next_nodes[0].op_type
        if op_type != "Relu":
            return self.none(node, inspect.currentframe().f_lineno)

        # FusedConv only exists for float32.
        dtypes = [(g.get_type(i) if g.has_type(i) else None) for i in node.input]
        if TensorProto.FLOAT not in dtypes:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_nodes[0]], self.apply, insert_at=next_nodes[0])

    def apply(
        self, g: "GraphBuilder", node: NodeProto, node_act: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        fc = g.make_node(
            "FusedConv",
            node.input,
            node_act.output,
            domain="com.microsoft",
            activation=node_act.op_type,
            name=f"{self.__class__.__name__}--{node.name}",
        )
        fc.attribute.extend(node.attribute)
        return [fc]
