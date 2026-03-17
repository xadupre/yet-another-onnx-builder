import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class UnsqueezeEqualPattern(PatternOptimization):
    """
    Replaces the sequence R -> Equal -> Unsqueeze, R -> Unsqueeze,
    into R -> Unsqueeze -> Equal.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_Y(["Y FLOAT(a, 1, b)"])
            I_m_one(["m_one FLOAT(1)"])
            I_X(["X FLOAT(a, b)"])
            I_axis(["axis INT64(1)"])

            Constant_0[["Constant() -#gt; axis"]]
            Constant_1[["Constant() -#gt; m_one"]]
            Unsqueeze_2[["Unsqueeze(., .)"]]
            Equal_3[["Equal(., .)"]]
            Unsqueeze_4[["Unsqueeze(., .)"]]

            I_X -->|"FLOAT(a, b)"| Unsqueeze_2
            Constant_0 -->|"INT64(1)"| Unsqueeze_2
            I_X -->|"FLOAT(a, b)"| Equal_3
            Constant_1 -->|"FLOAT(1)"| Equal_3
            Equal_3 -->|"BOOL(a, b)"| Unsqueeze_4
            Constant_0 -->|"INT64(1)"| Unsqueeze_4

            O_Y(["Y FLOAT(a, 1, b)"])
            Unsqueeze_2 --> O_Y
            O_Z(["Z BOOL(a, 1, b)"])
            Unsqueeze_4 --> O_Z

            class I_Y,I_m_one,I_X,I_axis,O_Y,O_Z ioNode
            class Constant_0,Constant_1 constNode
            class Unsqueeze_2,Equal_3,Unsqueeze_4 opNode
    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_Y(["Y FLOAT(a, 1, b)"])
            I_m_one(["m_one FLOAT(1)"])
            I_X(["X FLOAT(a, b)"])
            I_axis(["axis INT64(1)"])

            Unsqueeze_0[["Unsqueeze(., .)"]]
            Equal_1[["Equal(., .)"]]

            I_X -->|"FLOAT(a, b)"| Unsqueeze_0
            I_axis -->|"INT64(1)"| Unsqueeze_0
            Unsqueeze_0 -->|"FLOAT(a, 1, b)"| Equal_1
            I_m_one -->|"FLOAT(1)"| Equal_1

            O_Y(["Y FLOAT(a, 1, b)"])
            Unsqueeze_0 --> O_Y
            O_Z(["Z BOOL(a, 1, b)"])
            Equal_1 --> O_Z

            class I_Y,I_m_one,I_X,I_axis,O_Y,O_Z ioNode
            class Unsqueeze_0,Equal_1 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Equal" or node.domain != "":
            return self.none()
        if not g.is_constant_scalar(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        after = g.next_nodes(node.output[0])
        if len(after) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        next_path = g.next_nodes(node.input[0])
        if len(next_path) != 2:
            return self.none(node, inspect.currentframe().f_lineno)

        if next_path[0].op_type == node.op_type and next_path[1].op_type == "Unsqueeze":
            if next_path[1].input[1] != after[0].input[1]:
                return self.none(node, inspect.currentframe().f_lineno)
            return MatchResult(self, [next_path[1], node, after[0]], self.apply)
        if next_path[1].op_type == node.op_type and next_path[0].op_type == "Unsqueeze":
            if next_path[0].input[1] != after[0].input[1]:
                return self.none(node, inspect.currentframe().f_lineno)
            return MatchResult(self, [next_path[0], node, after[0]], self.apply)
        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_unsqueeze: NodeProto,
        node_equal: NodeProto,
        node_equal_unsqueeze: NodeProto,
    ) -> List[NodeProto]:
        return [
            node_unsqueeze,
            g.make_node(
                node_equal.op_type,
                [node_unsqueeze.output[0], node_equal.input[1]],
                [node_equal_unsqueeze.output[0]],
                domain=node_equal.domain,
                name=f"{self.__class__.__name__}--{node_equal.name}",
            ),
        ]
