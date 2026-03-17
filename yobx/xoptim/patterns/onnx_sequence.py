import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class SequenceConstructAtPattern(PatternOptimization):
    """
    Replaces the sequence ``SequenceConstruct(x1, x2, ...)`` followed
    by ``SequenceAt(seq, 0)``, ``SequenceAt(seq, 1)``, ...

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X1(["X1 FLOAT(a, b)"])
            I_X2(["X2 FLOAT(c, d)"])

            SequenceConstruct_0[["SequenceConstruct(., .)"]]
            SequenceAt_1[["SequenceAt(., 0)"]]
            SequenceAt_2[["SequenceAt(., 1)"]]

            I_X1 -->|"FLOAT(a, b)"| SequenceConstruct_0
            I_X2 -->|"FLOAT(c, d)"| SequenceConstruct_0
            SequenceConstruct_0 --> SequenceAt_1
            SequenceConstruct_0 --> SequenceAt_2

            O_Y1(["Y1 FLOAT(a, b)"])
            SequenceAt_1 --> O_Y1
            O_Y2(["Y2 FLOAT(c, d)"])
            SequenceAt_2 --> O_Y2

            class I_X1,I_X2,O_Y1,O_Y2 ioNode
            class SequenceConstruct_0,SequenceAt_1,SequenceAt_2 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X1(["X1 FLOAT(a, b)"])
            I_X2(["X2 FLOAT(c, d)"])

            Identity_0[["Identity(.)"]]
            Identity_1[["Identity(.)"]]

            I_X1 -->|"FLOAT(a, b)"| Identity_0
            I_X2 -->|"FLOAT(c, d)"| Identity_1

            O_Y1(["Y1 FLOAT(a, b)"])
            Identity_0 --> O_Y1
            O_Y2(["Y2 FLOAT(c, d)"])
            Identity_1 --> O_Y2

            class I_X1,I_X2,O_Y1,O_Y2 ioNode
            class Identity_0,Identity_1 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "SequenceConstruct" or node.domain != "":
            return self.none()

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != len(node.input):
            return self.none(node, inspect.currentframe().f_lineno)
        if any(n.op_type != "SequenceAt" for n in next_nodes):
            return self.none(node, inspect.currentframe().f_lineno)

        ats = [n.input[1] for n in next_nodes]
        if any(not g.is_constant_scalar(a) for a in ats):
            return self.none(node, inspect.currentframe().f_lineno)

        cst = [g.get_constant_scalar(a) for a in ats]
        if set(cst) != set(range(len(ats))):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, *next_nodes], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", node_seq: NodeProto, *node_ats: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        assert len(node_seq.input) == len(
            node_ats
        ), f"Matching failed because len({node_seq.input}) != {len(node_ats)}"

        new_nodes = []
        for n in node_ats:
            i = g.get_constant_scalar(n.input[1])
            new_nodes.append(
                g.make_node(
                    "Identity",
                    [node_seq.input[i]],
                    n.output,
                    name=f"{self.__class__.__name__}--{node_seq.name}",
                )
            )
        return new_nodes
