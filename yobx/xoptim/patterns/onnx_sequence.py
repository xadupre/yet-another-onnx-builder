import inspect
from typing import List, Optional
import numpy as np
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


class SplitToSequenceSequenceAtPattern(PatternOptimization):
    """
    Replaces ``SplitToSequence(x, split, axis=a)`` followed
    by ``SequenceAt(seq, 0)``, ``SequenceAt(seq, 1)``, ...
    into ``Split(x, split, axis=a)``.

    When ``keepdims=0``, a ``Squeeze`` node is added after each split output
    to remove the split axis dimension (which has size 1 for each chunk).

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, 6)"])
            I_split(["split INT64(3)"])

            SplitToSequence_0[["SplitToSequence(., ., axis=1)"]]
            SequenceAt_1[["SequenceAt(., 0)"]]
            SequenceAt_2[["SequenceAt(., 1)"]]
            SequenceAt_3[["SequenceAt(., 2)"]]

            I_X -->|"FLOAT(a, 6)"| SplitToSequence_0
            I_split -->|"INT64(3)"| SplitToSequence_0
            SplitToSequence_0 --> SequenceAt_1
            SplitToSequence_0 --> SequenceAt_2
            SplitToSequence_0 --> SequenceAt_3

            O_Y1(["Y1 FLOAT(a, 2)"])
            SequenceAt_1 --> O_Y1
            O_Y2(["Y2 FLOAT(a, 2)"])
            SequenceAt_2 --> O_Y2
            O_Y3(["Y3 FLOAT(a, 2)"])
            SequenceAt_3 --> O_Y3

            class I_X,I_split,O_Y1,O_Y2,O_Y3 ioNode
            class SplitToSequence_0,SequenceAt_1,SequenceAt_2,SequenceAt_3 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, 6)"])
            I_split(["split INT64(3)"])

            Split_0[["Split(., ., axis=1)"]]

            I_X -->|"FLOAT(a, 6)"| Split_0
            I_split -->|"INT64(3)"| Split_0

            O_Y1(["Y1 FLOAT(a, 2)"])
            Split_0 --> O_Y1
            O_Y2(["Y2 FLOAT(a, 2)"])
            Split_0 --> O_Y2
            O_Y3(["Y3 FLOAT(a, 2)"])
            Split_0 --> O_Y3

            class I_X,I_split,O_Y1,O_Y2,O_Y3 ioNode
            class Split_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "SplitToSequence" or node.domain != "":
            return self.none()

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) == 0:
            return self.none(node, inspect.currentframe().f_lineno)
        if any(n.op_type != "SequenceAt" for n in next_nodes):
            return self.none(node, inspect.currentframe().f_lineno)

        ats = [n.input[1] for n in next_nodes]
        if any(not g.is_constant_scalar(a) for a in ats):
            return self.none(node, inspect.currentframe().f_lineno)

        cst = [g.get_constant_scalar(a) for a in ats]
        if set(cst) != set(range(len(cst))):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, *next_nodes], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_seq: NodeProto,
        *node_ats: NodeProto,
    ) -> List[NodeProto]:
        keepdims = g.get_attribute_with_default(node_seq, "keepdims", default_value=1)
        axis = g.get_attribute_with_default(node_seq, "axis", default_value=0)

        index_to_output = {}
        for n in node_ats:
            i = int(g.get_constant_scalar(n.input[1]))
            index_to_output[i] = n.output[0]

        n = len(node_ats)

        if keepdims:
            outputs = [index_to_output[i] for i in range(n)]
        else:
            outputs = [
                g.unique_name(f"{self.__class__.__name__}--{index_to_output[i]}")
                for i in range(n)
            ]

        inputs = [node_seq.input[0]]
        if len(node_seq.input) > 1:
            inputs.append(node_seq.input[1])

        split_node = g.make_node(
            "Split",
            inputs,
            outputs,
            axis=axis,
            num_outputs=n,
            name=f"{self.__class__.__name__}--{node_seq.name}",
        )

        if not keepdims:
            axis_init = g.make_initializer(
                "",
                np.array([axis], dtype=np.int64),
                source=f"{self.__class__.__name__}.axes",
            )
            post_nodes = [
                g.make_node(
                    "Squeeze",
                    [outputs[i], axis_init],
                    [index_to_output[i]],
                    name=f"{self.__class__.__name__}--{node_ats[i].name}",
                )
                for i in range(n)
            ]
            return [split_node, *post_nodes]

        return [split_node]
