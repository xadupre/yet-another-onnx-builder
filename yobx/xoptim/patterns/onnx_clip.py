import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class ClipClipPattern(PatternOptimization):
    """
    Merges consecutive clips if one is defining min and the other max.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_zero(["zero FLOAT(1)"])
            I_X(["X FLOAT(a, b)"])
            I_one(["one FLOAT(1)"])

            Constant_0[["Constant() -#gt; zero"]]
            Constant_1[["Constant() -#gt; one"]]
            Clip_2[["Clip(., .)"]]
            Clip_3[["Clip(., , .)"]]

            I_X -->|"FLOAT(a, b)"| Clip_2
            Constant_0 -->|"FLOAT(1)"| Clip_2
            Clip_2 -->|"FLOAT(a, b)"| Clip_3
            Constant_1 -->|"FLOAT(1)"| Clip_3

            O_Y(["Y FLOAT(c, d)"])
            Clip_3 --> O_Y

            class I_zero,I_X,I_one,O_Y ioNode
            class Constant_0,Constant_1 constNode
            class Clip_2,Clip_3 opNode
    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_zero(["zero FLOAT(1)"])
            I_X(["X FLOAT(a, b)"])
            I_one(["one FLOAT(1)"])

            Clip_0[["Clip(., ., .)"]]

            I_X -->|"FLOAT(a, b)"| Clip_0
            I_zero -->|"FLOAT(1)"| Clip_0
            I_one -->|"FLOAT(1)"| Clip_0

            O_Y(["Y FLOAT(c, d)"])
            Clip_0 --> O_Y

            class I_zero,I_X,I_one,O_Y ioNode
            class Clip_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Clip" or node.domain != "":
            return self.none()
        before = g.node_before(node.input[0])
        if (
            before is None
            or g.is_used_more_than_once(node.input[0])
            or before.op_type != "Clip"
            or before.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        min1 = before.input[1] if len(before.input) > 1 else ""
        min2 = node.input[1] if len(node.input) > 1 else ""
        if (min1 and min2) or (not min1 and not min2):
            return self.none(node, inspect.currentframe().f_lineno)
        max1 = before.input[2] if len(before.input) > 2 else ""
        max2 = node.input[2] if len(node.input) > 2 else ""
        if (max1 and max2) or (not max1 and not max2):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [before, node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", before: NodeProto, node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        # merges clips
        min1 = before.input[1] if len(before.input) > 1 else ""
        min2 = node.input[1] if len(node.input) > 1 else ""
        max1 = before.input[2] if len(before.input) > 2 else ""
        max2 = node.input[2] if len(node.input) > 2 else ""

        return [
            g.make_node(
                "Clip",
                [before.input[0], min1 or min2, max1 or max2],
                node.output,
                name=f"{self.__class__.__name__}--{node.name}",
            )
        ]
