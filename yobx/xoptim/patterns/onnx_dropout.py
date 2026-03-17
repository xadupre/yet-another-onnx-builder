import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class DropoutPattern(PatternOptimization):
    """
    Checks that a Cast is really needed.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I__onx_add02(["_onx_add02 FLOAT16(4, 512, 128)"])

            Dropout_0[["Dropout(., 0.0, False)"]]

            I__onx_add02 -->|"FLOAT16(4, 512, 128)"| Dropout_0

            O_dropout(["dropout FLOAT16(4, 512, 128)"])
            Dropout_0 --> O_dropout

            class I__onx_add02,O_dropout ioNode
            class Dropout_0 opNode
    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I__onx_add02(["_onx_add02 FLOAT16(4, 512, 128)"])

            Identity_0[["Identity(.)"]]

            I__onx_add02 -->|"FLOAT16(4, 512, 128)"| Identity_0

            O_dropout(["dropout FLOAT16(4, 512, 128)"])
            Identity_0 --> O_dropout

            class I__onx_add02,O_dropout ioNode
            class Identity_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Dropout" or node.domain != "":
            return None

        for o in node.output[1:]:
            if o and g.is_used(o):
                return self.none(node, inspect.currentframe().f_lineno)

        if not (
            len(node.input) >= 3
            and node.input[2] != ""
            and g.is_constant_scalar(node.input[2])
            and not g.get_constant_scalar(node.input[2])
        ):
            return MatchResult(self, [node], self.apply, insert_at=node)

        if (
            len(node.input) >= 3
            and node.input[2] != ""
            and g.is_constant_scalar(node.input[2])
            and g.get_constant_scalar(node.input[2]) != 0
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(self, g: "GraphBuilder", dropout_node: NodeProto) -> List[NodeProto]:  # noqa: F821
        return [
            g.make_node(
                "Identity",
                dropout_node.input[:1],
                dropout_node.output[:1],
                name=f"{self.__class__.__name__}--{dropout_node.name}",
                doc_string=dropout_node.doc_string,
            )
        ]
