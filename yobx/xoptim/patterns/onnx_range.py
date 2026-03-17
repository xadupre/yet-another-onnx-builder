import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class SwapRangeAddScalarPattern(PatternOptimization):
    """
    Swap Range + Add when a scalar is added.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_END(["END INT64()"])
            I_PLUS(["PLUS INT64(1)"])
            I_one(["one INT64()"])
            I_START(["START INT64()"])

            Constant_0[["Constant() -#gt; one"]]
            Range_1[["Range(., ., .)"]]
            Add_2[["Add(., .)"]]

            I_START -->|"INT64()"| Range_1
            I_END -->|"INT64()"| Range_1
            Constant_0 -->|"INT64()"| Range_1
            Range_1 -->|"INT64(NEWDIM_range_0)"| Add_2
            I_PLUS -->|"INT64(1)"| Add_2

            O_Y(["Y INT64(NEWDIM_range)"])
            Add_2 --> O_Y

            class I_END,I_PLUS,I_one,I_START,O_Y ioNode
            class Constant_0 constNode
            class Range_1,Add_2 opNode
    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_END(["END INT64()"])
            I_PLUS(["PLUS INT64(1)"])
            I_one(["one INT64()"])
            I_START(["START INT64()"])

            Squeeze_0[["Squeeze(.)"]]
            Add_1[["Add(., .)"]]
            Add_2[["Add(., .)"]]
            Range_3[["Range(., ., .)"]]

            I_PLUS -->|"INT64(1)"| Squeeze_0
            I_END -->|"INT64()"| Add_1
            Squeeze_0 -->|"INT64()"| Add_1
            I_START -->|"INT64()"| Add_2
            Squeeze_0 -->|"INT64()"| Add_2
            Add_2 -->|"INT64()"| Range_3
            Add_1 -->|"INT64()"| Range_3
            I_one -->|"INT64()"| Range_3

            O_Y(["Y INT64(NEWDIM_range)"])
            Range_3 --> O_Y

            class I_END,I_PLUS,I_one,I_START,O_Y ioNode
            class Squeeze_0,Add_1,Add_2,Range_3 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Range" or node.domain != "":
            return self.none()

        node_add = g.next_nodes(node.output[0])
        if len(node_add) != 1 or node_add[0].op_type != "Add" or node_add[0].domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        cst = node_add[0].input[1]
        if not g.has_shape(cst) or g.get_shape(cst) != (1,):
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node, node_add[0]], self.apply, insert_at=node_add[0])

    def apply(
        self, g: "GraphBuilder", node_range: NodeProto, node_add: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        start, end = node_range.input[:2]

        squeezed = g.unique_name(f"{self.__class__.__name__}--{node_add.input[1]}")
        new_end = g.unique_name(f"{self.__class__.__name__}--{node_range.input[1]}")
        new_add = [
            g.make_node(
                "Squeeze",
                [node_add.input[1]],
                [squeezed],
                name=f"{self.__class__.__name__}--{node_add.name}",
                doc_string=node_add.doc_string,
            ),
            g.make_node(
                "Add",
                [end, squeezed],
                [new_end],
                name=f"{self.__class__.__name__}--{node_range.name}",
                doc_string=node_range.doc_string,
            ),
        ]

        new_range = None
        if g.is_constant(start):
            cst_start = g.get_constant_scalar(start)
            if cst_start == 0:
                new_range = g.make_node(
                    "Range",
                    [squeezed, new_end, *node_range.input[2:]],
                    [node_add.output[0]],
                    name=f"{self.__class__.__name__}--{node_range.name}",
                    doc_string=node_range.doc_string,
                )
        if new_range is None:
            new_start = g.unique_name(f"{self.__class__.__name__}--{node_range.input[0]}")
            new_add.append(
                g.make_node(
                    "Add",
                    [start, squeezed],
                    [new_start],
                    name=f"{self.__class__.__name__}--{node_range.name}",
                    doc_string=node_range.doc_string,
                )
            )
            new_range = g.make_node(
                "Range",
                [new_start, new_end, *node_range.input[2:]],
                [node_add.output[0]],
                name=f"{self.__class__.__name__}--{node_range.name}",
                doc_string=node_range.doc_string,
            )
        return [*new_add, new_range]
