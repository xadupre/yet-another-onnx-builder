import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class SliceSlicePattern(PatternOptimization):
    """
    Merges consecutive slices if axis are disjoints.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b)"])
            I_one(["one INT64(1)"])
            I_zero(["zero INT64(1)"])

            Constant_0[["Constant() -#gt; zero"]]
            Constant_1[["Constant() -#gt; one"]]
            Slice_2[["Slice(., ., ., .)"]]
            Slice_3[["Slice(., ., ., .)"]]

            I_X -->|"FLOAT(a, b)"| Slice_2
            Constant_0 -->|"INT64(1)"| Slice_2
            Constant_1 -->|"INT64(1)"| Slice_2
            Slice_2 -->|"FLOAT(1, b)"| Slice_3
            Constant_0 -->|"INT64(1)"| Slice_3
            Constant_1 -->|"INT64(1)"| Slice_3

            O_Y(["Y FLOAT(c, d)"])
            Slice_3 --> O_Y

            class I_X,I_one,I_zero,O_Y ioNode
            class Constant_0,Constant_1 constNode
            class Slice_2,Slice_3 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b)"])
            I_one(["one INT64(1)"])
            I_zero(["zero INT64(1)"])

            Concat_0[["Concat(., ., axis=0)"]]
            Concat_1[["Concat(., ., axis=0)"]]
            Concat_2[["Concat(., ., axis=0)"]]
            Slice_3[["Slice(., ., ., .)"]]

            I_zero -->|"INT64(1)"| Concat_0
            I_one -->|"INT64(1)"| Concat_1
            I_zero -->|"INT64(1)"| Concat_2
            I_one -->|"INT64(1)"| Concat_2
            I_X -->|"FLOAT(a, b)"| Slice_3
            Concat_0 -->|"INT64(2)"| Slice_3
            Concat_1 -->|"INT64(2)"| Slice_3
            Concat_2 -->|"INT64(2)"| Slice_3

            O_Y(["Y FLOAT(c, d)"])
            Slice_3 --> O_Y

            class I_X,I_one,I_zero,O_Y ioNode
            class Concat_0,Concat_1,Concat_2,Slice_3 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Slice" or node.domain != "":
            return self.none()
        before = g.node_before(node.input[0])
        if (
            before is None
            or g.is_used_more_than_once(node.input[0])
            or before.op_type != "Slice"
            or before.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        axis2 = None if len(node.input) < 3 else node.input[3]
        axis1 = None if len(before.input) < 3 else before.input[3]
        if axis1 is None or axis2 is None:
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant(axis1) or not g.is_constant(axis2):
            return self.none(node, inspect.currentframe().f_lineno)

        cst1 = g.get_computed_constant(axis1)
        cst2 = g.get_computed_constant(axis2)
        if cst1 is None or cst2 is None:
            return self.none(node, inspect.currentframe().f_lineno)

        set1 = set(map(int, cst1))
        set2 = set(map(int, cst2))
        if set1 & set2:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [before, node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", before: NodeProto, node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        # merges slices

        new_start = g.unique_name(f"{self.__class__.__name__}_{node.input[1]}_start")
        new_end = g.unique_name(f"{self.__class__.__name__}_{node.input[2]}_end")
        new_axis = g.unique_name(f"{self.__class__.__name__}_{node.input[3]}_axis")
        conc = [
            g.make_node(
                "Concat",
                [before.input[1], node.input[1]],
                [new_start],
                axis=0,
                name=f"{self.__class__.__name__}--{node.name}-start",
            ),
            g.make_node(
                "Concat",
                [before.input[2], node.input[2]],
                [new_end],
                axis=0,
                name=f"{self.__class__.__name__}--{node.name}-end",
            ),
            g.make_node(
                "Concat",
                [before.input[3], node.input[3]],
                [new_axis],
                axis=0,
                name=f"{self.__class__.__name__}--{node.name}-axis",
            ),
        ]
        inputs = [before.input[0], new_start, new_end, new_axis]
        if len(node.input) > 4 and len(before.input) > 4:
            new_step = g.unique_name(f"{self.__class__.__name__}_{node.input[0]}_step")
            conc.append(
                g.make_node(
                    "Concat",
                    [before.input[4], node.input[4]],
                    [new_step],
                    axis=0,
                    name=f"{self.__class__.__name__}--{node.name}-step",
                )
            )
            inputs.append(new_step)
        elif len(node.input) > 4:
            cst1 = g.get_computed_constant(before.input[3])
            one = g.make_initializer(
                "",
                np.array([1] * cst1.shape[0], dtype=np.int64),
                source="SliceSlicePattern.apply.step.1",
            )
            new_step = g.unique_name(f"{self.__class__.__name__}_{node.input[0]}_step")
            conc.append(
                g.make_node(
                    "Concat",
                    [one, node.input[4]],
                    [new_step],
                    axis=0,
                    name=f"{self.__class__.__name__}--{node.name}-step",
                )
            )
            inputs.append(new_step)
        elif len(before.input) > 4:
            cst2 = g.get_computed_constant(node.input[3])
            one = g.make_initializer(
                "",
                np.array([1] * cst2.shape[0], dtype=np.int64),
                source="SliceSlicePattern.apply.step.2",
            )
            new_step = g.unique_name(f"{self.__class__.__name__}_{node.input[0]}_step")
            conc.append(
                g.make_node(
                    "Concat",
                    [before.input[4], one],
                    [new_step],
                    axis=0,
                    name=f"{self.__class__.__name__}--{node.name}-step",
                )
            )
            inputs.append(new_step)

        node = g.make_node(
            "Slice", inputs, node.output, name=f"{self.__class__.__name__}--{node.name}"
        )
        return [*conc, node]
