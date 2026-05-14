import inspect
from typing import List, Optional
from onnx import NodeProto
from onnx.numpy_helper import to_array
from ..patterns_api import MatchResult, PatternOptimization


class ConstantOfShapeScatterNDPattern(PatternOptimization):
    """
    Replaces ConstantOfShape + ScatterND with ScatterNDOfShape.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_indices(["indices INT64(DIM1, DIM2, 1)"])
            I_shape(["shape INT64(DIM)"])
            I_masked_updates(["masked_updates FLOAT(DIM1^DIM3, DIM2^DIM4, DIM5)"])

            ConstantOfShape_0[["ConstantOfShape(.)"]]
            ScatterND_1[["ScatterND(., ., .)"]]

            I_shape -->|"INT64(DIM)"| ConstantOfShape_0
            ConstantOfShape_0 --> ScatterND_1
            I_indices -->|"INT64(DIM1, DIM2, 1)"| ScatterND_1
            I_masked_updates -->|"FLOAT(DIM1^DIM3, DIM2^DIM4, DIM5)"| ScatterND_1

            O_y(["y FLOAT(DIM6, DIM7)"])
            ScatterND_1 --> O_y

            class I_indices,I_shape,I_masked_updates,O_y ioNode
            class ConstantOfShape_0,ScatterND_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_indices(["indices INT64(DIM1, DIM2, 1)"])
            I_shape(["shape INT64(DIM)"])
            I_masked_updates(["masked_updates FLOAT(DIM1^DIM3, DIM2^DIM4, DIM5)"])

            ScatterNDOfShape_0[["yaourt.ortops.fused_kernel.cuda.ScatterNDOfShape(., ., .)"]]

            I_shape -->|"INT64(DIM)"| ScatterNDOfShape_0
            I_indices -->|"INT64(DIM1, DIM2, 1)"| ScatterNDOfShape_0
            I_masked_updates -->|"FLOAT(DIM1^DIM3, DIM2^DIM4, DIM5)"| ScatterNDOfShape_0

            O_y(["y FLOAT(DIM6, DIM7)"])
            ScatterNDOfShape_0 --> O_y

            class I_indices,I_shape,I_masked_updates,O_y ioNode
            class ScatterNDOfShape_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type != "ScatterND" or node.domain != "":
            return self.none()

        reduction = g.get_attribute(node, "reduction", exc=False)
        if reduction is None or reduction.s == b"none":
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.has_type(node.input[2]):
            itype = g.try_infer_type(node.input[2])
            if itype == 0:
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            itype = g.get_type(node.input[2])

        node_before = g.node_before(node.input[0])
        if node_before is None or node_before.op_type != "ConstantOfShape" or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        att = g.get_attribute(node_before, "value", False)
        if att is not None:
            arr = to_array(att.t)
            if arr[0] != 0:
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node_before, node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", node_before: NodeProto, node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        reduction = g.get_attribute(node, "reduction")
        new_node = g.make_node(
            "ScatterNDOfShape",
            [node_before.input[0], *node.input[1:]],
            node.output,
            strategy="optimize",
            name=f"{self.__class__.__name__}--{node.name}",
            domain="yaourt.ortops.fused_kernel.cuda",
        )
        new_node.attribute.append(reduction)
        return [new_node]


class MaskedShapeScatterNDPattern(PatternOptimization):
    """
    Replaces Equal, Where, ScatterNDOfShape by MaskedScatterNDOfShape.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_updates(["updates FLOAT(DIM3, DIM4, DIM5)"])
            I_indices(["indices INT64(DIM1, DIM2, 1)"])
            I_shape(["shape INT64(DIM)"])

            ScatterNDOfShape_0[["yaourt.ortops.fused_kernel.cuda.ScatterNDOfShape(., ., .)"]]
            Where_1[["Where(., [0.0], .)"]]
            Equal_2[["Equal(., [-1])"]]

            I_shape -->|"INT64(DIM)"| ScatterNDOfShape_0
            I_indices -->|"INT64(DIM1, DIM2, 1)"| ScatterNDOfShape_0
            Where_1 --> ScatterNDOfShape_0
            Equal_2 -->|"BOOL(DIM1, DIM2, 1)"| Where_1
            I_updates -->|"FLOAT(DIM3, DIM4, DIM5)"| Where_1
            I_indices -->|"INT64(DIM1, DIM2, 1)"| Equal_2

            O_y(["y FLOAT(DIM6, DIM7)"])
            ScatterNDOfShape_0 --> O_y

            class I_updates,I_indices,I_shape,O_y ioNode
            class ScatterNDOfShape_0,Where_1,Equal_2 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_updates(["updates FLOAT(DIM3, DIM4, DIM5)"])
            I_indices(["indices INT64(DIM1, DIM2, 1)"])
            I_shape(["shape INT64(DIM)"])

            MaskedScatterNDOfShape_0[["yaourt.ortops.fused_kernel.cuda.MaskedScatterNDOfShape(
            ., ., .)"]]

            I_shape -->|"INT64(DIM)"| MaskedScatterNDOfShape_0
            I_indices -->|"INT64(DIM1, DIM2, 1)"| MaskedScatterNDOfShape_0
            I_updates -->|"FLOAT(DIM3, DIM4, DIM5)"| MaskedScatterNDOfShape_0

            O_y(["y FLOAT(DIM6, DIM7)"])
            MaskedScatterNDOfShape_0 --> O_y

            class I_updates,I_indices,I_shape,O_y ioNode
            class MaskedScatterNDOfShape_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "ScatterNDOfShape":
            return self.none()

        reduction = g.get_attribute(node, "reduction", exc=False)
        if reduction is None or reduction.s != b"add":
            self.none(node, inspect.currentframe().f_lineno)

        if g.is_used_more_than_once(node.input[1]):
            self.none(node, inspect.currentframe().f_lineno)

        indices = node.input[1]

        where_node = g.node_before(node.input[2])
        if where_node.op_type != "Where" or where_node.domain != "":
            self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant_scalar(where_node.input[1]):
            self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_constant_scalar(where_node.input[1])
        if cst != 0:
            self.none(node, inspect.currentframe().f_lineno)

        equal_node = g.node_before(where_node.input[0])
        if equal_node.op_type != "Equal" or equal_node.domain != "":
            self.none(node, inspect.currentframe().f_lineno)

        indices_again = equal_node.input[0]
        if indices_again != indices:
            self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(equal_node.output[0]):
            self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant_scalar(equal_node.input[1]):
            self.none(node, inspect.currentframe().f_lineno)

        rank = g.get_rank(indices)
        if rank != 3:
            self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(indices):
            self.none(node, inspect.currentframe().f_lineno)
        shape = g.get_shape(indices)
        if shape[-1] != -1:
            self.none(node, inspect.currentframe().f_lineno)

        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        shape_shape = g.get_shape(node.input[0])
        if shape_shape != (2,):
            self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, where_node, equal_node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        scatter_node: NodeProto,
        where_node: NodeProto,
        equal_node: NodeProto,
    ) -> List[NodeProto]:
        mask = g.get_constant_scalar(equal_node.input[1])
        assert isinstance(mask, int), f"Unexpected type {type(mask)} for {equal_node.input[1]!r}"
        new_node = g.make_node(
            "MaskedScatterNDOfShape",
            [scatter_node.input[0], scatter_node.input[1], where_node.input[2]],
            scatter_node.output,
            reduction="add",
            maskedValue=int(mask),
            name=f"{self.__class__.__name__}--{scatter_node.name}",
            domain="yaourt.ortops.fused_kernel.cuda",
        )
        return [new_node]
