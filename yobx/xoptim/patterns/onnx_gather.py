import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class GatherGatherPattern(PatternOptimization):
    """
    Simplifies two consecutive Gather operations into one.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(n, a, b)"])
            I_cst1(["cst1 INT64(k)"])
            I_cst2(["cst2 INT64()"])

            Constant_0[["Constant() -#gt; cst1"]]
            Constant_1[["Constant() -#gt; cst2"]]
            Gather_2[["Gather(., ., axis=0)"]]
            Gather_3[["Gather(., ., axis=0)"]]

            I_X -->|"FLOAT(n, a, b)"| Gather_2
            Constant_0 -->|"INT64(k)"| Gather_2
            Gather_2 -->|"FLOAT(k, a, b)"| Gather_3
            Constant_1 -->|"INT64()"| Gather_3

            O_Y(["Y FLOAT(a, b)"])
            Gather_3 --> O_Y

            class I_X,I_cst1,I_cst2,O_Y ioNode
            class Constant_0,Constant_1 constNode
            class Gather_2,Gather_3 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(n, a, b)"])
            I_cst3(["cst3 INT64()"])

            Constant_0[["Constant() -#gt; cst3"]]
            Gather_1[["Gather(., ., axis=0)"]]

            I_X -->|"FLOAT(n, a, b)"| Gather_1
            Constant_0 -->|"INT64()"| Gather_1

            O_Y(["Y FLOAT(a, b)"])
            Gather_1 --> O_Y

            class I_X,I_cst3,O_Y ioNode
            class Constant_0 constNode
            class Gather_1 opNode

    The composed index is ``cst3 = cst1[cst2]``. This applies when both
    Gather operations use ``axis=0``, the inner Gather indices ``cst1`` are a
    1-D constant array, and the outer Gather indices ``cst2`` are a constant of
    any shape (scalar or 1-D).
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Gather" or node.domain != "":
            return self.none()

        # Outer Gather must use axis=0.
        axis2 = g.get_attribute_with_default(node, "axis", default_value=0)
        if axis2 != 0:
            return self.none(node, inspect.currentframe().f_lineno)

        # Outer indices must be a constant of integer type.
        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst2 = g.get_computed_constant(node.input[1])
        if cst2 is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if cst2.dtype not in (np.int64, np.int32):
            return self.none(node, inspect.currentframe().f_lineno)

        # The data input must come from another Gather node.
        inner_node = g.node_before(node.input[0])
        if inner_node is None or inner_node.op_type != "Gather" or inner_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # Inner Gather must also use axis=0.
        axis1 = g.get_attribute_with_default(inner_node, "axis", default_value=0)
        if axis1 != 0:
            return self.none(node, inspect.currentframe().f_lineno)

        # Inner indices must be a constant 1-D int64 array.
        if not g.is_constant(inner_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst1 = g.get_computed_constant(inner_node.input[1])
        if cst1 is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if cst1.dtype not in (np.int64, np.int32):
            return self.none(node, inspect.currentframe().f_lineno)
        if cst1.ndim != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [inner_node, node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", inner_node: NodeProto, outer_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        cst1 = g.get_computed_constant(inner_node.input[1])
        cst2 = g.get_computed_constant(outer_node.input[1])

        # Compose indices: cst3 = cst1[cst2].
        cst3 = np.asarray(cst1[cst2], dtype=np.int64)

        new_indices = g.make_initializer("", cst3, source=f"{self.__class__.__name__}.indices")
        new_node = g.make_node(
            "Gather",
            [inner_node.input[0], new_indices],
            outer_node.output,
            axis=0,
            name=f"{self.__class__.__name__}--{outer_node.name}",
            doc_string=outer_node.doc_string,
        )
        if g.is_used_more_than_once(inner_node.output[0]):
            return [inner_node, new_node]
        return [new_node]
