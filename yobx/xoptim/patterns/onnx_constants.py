import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class ConstantToInitializerPattern(PatternOptimization):
    """
    Replaces a node Constant by an initializer and a node Identity.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333


            Constant_0[["Constant() -#gt; cst"]]


            O_cst(["cst FLOAT(2)"])
            Constant_0 --> O_cst

            class O_cst ioNode
            class Constant_0 constNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333


            Identity_0[["Identity([1.0, 2.0])"]]


            O_cst(["cst FLOAT(2)"])
            Identity_0 --> O_cst

            class O_cst ioNode
            class Identity_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Constant" or node.domain != "":
            return self.none()
        if g.do_not_turn_constant_initializers_maybe_because_of_showing(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(self, g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
        cst = g.get_computed_constant(node.output[0])
        assert (
            cst is not None
        ), f"Node {g.pretty_node(node)} is a constant, it must be possible to evaluate it."
        # if not g.has_exact_same_constant_in_context(node.output[0]):
        init = g.make_initializer(f"{node.output[0]}_cst2init", cst)
        return [
            g.make_node(
                "Identity", [init], node.output, name=f"{self.__class__.__name__}--{node.name}"
            )
        ]
