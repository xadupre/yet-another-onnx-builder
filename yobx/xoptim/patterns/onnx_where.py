import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class NotWherePattern(PatternOptimization):
    """
    Replaces the sequence Where(Not(cond), X, Y) -> Where(cond, Y, X).

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_A(["A INT64(a, b)"])
            I_X(["X BOOL(a, b)"])
            I_B(["B INT64(a, b)"])

            Not_0[["Not(.)"]]
            Where_1[["Where(., ., .)"]]

            I_X -->|"BOOL(a, b)"| Not_0
            Not_0 -->|"BOOL(a, b)"| Where_1
            I_A -->|"INT64(a, b)"| Where_1
            I_B -->|"INT64(a, b)"| Where_1

            O_Y(["Y INT64(a, b)"])
            Where_1 --> O_Y

            class I_A,I_X,I_B,O_Y ioNode
            class Not_0,Where_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_A(["A INT64(a, b)"])
            I_X(["X BOOL(a, b)"])
            I_B(["B INT64(a, b)"])

            Where_0[["Where(., ., .)"]]

            I_X -->|"BOOL(a, b)"| Where_0
            I_B -->|"INT64(a, b)"| Where_0
            I_A -->|"INT64(a, b)"| Where_0

            O_Y(["Y INT64(a, b)"])
            Where_0 --> O_Y

            class I_A,I_X,I_B,O_Y ioNode
            class Where_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Not" or node.domain != "":
            return self.none()
        wheres = g.next_nodes(node.output[0])
        if len(wheres) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        where = wheres[0]
        if where.op_type != "Where" or where.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node, where], self.apply, insert_at=where)

    def apply(
        self, g: "GraphBuilder", not_node: NodeProto, where_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        return [
            g.make_node(
                "Where",
                [not_node.input[0], where_node.input[2], where_node.input[1]],
                [where_node.output[0]],
                name=f"{self.__class__.__name__}--{where_node.name}",
                doc_string=where_node.doc_string,
            )
        ]


class WhereAddPattern(PatternOptimization):
    """
    Replaces the sequence Add(X, Where(bool_mask, 0, -inf)) -> Where(bool_mask, X, -inf).

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_inf(["inf FLOAT(1)"])
            I_X(["X FLOAT(a, b)"])
            I_mask(["mask BOOL(a, b)"])

            Constant_0[["Constant() -#gt; inf"]]
            Where_1[["Where(., [0.0], .)"]]
            Add_2[["Add(., .)"]]

            I_mask -->|"BOOL(a, b)"| Where_1
            Constant_0 -->|"FLOAT(1)"| Where_1
            Where_1 -->|"FLOAT(a, b)"| Add_2
            I_X -->|"FLOAT(a, b)"| Add_2

            O_Y(["Y FLOAT(a, b)"])
            Add_2 --> O_Y

            class I_inf,I_X,I_mask,O_Y ioNode
            class Constant_0 constNode
            class Where_1,Add_2 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_inf(["inf FLOAT(1)"])
            I_X(["X FLOAT(a, b)"])
            I_mask(["mask BOOL(a, b)"])

            Where_0[["Where(., ., .)"]]

            I_mask -->|"BOOL(a, b)"| Where_0
            I_X -->|"FLOAT(a, b)"| Where_0
            I_inf -->|"FLOAT(1)"| Where_0

            O_Y(["Y FLOAT(a, b)"])
            Where_0 --> O_Y

            class I_inf,I_X,I_mask,O_Y ioNode
            class Where_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Where" or node.domain != "":
            return self.none()
        if not g.is_constant_scalar(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(node.input[2]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst1 = g.get_constant_scalar(node.input[1])
        cst2 = g.get_constant_scalar(node.input[2])
        if cst1 is None or cst2 is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if cst1 != 0:
            return self.none(node, inspect.currentframe().f_lineno)
        if not np.isinf(cst2):
            return self.none(node, inspect.currentframe().f_lineno)

        add_nodes = g.next_nodes(node.output[0])
        if len(add_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if add_nodes[0].op_type != "Add" or add_nodes[0].domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node, add_nodes[0]], self.apply, insert_at=add_nodes[0])

    def apply(
        self, g: "GraphBuilder", where_node: NodeProto, add_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        where_input1 = add_node.input[1 if add_node.input[0] == where_node.output[0] else 0]
        return [
            g.make_node(
                "Where",
                [where_node.input[0], where_input1, where_node.input[2]],
                [add_node.output[0]],
                name=f"{self.__class__.__name__}--{where_node.name}",
                doc_string=where_node.doc_string,
            )
        ]
