import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class ReduceSumNormalizePattern(PatternOptimization):
    """
    Nodes equivalent to a reduction.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_Y(["Y FLOAT(a, b)"])
            I_X(["X FLOAT16(a, b)"])
            I_axis(["axis INT64()"])

            Constant_0[["Constant() -#gt; axis"]]
            Cast_1[["Cast(., to=FLOAT)"]]
            ReduceSum_2[["ReduceSum(., .)"]]
            Mul_3[["Mul(., .)"]]
            Sub_4[["Sub(., .)"]]
            Cast_5[["Cast(., to=FLOAT16)"]]

            I_X -->|"FLOAT16(a, b)"| Cast_1
            Cast_1 -->|"FLOAT(a, b)"| ReduceSum_2
            Constant_0 -->|"INT64()"| ReduceSum_2
            ReduceSum_2 -->|"FLOAT(a, 1)"| Mul_3
            I_Y -->|"FLOAT(a, b)"| Mul_3
            Cast_1 -->|"FLOAT(a, b)"| Sub_4
            Mul_3 -->|"FLOAT(a, b)"| Sub_4
            Sub_4 -->|"FLOAT(a, b)"| Cast_5

            O_Z(["Z FLOAT16(a, b)"])
            Cast_5 --> O_Z

            class I_Y,I_X,I_axis,O_Z ioNode
            class Constant_0 constNode
            class Cast_1,ReduceSum_2,Mul_3,Sub_4,Cast_5 opNode
    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_Y(["Y FLOAT(a, b)"])
            I_X(["X FLOAT16(a, b)"])
            I_axis(["axis INT64()"])

            ReduceSum_0[["ReduceSum(., .)"]]
            Cast_1[["Cast(., to=FLOAT16)"]]
            Mul_2[["Mul(., .)"]]
            Sub_3[["Sub(., .)"]]

            I_X -->|"FLOAT16(a, b)"| ReduceSum_0
            I_axis -->|"INT64()"| ReduceSum_0
            I_Y -->|"FLOAT(a, b)"| Cast_1
            ReduceSum_0 --> Mul_2
            Cast_1 -->|"FLOAT16(a, b)"| Mul_2
            I_X -->|"FLOAT16(a, b)"| Sub_3
            Mul_2 --> Sub_3

            O_Z(["Z FLOAT16(a, b)"])
            Sub_3 --> O_Z

            class I_Y,I_X,I_axis,O_Z ioNode
            class ReduceSum_0,Cast_1,Mul_2,Sub_3 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "ReduceSum" or node.domain != "":
            return self.none()

        cast_node = g.node_before(node.input[0])
        if cast_node is None or cast_node.op_type != "Cast":
            return self.none(node, inspect.currentframe().f_lineno)

        mul_node = g.next_nodes(node.output[0])
        if len(mul_node) != 1 or mul_node[0].op_type != "Mul":
            return self.none(node, inspect.currentframe().f_lineno)

        sub_node = g.next_nodes(mul_node[0].output[0])
        if len(sub_node) != 1 or sub_node[0].op_type != "Sub":
            return self.none(node, inspect.currentframe().f_lineno)

        cast2_node = g.next_nodes(sub_node[0].output[0])
        if len(cast2_node) != 1 or cast2_node[0].op_type != "Cast":
            return self.none(node, inspect.currentframe().f_lineno)

        if not (set(sub_node[0].input) & set(node.input)):
            return self.none(node, inspect.currentframe().f_lineno)

        if g.get_type(cast_node.input[0]) != g.get_type(cast2_node[0].output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self, [cast_node, node, mul_node[0], sub_node[0], cast2_node[0]], self.apply
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        cast_node: NodeProto,
        node: NodeProto,
        mul_node: NodeProto,
        sub_node: NodeProto,
        cast2_node: NodeProto,
    ) -> List[NodeProto]:
        new_name = g.unique_name(f"{self.__class__.__name__}_{node.output[0]}")
        new_red = g.make_node(
            node.op_type,
            [cast_node.input[0], node.input[1]],
            [new_name],
            name=f"{self.__class__.__name__}--{node.name}",
        )
        new_red.attribute.extend(node.attribute)
        other_name = [n for n in mul_node.input if n != node.output[0]]
        assert len(other_name) == 1, f"Unexpected name {other_name!r}"
        new_name2 = g.unique_name(f"{self.__class__.__name__}_{other_name[0]}")
        new_cast = g.make_node(
            "Cast",
            other_name,
            [new_name2],
            to=g.get_attribute(cast2_node, "to").i,
            name=f"{self.__class__.__name__}--{cast_node.name}",
        )

        new_m = g.unique_name(f"{self.__class__.__name__}_{mul_node.output[0]}")
        new_mul = g.make_node(
            mul_node.op_type,
            [new_name, new_name2],
            [new_m],
            name=f"{self.__class__.__name__}--{mul_node.name}",
        )

        if mul_node.output[0] == sub_node.input[0]:
            inputs = [new_m, new_red.input[0]]
        else:
            inputs = [new_red.input[0], new_m]
        new_sub = g.make_node(
            sub_node.op_type,
            inputs,
            cast2_node.output,
            name=f"{self.__class__.__name__}--{sub_node.name}",
        )

        return [new_red, new_cast, new_mul, new_sub]


class ReduceArgTopKPattern(PatternOptimization):
    """
    Fuses ReduceMin(X, axis), ArgMin(X, axis) into TopK(, k=1).

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b)"])
            I_one(["one INT64(1)"])

            Constant_0[["Constant() -#gt; one"]]
            ReduceMin_1[["ReduceMin(., .)"]]
            ArgMin_2[["ArgMin(., axis=1)"]]

            I_X -->|"FLOAT(a, b)"| ReduceMin_1
            Constant_0 -->|"INT64(1)"| ReduceMin_1
            I_X -->|"FLOAT(a, b)"| ArgMin_2

            O_Y2(["Y2 INT64(a)"])
            ArgMin_2 --> O_Y2
            O_Y1(["Y1 FLOAT(a)"])
            ReduceMin_1 --> O_Y1

            class I_X,I_one,O_Y2,O_Y1 ioNode
            class Constant_0 constNode
            class ReduceMin_1,ArgMin_2 opNode
    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b)"])
            I_one(["one INT64(1)"])

            TopK_0[["TopK(., ., axis=1)"]]
            Squeeze_1[["Squeeze(., .)"]]
            Squeeze_2[["Squeeze(., .)"]]

            I_X -->|"FLOAT(a, b)"| TopK_0
            I_one -->|"INT64(1)"| TopK_0
            TopK_0 --> Squeeze_1
            I_one -->|"INT64(1)"| Squeeze_1
            TopK_0 --> Squeeze_2
            I_one -->|"INT64(1)"| Squeeze_2

            O_Y2(["Y2 INT64(a)"])
            Squeeze_2 --> O_Y2
            O_Y1(["Y1 FLOAT(a)"])
            Squeeze_1 --> O_Y1

            class I_X,I_one,O_Y2,O_Y1 ioNode
            class TopK_0,Squeeze_1,Squeeze_2 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if g.main_opset < 18:
            return self.none()
        if node.op_type not in ("ArgMin", "ArgMax") or node.domain != "":
            return self.none()

        next_nodes = g.next_nodes(node.input[0])
        if len(next_nodes) < 2:
            return self.none(node, inspect.currentframe().f_lineno)
        look_for = f"Reduce{node.op_type[3:]}"
        reduce = [n for n in next_nodes if n.op_type == look_for]
        if len(reduce) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        reduce_node = reduce[0]

        if not g.is_constant(reduce_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(reduce_node.input[1])
        if not cst:
            return self.none(node, inspect.currentframe().f_lineno)
        axes = tuple(cst)
        if len(axes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        axis = g.get_attribute_with_default(node, "axis", 0)
        if axis != cst[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        if g.get_attribute_with_default(node, "keepdims", 1) != g.get_attribute_with_default(
            reduce_node, "keepdims", 1
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_attribute_with_default(reduce_node, "noop_with_empty_axes", 0):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_attribute_with_default(reduce_node, "select_last_index", 0) == 1:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [reduce_node, node], self.apply)

    def apply(
        self, g: "GraphBuilder", reduce_node: NodeProto, arg_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        one = g.make_initializer(
            "", np.array([1], dtype=np.int64), source=f"{self.__class__.__name__}.K"
        )
        keepdims = g.get_attribute_with_default(arg_node, "keepdims", 1)
        axis = g.get_attribute_with_default(arg_node, "axis", 0)

        topk_names = (
            [reduce_node.output[0], arg_node.output[0]]
            if keepdims
            else [
                g.unique_name(f"{self.__class__.__name__}_{reduce_node.output[0]}"),
                g.unique_name(f"{self.__class__.__name__}_{arg_node.output[0]}"),
            ]
        )
        nodes = [
            g.make_node(
                "TopK",
                [reduce_node.input[0], one],
                topk_names,
                axis=axis,
                largest=1 if "Max" in arg_node.op_type else 0,
                name=f"{self.__class__.__name__}--{arg_node.name}",
            )
        ]
        if not keepdims:
            axis = g.make_initializer(
                "", np.array([axis], dtype=np.int64), source=f"{self.__class__.__name__}.K"
            )
            nodes.extend(
                [
                    g.make_node(
                        "Squeeze",
                        [topk_names[0], axis],
                        [reduce_node.output[0]],
                        name=f"{self.__class__.__name__}--{reduce_node.name}",
                    ),
                    g.make_node(
                        "Squeeze",
                        [topk_names[1], axis],
                        [arg_node.output[0]],
                        name=f"{self.__class__.__name__}--{arg_node.name}",
                    ),
                ]
            )
        return nodes
