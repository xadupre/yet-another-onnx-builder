import inspect
from typing import List, Optional
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization, EasyPatternOptimization
from ..patterns.onnx_functions import GeluPattern


class BiasGeluPattern(PatternOptimization):
    """
    Replaces by ``y = BiasGelu(x, B)``::

        t = x + B
        y = t ( Erf(1 / t) + 1)

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_B(["B FLOAT(8)"])
            I_X(["X FLOAT(2, 2, 4, 8)"])

            Constant_0[["Constant() -#gt; B"]]
            Add_1[["Add(., .)"]]
            Div_2[["Div(., [1.4140625])"]]
            Erf_3[["Erf(.)"]]
            Add_4[["Add(., [1.0])"]]
            Mul_5[["Mul(., .)"]]
            Mul_6[["Mul(., [0.5])"]]

            I_X -->|"FLOAT(2, 2, 4, 8)"| Add_1
            Constant_0 -->|"FLOAT(8)"| Add_1
            Add_1 -->|"FLOAT(2, 2, 4, 8)"| Div_2
            Div_2 -->|"FLOAT(2, 2, 4, 8)"| Erf_3
            Erf_3 -->|"FLOAT(2, 2, 4, 8)"| Add_4
            Add_1 -->|"FLOAT(2, 2, 4, 8)"| Mul_5
            Add_4 -->|"FLOAT(2, 2, 4, 8)"| Mul_5
            Mul_5 -->|"FLOAT(2, 2, 4, 8)"| Mul_6

            O_Y(["Y FLOAT(2, 2, 4, 8)"])
            Mul_6 --> O_Y

            class I_B,I_X,O_Y ioNode
            class Constant_0 constNode
            class Add_1,Div_2,Erf_3,Add_4,Mul_5,Mul_6 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_B(["B FLOAT(8)"])
            I_X(["X FLOAT(2, 2, 4, 8)"])

            BiasGelu_0[["com.microsoft.BiasGelu(., .)"]]

            I_X -->|"FLOAT(2, 2, 4, 8)"| BiasGelu_0
            I_B -->|"FLOAT(8)"| BiasGelu_0

            O_Y(["Y FLOAT(2, 2, 4, 8)"])
            BiasGelu_0 --> O_Y

            class I_B,I_X,O_Y ioNode
            class BiasGelu_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Erf" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        div = g.node_before(node.input[0])
        if (
            not g.is_constant_scalar(div.input[1])
            or g.get_constant_scalar(div.input[1]) != 1.4140625
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        add = g.node_before(div.input[0])
        if add.op_type != "Add" or add.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(add.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        add1_nexts = g.next_nodes(add.output[0])
        if len(add1_nexts) != 2:
            return self.none(node, inspect.currentframe().f_lineno)

        add_next = g.next_nodes(node.output[0])
        if len(add_next) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        add_1 = add_next[0]
        if add_1.op_type != "Add" or add_1.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(add_1.input[1]) or g.get_constant_scalar(add_1.input[1]) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        muls = g.next_nodes(add_1.output[0])
        if len(muls) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        mul = muls[0]
        if mul.op_type != "Mul" or mul.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if set(mul.input) != {add.output[0], add_1.output[0]}:
            return self.none(node, inspect.currentframe().f_lineno)

        halves = g.next_nodes(mul.output[0])
        if len(halves) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        half = halves[0]
        if half.op_type != "Mul" or half.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        index = 1 if half.input[0] == mul.output[0] else 0
        if (
            not g.is_constant_scalar(half.input[index])
            or g.get_constant_scalar(half.input[index]) != 0.5
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [add, div, node, add_1, mul, half], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        add_node: NodeProto,
        div_node: NodeProto,
        erf_node: NodeProto,
        add_1_node: NodeProto,
        mul_node: NodeProto,
        half_node: NodeProto,
    ) -> List[NodeProto]:
        return [
            g.make_node(
                "BiasGelu",
                add_node.input,
                half_node.output,
                domain="com.microsoft",
                doc_string=erf_node.doc_string,
                name=f"{self.__class__.__name__}--{erf_node.name}",
            )
        ]


class BiasSplitGeluPattern(PatternOptimization):
    """
    Replaces by ``y = BiasSplitGelu(x, B)``::

        t = x + B
        t1, t2 = Split(t, 2, axis=-1)
        y = t1 * Gelu(t2)

    where ``Gelu(t2) = t2 * 0.5 * (1 + Erf(t2 / sqrt(2)))``.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_B(["B FLOAT(8)"])
            I_X(["X FLOAT(2, 4, 8)"])

            Constant_0[["Constant() -#gt; B"]]
            Add_1[["Add(., .)"]]
            Split_2[["Split(., axis=-1)"]]
            Div_3[["Div(., [1.4140625])"]]
            Erf_4[["Erf(.)"]]
            Add_5[["Add(., [1.0])"]]
            Mul_6[["Mul(., .)"]]
            Mul_7[["Mul(., [0.5])"]]
            Mul_8[["Mul(., .)"]]

            I_X -->|"FLOAT(2, 4, 8)"| Add_1
            Constant_0 -->|"FLOAT(8)"| Add_1
            Add_1 -->|"FLOAT(2, 4, 8)"| Split_2
            Split_2 -->|"FLOAT(2, 4, 4)"| Div_3
            Split_2 -->|"FLOAT(2, 4, 4)"| Mul_6
            Split_2 -->|"FLOAT(2, 4, 4)"| Mul_8
            Div_3 -->|"FLOAT(2, 4, 4)"| Erf_4
            Erf_4 -->|"FLOAT(2, 4, 4)"| Add_5
            Add_5 -->|"FLOAT(2, 4, 4)"| Mul_6
            Mul_6 -->|"FLOAT(2, 4, 4)"| Mul_7
            Mul_7 -->|"FLOAT(2, 4, 4)"| Mul_8

            O_Y(["Y FLOAT(2, 4, 4)"])
            Mul_8 --> O_Y

            class I_B,I_X,O_Y ioNode
            class Constant_0 constNode
            class Add_1,Split_2,Div_3,Erf_4,Add_5,Mul_6,Mul_7,Mul_8 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_B(["B FLOAT(8)"])
            I_X(["X FLOAT(2, 4, 8)"])

            BiasSplitGelu_0[["com.microsoft.BiasSplitGelu(., .)"]]

            I_X -->|"FLOAT(2, 4, 8)"| BiasSplitGelu_0
            I_B -->|"FLOAT(8)"| BiasSplitGelu_0

            O_Y(["Y FLOAT(2, 4, 4)"])
            BiasSplitGelu_0 --> O_Y

            class I_B,I_X,O_Y ioNode
            class BiasSplitGelu_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Erf" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        div = g.node_before(node.input[0])
        if div is None or div.op_type != "Div" or div.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if (
            not g.is_constant_scalar(div.input[1])
            or g.get_constant_scalar(div.input[1]) != 1.4140625
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # The Div input must come from the right half (output[1]) of a Split node.
        split = g.node_before(div.input[0])
        if split is None or split.op_type != "Split" or split.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if len(split.output) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        # BiasSplitGelu computes left * Gelu(right): Gelu goes on the right half (output[1]).
        if div.input[0] != split.output[1]:
            return self.none(node, inspect.currentframe().f_lineno)

        # Check Split axis=-1.
        atts = g.get_attributes_with_default(split, axis=0)
        if atts["axis"] != -1:
            return self.none(node, inspect.currentframe().f_lineno)

        # The Split input must be Add(X, bias) where bias is a constant.
        add = g.node_before(split.input[0])
        if add is None or add.op_type != "Add" or add.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(add.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        # split.output[1] (right half) must be used by exactly 2 nodes: Div and the inner Mul.
        right_nexts = g.next_nodes(split.output[1])
        if len(right_nexts) != 2:
            return self.none(node, inspect.currentframe().f_lineno)

        # Traverse the Gelu chain: Erf -> Add(_, 1.0) -> Mul(t2, _) -> Mul(_, 0.5)
        add_next = g.next_nodes(node.output[0])
        if len(add_next) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        add_1 = add_next[0]
        if add_1.op_type != "Add" or add_1.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(add_1.input[1]) or g.get_constant_scalar(add_1.input[1]) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        muls = g.next_nodes(add_1.output[0])
        if len(muls) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        mul = muls[0]
        if mul.op_type != "Mul" or mul.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if set(mul.input) != {split.output[1], add_1.output[0]}:
            return self.none(node, inspect.currentframe().f_lineno)

        halves = g.next_nodes(mul.output[0])
        if len(halves) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        half = halves[0]
        if half.op_type != "Mul" or half.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        index = 1 if half.input[0] == mul.output[0] else 0
        if (
            not g.is_constant_scalar(half.input[index])
            or g.get_constant_scalar(half.input[index]) != 0.5
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # The Gelu result must be multiplied by split.output[0] (the left half).
        finals = g.next_nodes(half.output[0])
        if len(finals) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        final = finals[0]
        if final.op_type != "Mul" or final.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if split.output[0] not in final.input:
            return self.none(node, inspect.currentframe().f_lineno)

        # split.output[0] (left half) must only be used by the final Mul.
        left_nexts = g.next_nodes(split.output[0])
        if len(left_nexts) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self, [add, split, div, node, add_1, mul, half, final], self.apply, insert_at=node
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        add_node: NodeProto,
        split_node: NodeProto,
        div_node: NodeProto,
        erf_node: NodeProto,
        add_1_node: NodeProto,
        mul_node: NodeProto,
        half_node: NodeProto,
        final_node: NodeProto,
    ) -> List[NodeProto]:
        return [
            g.make_node(
                "BiasSplitGelu",
                add_node.input,
                final_node.output,
                domain="com.microsoft",
                doc_string=erf_node.doc_string,
                name=f"{self.__class__.__name__}--{erf_node.name}",
            )
        ]


class GeluOrtPattern(GeluPattern):
    """
    Detects the decomposed version of Gelu with Tanh

    .. math::

        y = \\frac{x}{2} \\left(1 + \\tanh\\left(\\sqrt{\\frac{2}{\\pi}}
        (x + 0.044715 * x^3)\\right)\\right)

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_linear_73(["linear_73 FLOAT16(4, 512, 128)"])

            Pow_0[["Pow(., [3.0])"]]
            Mul_1[["Mul(., 0.0447)"]]
            Add_2[["Add(., .)"]]
            Mul_3[["Mul(., 0.798)"]]
            Tanh_4[["Tanh(.)"]]
            Add_5[["Add(., 1.0)"]]
            Mul_6[["Mul(., 0.5)"]]
            Mul_7[["Mul(., .)"]]

            I_linear_73 -->|"FLOAT16(4, 512, 128)"| Pow_0
            Pow_0 -->|"FLOAT16(4, 512, 128)"| Mul_1
            I_linear_73 -->|"FLOAT16(4, 512, 128)"| Add_2
            Mul_1 -->|"FLOAT16(4, 512, 128)"| Add_2
            Add_2 -->|"FLOAT16(4, 512, 128)"| Mul_3
            Mul_3 -->|"FLOAT16(4, 512, 128)"| Tanh_4
            Tanh_4 -->|"FLOAT16(4, 512, 128)"| Add_5
            I_linear_73 -->|"FLOAT16(4, 512, 128)"| Mul_6
            Mul_6 -->|"FLOAT16(4, 512, 128)"| Mul_7
            Add_5 -->|"FLOAT16(4, 512, 128)"| Mul_7

            O_mul_52(["mul_52 FLOAT16(4, 512, 128)"])
            Mul_7 --> O_mul_52

            class I_linear_73,O_mul_52 ioNode
            class Pow_0,Mul_1,Add_2,Mul_3,Tanh_4,Add_5,Mul_6,Mul_7 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_linear_73(["linear_73 FLOAT16(4, 512, 128)"])

            Gelu_0[["com.microsoft.Gelu(.)"]]

            I_linear_73 -->|"FLOAT16(4, 512, 128)"| Gelu_0

            O_mul_52(["mul_52 FLOAT16(4, 512, 128)"])
            Gelu_0 --> O_mul_52

            class I_linear_73,O_mul_52 ioNode
            class Gelu_0 opNode
    """

    def __init__(
        self,
        verbose: int = 0,
        priority: int = 0,
        min_opset: int = 1,
        domain: str = "com.microsoft",
    ):
        super().__init__(verbose, priority, min_opset=min_opset)
        self.domain = domain


class GeluErfPattern(EasyPatternOptimization):
    """
    Detects the decomposed version of Gelu with Erf.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(2, 2, 4, 8)"])

            Div_0[["Div(., [1.4140625])"]]
            Erf_1[["Erf(.)"]]
            Add_2[["Add(., [1.0])"]]
            Mul_3[["Mul(., .)"]]
            Mul_4[["Mul([0.5], .)"]]

            I_X -->|"FLOAT(2, 2, 4, 8)"| Div_0
            Div_0 -->|"FLOAT(2, 2, 4, 8)"| Erf_1
            Erf_1 -->|"FLOAT(2, 2, 4, 8)"| Add_2
            I_X -->|"FLOAT(2, 2, 4, 8)"| Mul_3
            Add_2 -->|"FLOAT(2, 2, 4, 8)"| Mul_3
            Mul_3 -->|"FLOAT(2, 2, 4, 8)"| Mul_4

            O_Y(["Y FLOAT(2, 2, 4, 8)"])
            Mul_4 --> O_Y

            class I_X,O_Y ioNode
            class Div_0,Erf_1,Add_2,Mul_3,Mul_4 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(2, 2, 4, 8)"])

            Gelu_0[["com.microsoft.Gelu(.)"]]

            I_X -->|"FLOAT(2, 2, 4, 8)"| Gelu_0

            O_Y(["Y FLOAT(2, 2, 4, 8)"])
            Gelu_0 --> O_Y

            class I_X,O_Y ioNode
            class Gelu_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0, min_opset: int = 1):
        super().__init__(verbose, priority, min_opset=min_opset)

    def match_pattern(self, g: "GraphBuilder", x, cst2, one, c05):  # noqa: F821
        xd = g.op.Div(x, cst2)  # 1.4140625
        exd = g.op.Erf(xd)
        aexd = g.op.Add(exd, one)  # 1
        mul = g.op.Mul(x, aexd)
        return g.op.Mul(c05, mul)  # 0.5

    def apply_pattern(self, g: "GraphBuilder", x, cst2, one, c05):  # noqa: F821
        return g.anyop.Gelu(x, domain="com.microsoft")

    def validate_mapping(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        deleted_nodes: List[NodeProto],
        pattern_nodes: Optional[List[NodeProto]] = None,
    ) -> bool:
        assert len(deleted_nodes) == 5, f"Unexpected pattern length {len(deleted_nodes)}"
        assert deleted_nodes[0].op_type == "Div", f"-- {deleted_nodes[0]}"
        cst2 = deleted_nodes[0].input[1]
        assert deleted_nodes[2].op_type == "Add", f"-- {deleted_nodes[2]}"
        one = deleted_nodes[2].input[1]
        assert deleted_nodes[4].op_type == "Mul", f"-- {deleted_nodes[4]}"
        c05 = deleted_nodes[4].input[0]

        node = deleted_nodes[1]
        if not g.is_constant_scalar(cst2) or g.get_constant_scalar(cst2) != 1.4140625:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(one) or g.get_constant_scalar(one) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(c05) or g.get_constant_scalar(c05) != 0.5:
            return self.none(node, inspect.currentframe().f_lineno)
        return True


class FastGeluPattern(PatternOptimization):
    """
    Replaces Gelu by FastGelu.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_linear_65(["linear_65 FLOAT16(4, 512, 16384)"])

            Gelu_0[["Gelu(.)"]]

            I_linear_65 -->|"FLOAT16(4, 512, 16384)"| Gelu_0

            O_mul_44(["mul_44 FLOAT16(4, 512, 16384)"])
            Gelu_0 --> O_mul_44

            class I_linear_65,O_mul_44 ioNode
            class Gelu_0 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_linear_65(["linear_65 FLOAT16(4, 512, 16384)"])

            FastGelu_0[["com.microsoft.FastGelu(.)"]]

            I_linear_65 -->|"FLOAT16(4, 512, 16384)"| FastGelu_0

            O_mul_44(["mul_44 FLOAT16(4, 512, 16384)"])
            FastGelu_0 --> O_mul_44

            class I_linear_65,O_mul_44 ioNode
            class FastGelu_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Gelu" or node.domain not in ("", "com.microsoft"):
            return self.none()
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(self, g: "GraphBuilder", gelu_node: NodeProto) -> List[NodeProto]:  # noqa: F821
        return [
            g.make_node(
                "FastGelu",
                gelu_node.input,
                gelu_node.output,
                domain="com.microsoft",
                doc_string=gelu_node.doc_string,
                name=f"{self.__class__.__name__}--{gelu_node.name}",
            )
        ]


class BiasSoftmaxPattern(PatternOptimization):
    """
    Replaces Softmax(Add(x,y), axis=-1) by BiasSoftmax(x,y,axis=-1)

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(16, 8, 4, 8)"])
            I_Y(["Y FLOAT(16, 1, 4, 8)"])

            Add_0[["Add(., .)"]]
            Softmax_1[["Softmax(., axis=-1)"]]

            I_X -->|"FLOAT(16, 8, 4, 8)"| Add_0
            I_Y -->|"FLOAT(16, 1, 4, 8)"| Add_0
            Add_0 -->|"FLOAT(16, 8, 4, 8)"| Softmax_1

            O_Z(["Z FLOAT(16, 8, 4, 8)"])
            Softmax_1 --> O_Z

            class I_X,I_Y,O_Z ioNode
            class Add_0,Softmax_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(16, 8, 4, 8)"])
            I_Y(["Y FLOAT(16, 1, 4, 8)"])

            BiasSoftmax_0[["com.microsoft.BiasSoftmax(., ., axis=-1)"]]

            I_X -->|"FLOAT(16, 8, 4, 8)"| BiasSoftmax_0
            I_Y -->|"FLOAT(16, 1, 4, 8)"| BiasSoftmax_0

            O_Z(["Z FLOAT(16, 8, 4, 8)"])
            BiasSoftmax_0 --> O_Z

            class I_X,I_Y,O_Z ioNode
            class BiasSoftmax_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not g.has_processor("CUDA"):
            return self.none()
        if node.op_type != "Softmax" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        atts = g.get_attributes_with_default(node, axis=-1)
        if atts["axis"] != -1:
            return self.none(node, inspect.currentframe().f_lineno)
        before = g.node_before(node.input[0])
        if before is None or before.op_type != "Add":
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [before, node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", add_node: NodeProto, softmax_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        return [
            g.make_node(
                "BiasSoftmax",
                add_node.input,
                softmax_node.output,
                axis=-1,
                is_inner_broadcast=0,
                domain="com.microsoft",
                doc_string=softmax_node.doc_string,
                name=f"{self.__class__.__name__}--{softmax_node.name}",
            )
        ]


class QuickGeluPattern(PatternOptimization):
    """
    Replaces Mul(x, Sigmoid(x)) by QuickGelu(x, alpha=1)

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(1, 8, 6, 6)"])

            Sigmoid_0[["Sigmoid(.)"]]
            Mul_1[["Mul(., .)"]]

            I_X -->|"FLOAT(1, 8, 6, 6)"| Sigmoid_0
            I_X -->|"FLOAT(1, 8, 6, 6)"| Mul_1
            Sigmoid_0 -->|"FLOAT(1, 8, 6, 6)"| Mul_1

            O_Y(["Y FLOAT(1, 8, 6, 6)"])
            Mul_1 --> O_Y

            class I_X,O_Y ioNode
            class Sigmoid_0,Mul_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(1, 8, 6, 6)"])

            QuickGelu_0[["com.microsoft.QuickGelu(.)"]]

            I_X -->|"FLOAT(1, 8, 6, 6)"| QuickGelu_0

            O_Y(["Y FLOAT(1, 8, 6, 6)"])
            QuickGelu_0 --> O_Y

            class I_X,O_Y ioNode
            class QuickGelu_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Sigmoid" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        after = g.next_nodes(node.output[0])
        if not after or after[0].op_type != "Mul":
            return self.none(node, inspect.currentframe().f_lineno)
        mul_node = after[0]
        if node.input[0] not in mul_node.input:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node, mul_node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", sigmoid: NodeProto, mul_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        return [
            g.make_node(
                "QuickGelu",
                sigmoid.input,
                mul_node.output,
                alpha=1.0,
                domain="com.microsoft",
                doc_string=sigmoid.doc_string,
                name=f"{self.__class__.__name__}--{sigmoid.name}",
            )
        ]
