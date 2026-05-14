import inspect
from typing import List, Optional
from onnx import NodeProto, TensorProto
from ..patterns_api import EasyPatternOptimization, MatchResult, PatternOptimization


class GeluPattern(EasyPatternOptimization):
    """
    Detects the decomposed version of Gelu with Tanh

    .. math::

        y = \\frac{x}{2}
        \\left(1 + \\tanh\\left(\\sqrt{\\frac{2}{\\pi}} (x + 0.044715 * x^3)\\right)\\right)

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_linear_5(["linear_5 FLOAT16(4, 512, 16384)"])

            Pow_0[["Pow(., [3.0])"]]
            Mul_1[["Mul(., 0.0447)"]]
            Add_2[["Add(., .)"]]
            Mul_3[["Mul(., 0.798)"]]
            Tanh_4[["Tanh(.)"]]
            Add_5[["Add(., 1.0)"]]
            Mul_6[["Mul(., 0.5)"]]
            Mul_7[["Mul(., .)"]]

            I_linear_5 -->|"FLOAT16(4, 512, 16384)"| Pow_0
            Pow_0 -->|"FLOAT16(4, 512, 16384)"| Mul_1
            I_linear_5 -->|"FLOAT16(4, 512, 16384)"| Add_2
            Mul_1 -->|"FLOAT16(4, 512, 16384)"| Add_2
            Add_2 -->|"FLOAT16(4, 512, 16384)"| Mul_3
            Mul_3 -->|"FLOAT16(4, 512, 16384)"| Tanh_4
            Tanh_4 -->|"FLOAT16(4, 512, 16384)"| Add_5
            I_linear_5 -->|"FLOAT16(4, 512, 16384)"| Mul_6
            Mul_6 -->|"FLOAT16(4, 512, 16384)"| Mul_7
            Add_5 -->|"FLOAT16(4, 512, 16384)"| Mul_7

            O_mul_4(["mul_4 FLOAT16(4, 512, 16384)"])
            Mul_7 --> O_mul_4

            class I_linear_5,O_mul_4 ioNode
            class Pow_0,Mul_1,Add_2,Mul_3,Tanh_4,Add_5,Mul_6,Mul_7 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_linear_5(["linear_5 FLOAT16(4, 512, 16384)"])

            Gelu_0[["Gelu(.)"]]

            I_linear_5 -->|"FLOAT16(4, 512, 16384)"| Gelu_0

            O_mul_4(["mul_4 FLOAT16(4, 512, 16384)"])
            Gelu_0 --> O_mul_4

            class I_linear_5,O_mul_4 ioNode
            class Gelu_0 opNode
    """

    def __init__(
        self, verbose: int = 0, priority: int = 0, min_opset: int = 20, domain: str = ""
    ):
        super().__init__(verbose, priority, min_opset=min_opset)
        self.domain = domain

    def match_pattern(self, g: "GraphBuilder", x, c3, c04, cpi, one, c2):  # noqa: F821
        x3 = g.op.Pow(x, c3)  # 3
        cx3 = g.op.Mul(x3, c04)  # 0.044715
        add = g.op.Add(x, cx3)
        addm = g.op.Mul(add, cpi)  # 0.7978515625 = 2/pi
        tanh = g.op.Tanh(addm)
        tanh1 = g.op.Add(tanh, one)  # 1
        x2 = g.op.Mul(x, c2)  # 0.5
        return g.op.Mul(x2, tanh1)

    def apply_pattern(self, g: "GraphBuilder", x, c3, c04, cpi, one, c2):  # noqa: F821
        return g.op.Gelu(x, approximate="tanh", domain=self.domain)

    def validate_mapping(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        deleted_nodes: List[NodeProto],
        pattern_nodes: Optional[List[NodeProto]] = None,
    ) -> bool:
        assert len(deleted_nodes) == 8, f"Unexpected pattern length {len(deleted_nodes)}"
        assert deleted_nodes[0].op_type == "Pow", f"-- {deleted_nodes[0]}"
        c3 = deleted_nodes[0].input[1]
        assert deleted_nodes[1].op_type == "Mul", f"-- {deleted_nodes[1]}"
        cx3 = deleted_nodes[1].input[1]
        assert deleted_nodes[3].op_type == "Mul", f"-- {deleted_nodes[3]}"
        cpi = deleted_nodes[3].input[1]
        assert deleted_nodes[5].op_type == "Add", f"-- {deleted_nodes[5]}"
        one = deleted_nodes[5].input[1]
        assert deleted_nodes[6].op_type == "Mul", f"-- {deleted_nodes[6]}"
        c2 = deleted_nodes[6].input[1]

        node = deleted_nodes[0]

        if not g.is_constant_scalar(c3) or g.get_constant_scalar(c3) != 3:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(cx3) or g.get_constant_scalar(cx3) not in (
            0.044715,
            0.044708251953125,
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(cpi) or g.get_constant_scalar(cpi) != 0.7978515625:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(one) or g.get_constant_scalar(one) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(c2) or g.get_constant_scalar(c2) != 0.5:
            return self.none(node, inspect.currentframe().f_lineno)
        return True


class LeakyReluPattern(EasyPatternOptimization):
    """
    Detects the decomposed version of LeakyRelu.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X1(["X1 FLOAT(3, 3)"])

            Greater_0[["Greater(., [0.0])"]]
            Mul_1[["Mul(., [-0.33])"]]
            Where_2[["Where(., ., .)"]]

            I_X1 -->|"FLOAT(3, 3)"| Greater_0
            I_X1 -->|"FLOAT(3, 3)"| Mul_1
            Greater_0 -->|"BOOL(3, 3)"| Where_2
            I_X1 -->|"FLOAT(3, 3)"| Where_2
            Mul_1 -->|"FLOAT(3, 3)"| Where_2

            O_Y(["Y FLOAT(3, 3)"])
            Where_2 --> O_Y

            class I_X1,O_Y ioNode
            class Greater_0,Mul_1,Where_2 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X1(["X1 FLOAT(3, 3)"])

            LeakyRelu_0[["LeakyRelu(.)"]]

            I_X1 -->|"FLOAT(3, 3)"| LeakyRelu_0

            O_Y(["Y FLOAT(3, 3)"])
            LeakyRelu_0 --> O_Y

            class I_X1,O_Y ioNode
            class LeakyRelu_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0, min_opset: int = 6):
        super().__init__(verbose, priority, min_opset=min_opset)

    def match_pattern(self, g: "GraphBuilder", x, zero, slope):  # noqa: F821
        return g.op.Where(g.op.Greater(x, zero), x, g.op.Mul(x, slope))

    def apply_pattern(self, g: "GraphBuilder", x, zero, slope):  # noqa: F821
        # g is not the GraphBuilder for the main graph.
        return g.op.LeakyRelu(x, alpha=self.get_validate_param("slope"))

    def validate_mapping(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        deleted_nodes: List[NodeProto],
        pattern_nodes: Optional[List[NodeProto]] = None,
    ) -> bool:
        assert len(deleted_nodes) == 3, f"Unexpected pattern length {len(deleted_nodes)}"
        assert deleted_nodes[2].op_type == "Where", f"-- {deleted_nodes[2]}"
        greater, mul = (
            (deleted_nodes[0], deleted_nodes[1])
            if deleted_nodes[0].op_type == "Greater"
            else (deleted_nodes[1], deleted_nodes[0])
        )
        zero = greater.input[1]
        slope = mul.input[1]

        if not g.is_constant_scalar(zero) or g.get_constant_scalar(zero) != 0:
            return self.none(greater, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(slope):
            return self.none(mul, inspect.currentframe().f_lineno)
        self.add_validate_param("slope", g.get_constant_scalar(slope))
        return True


class SoftmaxCrossEntropyLossCastPattern(EasyPatternOptimization):
    """
    Detects one decomposed version of SoftmaxCrossEntropyLoss.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_I(["I INT64(A)"])
            I_X(["X FLOAT16(A, B)"])

            Equal_0[["Equal(., [-100])"]]
            Not_1[["Not(.)"]]
            Where_2[["Where(., ., [0])"]]
            Unsqueeze_3[["Unsqueeze(., [1])"]]
            LogSoftmax_4[["LogSoftmax(., axis=1)"]]
            GatherElements_5[["GatherElements(., ., axis=1)"]]
            Squeeze_6[["Squeeze(., [1])"]]
            Neg_7[["Neg(.)"]]
            Where_8[["Where(., ., [0.0])"]]
            Cast_9[["Cast(., to=FLOAT)"]]
            ReduceSum_10[["ReduceSum(.)"]]
            Cast_11[["Cast(., to=FLOAT16)"]]
            Cast_12[["Cast(., to=FLOAT)"]]
            ReduceSum_13[["ReduceSum(.)"]]
            Cast_14[["Cast(., to=FLOAT16)"]]
            Div_15[["Div(., .)"]]

            I_I -->|"INT64(A)"| Equal_0
            Equal_0 -->|"BOOL(A)"| Not_1
            Not_1 -->|"BOOL(A)"| Where_2
            I_I -->|"INT64(A)"| Where_2
            Where_2 -->|"INT64(A)"| Unsqueeze_3
            I_X -->|"FLOAT16(A, B)"| LogSoftmax_4
            LogSoftmax_4 -->|"FLOAT16(A, B)"| GatherElements_5
            Unsqueeze_3 -->|"INT64(A, 1)"| GatherElements_5
            GatherElements_5 -->|"FLOAT16(A, 1)"| Squeeze_6
            Squeeze_6 -->|"FLOAT16(A)"| Neg_7
            Not_1 -->|"BOOL(A)"| Where_8
            Neg_7 -->|"FLOAT16(A)"| Where_8
            Not_1 -->|"BOOL(A)"| Cast_9
            Cast_9 -->|"FLOAT(A)"| ReduceSum_10
            ReduceSum_10 -->|"FLOAT()"| Cast_11
            Where_8 -->|"FLOAT16(A)"| Cast_12
            Cast_12 -->|"FLOAT(A)"| ReduceSum_13
            ReduceSum_13 -->|"FLOAT()"| Cast_14
            Cast_14 -->|"FLOAT16()"| Div_15
            Cast_11 -->|"FLOAT16()"| Div_15

            O_Y(["Y FLOAT16()"])
            Div_15 --> O_Y

            class I_I,I_X,O_Y ioNode
            class Equal_0,Not_1,Where_2,Unsqueeze_3,LogSoftmax_4,GatherElements_5,Squeeze_6 opNode
            class Neg_7,Where_8,Cast_9,ReduceSum_10,Cast_11,Cast_12,ReduceSum_13 opNode
            class Cast_14,Div_15 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_I(["I INT64(A)"])
            I_X(["X FLOAT16(A, B)"])

            SoftmaxCrossEntropyLoss_0[["SoftmaxCrossEntropyLoss(., .)"]]

            I_X -->|"FLOAT16(A, B)"| SoftmaxCrossEntropyLoss_0
            I_I -->|"INT64(A)"| SoftmaxCrossEntropyLoss_0

            O_Y(["Y FLOAT16()"])
            SoftmaxCrossEntropyLoss_0 --> O_Y

            class I_I,I_X,O_Y ioNode
            class SoftmaxCrossEntropyLoss_0 opNode
    """

    def __init__(
        self, verbose: int = 0, priority: int = 0, min_opset: int = 14, domain: str = ""
    ):
        super().__init__(verbose, priority, min_opset=min_opset)
        self.domain = domain

    def match_pattern(self, g: "GraphBuilder", X, indices, axis, zerof, zeroi, b):  # noqa: F821
        neq1 = g.op.Not(g.op.Equal(indices, b))
        wh1 = g.op.Where(neq1, indices, zeroi)
        uns = g.op.Unsqueeze(wh1, axis)
        ge = g.op.GatherElements(g.op.LogSoftmax(X, axis=1), uns, axis=1)
        wh2 = g.op.Where(neq1, g.op.Neg(g.op.Squeeze(ge, axis)), zerof)
        denominator = g.op.Cast(
            g.op.ReduceSum(g.op.Cast(neq1, to=TensorProto.FLOAT), keepdims=0),
            to=TensorProto.FLOAT16,
        )
        numerator = g.op.Cast(
            g.op.ReduceSum(g.op.Cast(wh2, to=TensorProto.FLOAT), keepdims=0),
            to=TensorProto.FLOAT16,
        )
        return g.op.Div(numerator, denominator)

    @classmethod
    def apply_pattern(cls, g: "GraphBuilder", X, indices, axis, zerof, zeroi, b):  # noqa: F821
        return g.op.SoftmaxCrossEntropyLoss(X, indices, ignore_index=-100, reduction="mean")

    def validate_mapping(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        deleted_nodes: List[NodeProto],
        pattern_nodes: Optional[List[NodeProto]] = None,
    ) -> bool:
        assert len(deleted_nodes) == 16, f"Unexpected pattern length {len(deleted_nodes)}"
        node = deleted_nodes[-1]

        for n in deleted_nodes:
            if n.op_type in {"Squeeze", "Unsqueeze"}:
                c = n.input[1]
                if not g.is_constant_scalar(c) or g.get_constant_scalar(c) != 1:
                    return self.none(node, inspect.currentframe().f_lineno)
                continue
            if n.op_type in {"Equal"}:
                c = n.input[1]
                if not g.is_constant_scalar(c) or g.get_constant_scalar(c) != -100:
                    return self.none(node, inspect.currentframe().f_lineno)
                continue
        return True


class MaxReluPattern(PatternOptimization):
    """
    Replaces ``Max(x, 0)`` or ``Max(0, x)`` with ``Relu(x)``.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b)"])
            I_zero(["zero FLOAT(1)"])

            Constant_0[["Constant() -#gt; zero"]]
            Max_1[["Max(., .)"]]

            I_X -->|"FLOAT(a, b)"| Max_1
            Constant_0 -->|"FLOAT(1)"| Max_1

            O_Y(["Y FLOAT(a, b)"])
            Max_1 --> O_Y

            class I_X,O_Y ioNode
            class Constant_0 constNode
            class Max_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b)"])

            Relu_0[["Relu(.)"]]

            I_X -->|"FLOAT(a, b)"| Relu_0

            O_Y(["Y FLOAT(a, b)"])
            Relu_0 --> O_Y

            class I_X,O_Y ioNode
            class Relu_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Max" or node.domain != "":
            return self.none()
        if len(node.input) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_type(node.input[0]) or g.get_type(node.input[0]) not in {
            TensorProto.FLOAT,
            TensorProto.FLOAT16,
            TensorProto.INT16,
            TensorProto.INT32,
        }:
            return self.none(node, inspect.currentframe().f_lineno)

        # Require exactly one zero-valued constant input among the two inputs.
        zero_const_inputs = [
            inp
            for inp in node.input
            if g.is_constant_scalar(inp) and g.get_constant_scalar(inp) == 0
        ]
        if len(zero_const_inputs) != 1:
            # Either no zero-constant inputs or both inputs are zero-constants:
            # do not apply the Max->Relu fusion in these cases.
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(self, g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
        # Identify which input is the non-zero tensor
        x = None
        for inp in node.input:
            if not (g.is_constant_scalar(inp) and g.get_constant_scalar(inp) == 0):
                x = inp
                break

        return [
            g.make_node("Relu", [x], node.output, name=f"{self.__class__.__name__}--{node.name}")
        ]
