import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ...xshape._shape_helper import all_int
from ..patterns_api import MatchResult, PatternOptimization


class FusedMatMulDivPattern(PatternOptimization):
    """
    Replaces the Matmul, Div into FusedMatMul.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_Y(["Y FLOAT(2, 2, 128, 64)"])
            I_X(["X FLOAT(2, 2, 32, 128)"])

            FusedMatMul_0[["com.microsoft.FusedMatMul(., .)"]]
            Div_1[["Div(., [2.0])"]]

            I_X -->|"FLOAT(2, 2, 32, 128)"| FusedMatMul_0
            I_Y -->|"FLOAT(2, 2, 128, 64)"| FusedMatMul_0
            FusedMatMul_0 -->|"FLOAT(2, 2, 32, 64)"| Div_1

            O_Z(["Z FLOAT(2, 2, 32, 64)"])
            Div_1 --> O_Z

            class I_Y,I_X,O_Z ioNode
            class FusedMatMul_0,Div_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_Y(["Y FLOAT(2, 2, 128, 64)"])
            I_X(["X FLOAT(2, 2, 32, 128)"])

            FusedMatMul_0[["com.microsoft.FusedMatMul(., .)"]]

            I_X -->|"FLOAT(2, 2, 32, 128)"| FusedMatMul_0
            I_Y -->|"FLOAT(2, 2, 128, 64)"| FusedMatMul_0

            O_Z(["Z FLOAT(2, 2, 32, 64)"])
            FusedMatMul_0 --> O_Z

            class I_Y,I_X,O_Z ioNode
            class FusedMatMul_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (node.op_type != "MatMul" or node.domain != "") and (
            node.op_type != "FusedMatMul" or node.domain != "com.microsoft"
        ):
            return self.none()

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        op_type = next_nodes[0].op_type
        if op_type not in ("Mul", "Div"):
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant_scalar(next_nodes[0].input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_nodes[0]], self.apply, insert_at=next_nodes[0])

    def apply(
        self, g: "GraphBuilder", node: NodeProto, node_div: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        alpha = 1.0
        atts = []
        if node.op_type == "FusedMatMul":
            for att in node.attribute:
                if att.name == "alpha":
                    alpha *= att.f
                else:
                    atts.append(att)

        cst = g.get_computed_constant(node_div.input[1])
        scale = float(cst if len(cst.shape) == 0 else cst[0])
        if node_div.op_type == "Div":
            alpha /= scale
        else:
            alpha *= scale

        mm = g.make_node(
            "FusedMatMul",
            node.input,
            node_div.output,
            domain="com.microsoft",
            alpha=alpha,
            name=f"{self.__class__.__name__}--{node.name}",
        )
        if atts:
            mm.attribute.extend(atts)
        return [mm]


class FusedMatMulPattern(PatternOptimization):
    """
    Replaces the sequence Transpose, Matmul into FusedMatMul.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(2, 2, 32, 128)"])
            I_Y(["Y FLOAT(2, 2, 64, 128)"])

            Transpose_0[["Transpose(., perm=[0, 1, 3, 2])"]]
            MatMul_1[["MatMul(., .)"]]

            I_X -->|"FLOAT(2, 2, 32, 128)"| Transpose_0
            I_Y -->|"FLOAT(2, 2, 64, 128)"| MatMul_1
            Transpose_0 -->|"FLOAT(2, 2, 128, 32)"| MatMul_1

            O_Z(["Z FLOAT(2, 2, 64, 32)"])
            MatMul_1 --> O_Z

            class I_X,I_Y,O_Z ioNode
            class Transpose_0,MatMul_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(2, 2, 32, 128)"])
            I_Y(["Y FLOAT(2, 2, 64, 128)"])

            FusedMatMul_0[["com.microsoft.FusedMatMul(., .)"]]

            I_Y -->|"FLOAT(2, 2, 64, 128)"| FusedMatMul_0
            I_X -->|"FLOAT(2, 2, 32, 128)"| FusedMatMul_0

            O_Z(["Z FLOAT(2, 2, 64, 32)"])
            FusedMatMul_0 --> O_Z

            class I_X,I_Y,O_Z ioNode
            class FusedMatMul_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (node.op_type != "MatMul" or node.domain != "") and (
            node.op_type != "FusedMatMul" or node.domain != "com.microsoft"
        ):
            return self.none()

        if node.op_type == "FusedMatMul":
            transA = g.get_attribute(node, "transA", exc=False) or 0
            transB = g.get_attribute(node, "transB", exc=False) or 0
            if transA != transB:
                # one side is already transposed.
                return self.none(node, inspect.currentframe().f_lineno)

        if not g.has_rank(node.input[0]) or not g.has_rank(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_rank(node.input[0]) < 2 or g.get_rank(node.input[1]) < 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_rank(node.input[0]) <= 2 and g.get_rank(node.input[1]) <= 2:
            # Regular Gemm.
            return self.none(node, inspect.currentframe().f_lineno)

        nodes_before = [g.node_before(node.input[0]), g.node_before(node.input[1])]
        ns = [
            (n if n is not None and n.op_type == "Transpose" and n.domain == "" else None)
            for n in nodes_before
        ]
        if len([_ for _ in ns if _ is not None]) == 0:
            return self.none(node, inspect.currentframe().f_lineno)

        if g.has_processor("CUDA"):
            nns = []
            for n in ns:
                if n is None:
                    nns.append(n)
                    continue
                if g.is_used_more_than_once(n.output[0]):
                    nns.append(None)
                    continue
                nns.append(n)
            if len([_ for _ in ns if _ is not None]) == 0:
                return self.none(node, inspect.currentframe().f_lineno)
            ns = nns

        hints = []
        found = False
        nns = []
        for n in ns:
            if n is None:
                nns.append(None)
                continue
            perm = list(g.get_attribute(n, "perm").ints)
            expecting = list(range(len(perm)))
            expecting[-2], expecting[-1] = expecting[-1], expecting[-2]
            if perm != expecting:
                hints.append(dict(expecting=expecting, perm=perm))
                nns.append(None)
                continue
            found = True
            nns.append(n)

        ns = nns
        if not found:
            # unexpected transpose
            return self.none(node, inspect.currentframe().f_lineno, lambda: f"hints={hints}")

        # At this stage, one or two inputs are transposed before being used.
        # MatMul or Gemm are operating on 2D tensors.
        nodes = [*ns, node]
        if nodes[0] is not None and nodes[1] is not None:
            # Both are available, we only transpose one.
            nodes[0] = None
        if not g.is_used_more_than_once(node.output[0]):
            next_node = g.next_node(node.output[0])
            if (
                next_node.op_type in {"Div", "Mul"}
                and next_node.domain == ""
                and g.is_constant_scalar(next_node.input[1])
            ):
                # The node can be fused with matmul
                nodes.append(next_node)

        return MatchResult(self, nodes, self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        node_before_left: Optional[NodeProto],
        node_before_right: Optional[NodeProto],
        node: NodeProto,
        scale: Optional[NodeProto] = None,
    ) -> List[NodeProto]:
        inputs = [
            (node.input[0] if node_before_left is None else node_before_left.input[0]),
            (node.input[1] if node_before_right is None else node_before_right.input[0]),
            *node.input[2:],
        ]

        transA = 0 if node_before_left is None else 1
        transB = 0 if node_before_right is None else 1
        transBatchA = 0
        transBatchB = 0
        keep = []
        for att in node.attribute:
            if att.name in {"alpha", "beta"}:
                keep.append(att)
            elif att.name == "transA":
                transA = (att.i + transA) % 2
            elif att.name == "transB":
                transB = (att.i + transB) % 2
            elif att.name == "transBatchA":
                transBatchA = att.i
            elif att.name == "transBatchB":
                transBatchB = att.i
            else:
                raise NotImplementedError(
                    f"Unexpected attribute {att.name!r}={att} for node={node}"
                )

        kwargs = dict(
            transA=transA, transB=transB, transBatchA=transBatchA, transBatchB=transBatchB
        )

        if scale is not None:
            # Let's include the scale as well
            cst = g.get_computed_constant(scale.input[1])
            value = float(cst[0] if cst.shape == (1,) else cst)
            assert scale.op_type in {
                "Div",
                "Mul",
            }, f"Match did not check next_node type {scale.op_type!r}"
            alpha = value if scale.op_type == "Mul" else (1.0 / value)
            kwargs["alpha"] = alpha
            output = scale.output[0]
        else:
            output = node.output[0]

        new_node = g.make_node(
            "FusedMatMul",
            inputs,
            [output],
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
            domain="com.microsoft",
            **kwargs,
        )
        new_node.attribute.extend(keep)
        res = [new_node]
        if node_before_left is not None and g.is_used_more_than_once(node_before_left.output[0]):
            # This is not efficient on CUDA.
            res.append(node_before_left)
        if node_before_right is not None and g.is_used_more_than_once(
            node_before_right.output[0]
        ):
            # This is not efficient on CUDA.
            res.append(node_before_right)
        return res


class FusedMatMulx2Pattern(PatternOptimization):
    """
    Replaces the sequence Div by a scalar consumed by two FusedMatMul.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(2, 2, 4, 4)"])

            Div_0[["Div(., [2.0])"]]
            FusedMatMul_1[["com.microsoft.FusedMatMul(., .)"]]
            FusedMatMul_2[["com.microsoft.FusedMatMul(., .)"]]

            I_X -->|"FLOAT(2, 2, 4, 4)"| Div_0
            Div_0 -->|"FLOAT(2, 2, 4, 4)"| FusedMatMul_1
            I_X -->|"FLOAT(2, 2, 4, 4)"| FusedMatMul_1
            I_X -->|"FLOAT(2, 2, 4, 4)"| FusedMatMul_2
            Div_0 -->|"FLOAT(2, 2, 4, 4)"| FusedMatMul_2

            O_x2(["x2 FLOAT(2, 2, 4, 4)"])
            FusedMatMul_2 --> O_x2
            O_x1(["x1 FLOAT(2, 2, 4, 4)"])
            FusedMatMul_1 --> O_x1

            class I_X,O_x2,O_x1 ioNode
            class Div_0,FusedMatMul_1,FusedMatMul_2 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(2, 2, 4, 4)"])

            FusedMatMul_0[["com.microsoft.FusedMatMul(., .)"]]
            FusedMatMul_1[["com.microsoft.FusedMatMul(., .)"]]

            I_X -->|"FLOAT(2, 2, 4, 4)"| FusedMatMul_0
            I_X -->|"FLOAT(2, 2, 4, 4)"| FusedMatMul_1

            O_x2(["x2 FLOAT(2, 2, 4, 4)"])
            FusedMatMul_1 --> O_x2
            O_x1(["x1 FLOAT(2, 2, 4, 4)"])
            FusedMatMul_0 --> O_x1

            class I_X,O_x2,O_x1 ioNode
            class FusedMatMul_0,FusedMatMul_1 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 3):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (node.op_type not in "MatMul" or node.domain != "") and (
            node.op_type != "FusedMatMul" or node.domain != "com.microsoft"
        ):
            return self.none()

        div_node = None
        for name in node.input:
            n = g.node_before(name)
            if n is None:
                continue
            if n.op_type not in {"Mul", "Div"} or n.domain != "":
                continue
            if not g.is_constant_scalar(n.input[1]):
                continue
            div_node = n
            break

        if div_node is None:
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(div_node.output[0])
        op_types = [n.op_type for n in next_nodes]
        if any(t not in {"FusedMatMul", "MatMul"} for t in op_types):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [div_node, *next_nodes], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        div_node: Optional[NodeProto],
        *mnodes: Optional[NodeProto],
    ) -> List[NodeProto]:
        cst = g.get_constant_scalar(div_node.input[1])
        if div_node.op_type == "Div":
            cst = 1.0 / cst

        new_nodes = []
        for node in mnodes:
            alpha = 1.0
            atts = []
            for att in node.attribute:
                if att.name == "alpha":
                    alpha = float(att.f)
                else:
                    atts.append(att)
            new_inputs = [
                (div_node.input[0] if i == div_node.output[0] else i) for i in node.input
            ]
            alpha *= cst
            new_node = g.make_node(
                "FusedMatMul",
                new_inputs,
                node.output,
                domain="com.microsoft",
                alpha=alpha,
                name=f"{self.__class__.__name__}--{node.name}",
            )
            if atts:
                new_node.attribute.extend(atts)
            new_nodes.append(new_node)
        return new_nodes


class FusedMatMulTransposePattern(PatternOptimization):
    """
    Replaces the sequence (Fused)Matmul(A,B) + Transpose
    into FusedMatMul(B.T, A.T).

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(2, 2, 6, 3)"])
            I_Y(["Y FLOAT(2, 2, 5, 6)"])

            FusedMatMul_0[["com.microsoft.FusedMatMul(., .)"]]
            Transpose_1[["Transpose(., perm=[0, 1, 3, 2])"]]

            I_X -->|"FLOAT(2, 2, 6, 3)"| FusedMatMul_0
            I_Y -->|"FLOAT(2, 2, 5, 6)"| FusedMatMul_0
            FusedMatMul_0 -->|"FLOAT(2, 2, 3, 5)"| Transpose_1

            O_Z(["Z FLOAT(2, 2, UNKNOWNDIM, UNKNOWNDIM1)"])
            Transpose_1 --> O_Z

            class I_X,I_Y,O_Z ioNode
            class FusedMatMul_0,Transpose_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(2, 2, 6, 3)"])
            I_Y(["Y FLOAT(2, 2, 5, 6)"])

            FusedMatMul_0[["com.microsoft.FusedMatMul(., .)"]]

            I_Y -->|"FLOAT(2, 2, 5, 6)"| FusedMatMul_0
            I_X -->|"FLOAT(2, 2, 6, 3)"| FusedMatMul_0

            O_Z(["Z FLOAT(2, 2, UNKNOWNDIM, UNKNOWNDIM1)"])
            FusedMatMul_0 --> O_Z

            class I_X,I_Y,O_Z ioNode
            class FusedMatMul_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 3):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (node.op_type != "MatMul" or node.domain != "") and (
            node.op_type != "FusedMatMul" or node.domain != "com.microsoft"
        ):
            return self.none()

        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        next_nodes = g.next_nodes(node.output[0])
        if (
            len(next_nodes) != 1
            or next_nodes[0].op_type != "Transpose"
            or next_nodes[0].domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        transpose_node = next_nodes[0]
        perm = list(g.get_attribute(transpose_node, "perm").ints)
        if len(perm) > 2:
            if perm[:-2] != list(range(len(perm) - 2)):
                return self.none(node, inspect.currentframe().f_lineno)
        if perm[-2:] != [len(perm) - 1, len(perm) - 2]:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, transpose_node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", node: NodeProto, transpose_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        default_values = dict(transA=0, transB=0, transBatchA=0, transBatchB=0, alpha=1.0)
        kwargs = g.get_attributes_with_default(node, **default_values)
        kwargs["transA"], kwargs["transB"] = 1 - kwargs["transB"], 1 - kwargs["transA"]
        remove = []
        for k in kwargs:
            if kwargs[k] == default_values[k]:
                remove.append(k)
        for r in remove:
            del kwargs[r]
        new_node = g.make_node(
            "FusedMatMul",
            [node.input[1], node.input[0]],
            transpose_node.output,
            domain="com.microsoft",
            name=f"{self.__class__.__name__}--{node.name}",
            **kwargs,
        )
        return [new_node]


class ReshapeGemmPattern(PatternOptimization):
    """
    Replaces the sequence Reshape(-1, ...) + Gemm into FusedMatMul().

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_B(["B FLOAT(4, 8)"])
            I_A(["A FLOAT(a, b, 8)"])

            Reshape_0[["Reshape(., [-1, 8])"]]
            Gemm_1[["Gemm(., .)"]]

            I_A -->|"FLOAT(a, b, 8)"| Reshape_0
            Reshape_0 -->|"FLOAT(a*b, 8)"| Gemm_1
            I_B -->|"FLOAT(4, 8)"| Gemm_1

            O_Y(["Y FLOAT(f, g)"])
            Gemm_1 --> O_Y

            class I_B,I_A,O_Y ioNode
            class Reshape_0,Gemm_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_B(["B FLOAT(4, 8)"])
            I_A(["A FLOAT(a, b, 8)"])

            FusedMatMul_0[["com.microsoft.FusedMatMul(., .)"]]
            Reshape_1[["Reshape(., [-1, 4])"]]

            I_A -->|"FLOAT(a, b, 8)"| FusedMatMul_0
            I_B -->|"FLOAT(4, 8)"| FusedMatMul_0
            FusedMatMul_0 -->|"FLOAT(a, b, 4)"| Reshape_1

            O_Y(["Y FLOAT(f, g)"])
            Reshape_1 --> O_Y

            class I_B,I_A,O_Y ioNode
            class FusedMatMul_0,Reshape_1 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 3):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Gemm" or node.domain != "" or len(node.input) == 3:
            return self.none()

        transA = g.get_attributes_with_default(node, transA=0)["transA"]
        if transA != 0:
            return self.none(node, inspect.currentframe().f_lineno)
        node_before = g.node_before(node.input[0])
        if node_before is None or node_before.op_type != "Reshape" or node_before.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape = g.get_shape(node.input[1])
        if not all_int(shape):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(node_before.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        shape = g.get_computed_constant(node_before.input[1])
        if shape.shape != (2,) or shape[0] != -1:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node_before, node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", reshape_node: NodeProto, gemm_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        kwargs = {}
        transB = g.get_attributes_with_default(gemm_node, transB=0)["transB"]
        if transB:
            kwargs["transB"] = transB
        gemm_output = g.unique_name(f"{self.__class__.__name__}--{gemm_node.output[0]}")
        new_node = g.make_node(
            "FusedMatMul",
            [reshape_node.input[0], *gemm_node.input[1:]],
            [gemm_output],
            domain="com.microsoft",
            name=f"{self.__class__.__name__}--{gemm_node.name}",
            **kwargs,
        )
        shape = g.get_shape(gemm_node.input[1])
        new_shape = g.make_initializer(
            "",
            np.array([-1, shape[1 - transB]], dtype=np.int64),
            source=f"ReshapeGemm.shape({gemm_node.name})",
        )
        reshape_node = g.make_node(
            "Reshape",
            [gemm_output, new_shape],
            gemm_node.output,
            name=f"{self.__class__.__name__}--{gemm_node.name}",
        )
        return [new_node, reshape_node]


class ReshapeGemmReshapePattern(PatternOptimization):
    """
    Replaces the sequence Reshape + Gemm + Reshape into FusedMatMul.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_B(["B FLOAT(8, 4)"])
            I_A(["A FLOAT(a, b, c)"])
            I_shapey(["shapey INT64(e)"])

            Reshape_0[["Reshape(., [-1, 8])"]]
            Gemm_1[["Gemm(., .)"]]
            Reshape_2[["Reshape(., .)"]]

            I_A -->|"FLOAT(a, b, c)"| Reshape_0
            Reshape_0 -->|"FLOAT(a*b*c//8, 8)"| Gemm_1
            I_B -->|"FLOAT(8, 4)"| Gemm_1
            Gemm_1 -->|"FLOAT(a*b*c//8, 4)"| Reshape_2
            I_shapey -->|"INT64(e)"| Reshape_2

            O_Y(["Y FLOAT(a, b, c)"])
            Reshape_2 --> O_Y

            class I_B,I_A,I_shapey,O_Y ioNode
            class Reshape_0,Gemm_1,Reshape_2 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_B(["B FLOAT(8, 4)"])
            I_A(["A FLOAT(a, b, c)"])

            FusedMatMul_0[["com.microsoft.FusedMatMul(., .)"]]

            I_A -->|"FLOAT(a, b, c)"| FusedMatMul_0
            I_B -->|"FLOAT(8, 4)"| FusedMatMul_0

            O_Y(["Y FLOAT(a, b, c)"])
            FusedMatMul_0 --> O_Y

            class I_B,I_A,O_Y ioNode
            class FusedMatMul_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 3):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Gemm" or node.domain != "" or len(node.input) != 2:
            return self.none()
        transA = g.get_attributes_with_default(node, transA=0)["transA"]
        if transA != 0:
            return self.none(node, inspect.currentframe().f_lineno)
        node_before = g.node_before(node.input[0])
        if node_before is None or node_before.op_type != "Reshape" or node_before.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(node_before.input[1])
        if cst is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if cst[0] != -1:
            return self.none(node, inspect.currentframe().f_lineno)
        next_nodes = g.next_nodes(node.output[0])
        if (
            len(next_nodes) != 1
            or next_nodes[0].op_type != "Reshape"
            or next_nodes[0].domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node_before.input[0]) or not g.has_shape(next_nodes[0].output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape_final = g.get_shape(next_nodes[0].output[0])[:-1]
        if g.get_shape(node_before.input[0])[: len(shape_final)] != shape_final:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node_before, node, next_nodes[0]], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        reshape_before: NodeProto,
        gemm_node: NodeProto,
        reshape_after: NodeProto,
    ) -> List[NodeProto]:
        kwargs = {}
        transB = g.get_attributes_with_default(gemm_node, transB=0)["transB"]
        if transB:
            kwargs["transB"] = transB
        return [
            g.make_node(
                "FusedMatMul",
                [reshape_before.input[0], *gemm_node.input[1:]],
                [reshape_after.output[0]],
                domain="com.microsoft",
                name=f"{self.__class__.__name__}--{gemm_node.name}",
                **kwargs,
            )
        ]


class TransposeFusedMatMulBPattern(PatternOptimization):
    """
    Replaces the sequence Transpose(B, [0, 2, 3, 1] + (Fused)Matmul(A,B)
    into Transpose(A, [0, 2, 1, 3]) + FusedMatMul(A, B, transB=1).

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_B(["B FLOAT(i, j, k, l)"])
            I_A(["A FLOAT(a, b, c, d)"])

            Transpose_0[["Transpose(., perm=[0, 2, 3, 1])"]]
            MatMul_1[["MatMul(., .)"]]

            I_B -->|"FLOAT(i, j, k, l)"| Transpose_0
            I_A -->|"FLOAT(a, b, c, d)"| MatMul_1
            Transpose_0 -->|"FLOAT(i, k, l, j)"| MatMul_1

            O_Y(["Y FLOAT(m, n, o, p)"])
            MatMul_1 --> O_Y

            class I_B,I_A,O_Y ioNode
            class Transpose_0,MatMul_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_B(["B FLOAT(i, j, k, l)"])
            I_A(["A FLOAT(a, b, c, d)"])

            Transpose_0[["Transpose(., perm=[0, 2, 1, 3])"]]
            FusedMatMul_1[["com.microsoft.FusedMatMul(., .)"]]

            I_B -->|"FLOAT(i, j, k, l)"| Transpose_0
            I_A -->|"FLOAT(a, b, c, d)"| FusedMatMul_1
            Transpose_0 -->|"FLOAT(i, k, j, l)"| FusedMatMul_1

            O_Y(["Y FLOAT(m, n, o, p)"])
            FusedMatMul_1 --> O_Y

            class I_B,I_A,O_Y ioNode
            class Transpose_0,FusedMatMul_1 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 3):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (node.op_type not in "MatMul" or node.domain != "") and (
            node.op_type != "FusedMatMul" or node.domain != "com.microsoft"
        ):
            return self.none()
        if g.is_used_more_than_once(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        transB = g.get_attributes_with_default(node, transB=0)["transB"]
        if transB != 0:
            return self.none(node, inspect.currentframe().f_lineno)

        transpose_node = g.node_before(node.input[1])
        if (
            transpose_node is None
            or transpose_node.op_type != "Transpose"
            or transpose_node.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        perm = list(g.get_attribute(transpose_node, "perm").ints)
        if perm != [0, 2, 3, 1]:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [transpose_node, node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", transpose_node: NodeProto, node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        tout = g.unique_name(f"{self.__class__.__name__}--{node.input[1]}")
        nodes = [
            g.make_node(
                "Transpose",
                transpose_node.input,
                [tout],
                name=f"{self.__class__.__name__}--{node.name}",
                perm=[0, 2, 1, 3],
            ),
            g.make_node(
                "FusedMatMul",
                [node.input[0], tout],
                node.output,
                domain="com.microsoft",
                name=f"{self.__class__.__name__}--{node.name}",
                transB=1,
            ),
        ]
        for att in node.attribute:
            if att.name != "transB":
                nodes[-1].attribute.append(att)
        return nodes


class FusedMatMulActivationPattern(PatternOptimization):
    """
    Replaces the sequence (Fused)MatMul followed by an activation function
    into com.microsoft.FusedMatMulActivation.

    Supported activations: ``Relu``, ``Tanh``, ``Sigmoid``, ``LeakyRelu``,
    ``HardSigmoid``.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(2, 2, 32, 64)"])
            I_Y(["Y FLOAT(2, 2, 64, 16)"])

            FusedMatMul_0[["com.microsoft.FusedMatMul(., .)"]]
            Relu_1[["Relu(.)"]]

            I_X -->|"FLOAT(2, 2, 32, 64)"| FusedMatMul_0
            I_Y -->|"FLOAT(2, 2, 64, 16)"| FusedMatMul_0
            FusedMatMul_0 -->|"FLOAT(2, 2, 32, 16)"| Relu_1

            O_Z(["Z FLOAT(2, 2, 32, 16)"])
            Relu_1 --> O_Z

            class I_X,I_Y,O_Z ioNode
            class FusedMatMul_0,Relu_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(2, 2, 32, 64)"])
            I_Y(["Y FLOAT(2, 2, 64, 16)"])

            FusedMatMulActivation_0[["com.microsoft.FusedMatMulActivation(., .)"]]

            I_X -->|"FLOAT(2, 2, 32, 64)"| FusedMatMulActivation_0
            I_Y -->|"FLOAT(2, 2, 64, 16)"| FusedMatMulActivation_0

            O_Z(["Z FLOAT(2, 2, 32, 16)"])
            FusedMatMulActivation_0 --> O_Z

            class I_X,I_Y,O_Z ioNode
            class FusedMatMulActivation_0 opNode
    """

    #: Activation op types (ONNX domain ``""``) fused without extra parameters.
    _SIMPLE_ACTIVATIONS = frozenset({"Relu", "Tanh", "Sigmoid"})

    #: Activation op types that carry extra scalar parameters.
    _PARAMETRIC_ACTIVATIONS = frozenset({"LeakyRelu", "HardSigmoid"})

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (node.op_type != "MatMul" or node.domain != "") and (
            node.op_type != "FusedMatMul" or node.domain != "com.microsoft"
        ):
            return self.none()

        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        act_node = next_nodes[0]
        if act_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if act_node.op_type not in self._SIMPLE_ACTIVATIONS | self._PARAMETRIC_ACTIVATIONS:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, act_node], self.apply, insert_at=act_node)

    def apply(
        self, g: "GraphBuilder", node: NodeProto, act_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        kwargs = {}
        for att in node.attribute:
            if att.name in {"alpha", "transA", "transB", "transBatchA", "transBatchB"}:
                kwargs[att.name] = att.f if att.name == "alpha" else att.i

        kwargs["activation"] = act_node.op_type

        if act_node.op_type == "LeakyRelu":
            alpha_val = 0.01
            for att in act_node.attribute:
                if att.name == "alpha":
                    alpha_val = att.f
            kwargs["activation_alpha"] = alpha_val
        elif act_node.op_type == "HardSigmoid":
            alpha_val = 0.2
            beta_val = 0.5
            for att in act_node.attribute:
                if att.name == "alpha":
                    alpha_val = att.f
                elif att.name == "beta":
                    beta_val = att.f
            kwargs["activation_alpha"] = alpha_val
            kwargs["activation_beta"] = beta_val

        return [
            g.make_node(
                "FusedMatMulActivation",
                node.input,
                act_node.output,
                domain="com.microsoft",
                name=f"{self.__class__.__name__}--{node.name}",
                **kwargs,
            )
        ]
