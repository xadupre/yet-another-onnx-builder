import inspect
from typing import List, Optional, Tuple
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


def _get_gather_source(
    g: "GraphBuilderPatternOptimization", name: str, expected_index: int  # noqa: F821
) -> Optional[str]:
    """Returns the source name if ``name`` is produced by
    ``Gather(source, expected_index, axis=-1)``.

    Returns ``None`` when the node does not match.
    """
    node = g.node_before(name)
    if node is None or node.op_type != "Gather" or node.domain != "":
        return None
    atts = g.get_attributes_with_default(node, axis=0)
    axis = atts["axis"]
    if axis != -1:
        if g.has_rank(node.input[0]):
            if axis != g.get_rank(node.input[0]) - 1:
                return None
        else:
            return None
    if not g.is_constant_scalar(node.input[1]):
        return None
    if g.get_constant_scalar(node.input[1]) != expected_index:
        return None
    return node.input[0]


def _is_unsqueeze_last_dim(
    g: "GraphBuilderPatternOptimization", node: Optional[NodeProto]  # noqa: F821
) -> bool:
    """Returns ``True`` when ``node`` is ``Unsqueeze(x, axes=[-1])``."""
    if node is None or node.op_type != "Unsqueeze" or node.domain != "":
        return False
    if len(node.input) >= 2:
        if not g.is_constant(node.input[1]):
            return False
        axes_val = g.get_computed_constant(node.input[1])
        if axes_val is None:
            return False
        axes = list(np.array(axes_val).reshape(-1))
        return len(axes) == 1 and int(axes[0]) == -1
    atts = g.get_attributes_with_default(node, axes=None)
    axes = atts.get("axes")
    if axes is None:
        return False
    return len(axes) == 1 and axes[0] == -1


def _classify_mul(
    g: "GraphBuilderPatternOptimization", mul_output: str, src_a: str, src_b: str  # noqa: F821
) -> Optional[Tuple[int, int]]:
    """Returns ``(a_idx, b_idx)`` when ``mul_output`` is
    ``Mul(Gather(src_a, a_idx, -1), Gather(src_b, b_idx, -1))``.

    Handles commutativity of ``Mul``. Returns ``None`` when no match.
    """
    node = g.node_before(mul_output)
    if node is None or node.op_type != "Mul" or node.domain != "":
        return None
    for inp1, inp2 in [(node.input[0], node.input[1]), (node.input[1], node.input[0])]:
        for a_idx in (0, 1):
            a_src = _get_gather_source(g, inp1, a_idx)
            if a_src != src_a:
                continue
            for b_idx in (0, 1):
                b_src = _get_gather_source(g, inp2, b_idx)
                if b_src != src_b:
                    continue
                return (a_idx, b_idx)
    return None


def _find_gather_node(
    g: "GraphBuilderPatternOptimization", mul_node: NodeProto, src: str, idx: int  # noqa: F821
) -> Optional[NodeProto]:
    """Returns the ``Gather(src, idx, axis=-1)`` node consumed by ``mul_node``."""
    for inp in mul_node.input:
        if _get_gather_source(g, inp, idx) == src:
            return g.node_before(inp)
    return None


def _parse_complex_mul_core(
    g: "GraphBuilderPatternOptimization",  # noqa: F821
    real_node: NodeProto,
    imag_node: NodeProto,
    real_sub_imag_add: bool,
) -> Optional[Tuple]:
    """Parses the four Mul+Gather components of complex multiplication.

    When ``real_sub_imag_add`` is ``True``, expects::

        real_node = Sub(A_r * B_r, A_i * B_i)  # real part for ComplexMul
        imag_node = Add(A_r * B_i, A_i * B_r)  # imag part for ComplexMul

    When ``real_sub_imag_add`` is ``False``, expects::

        real_node = Add(A_r * B_r, A_i * B_i)  # real part for ComplexMulConj
        imag_node = Sub(A_i * B_r, A_r * B_i)  # imag part for ComplexMulConj

    Returns ``(A, B, gather_a0, gather_a1, gather_b0, gather_b1, mul_rr, mul_ii, mul_ri, mul_ir)``
    or ``None`` when no match.
    """
    # Retrieve the four Mul nodes from the real and imag nodes.
    real_mul1 = g.node_before(real_node.input[0])
    real_mul2 = g.node_before(real_node.input[1])
    imag_mul1 = g.node_before(imag_node.input[0])
    imag_mul2 = g.node_before(imag_node.input[1])

    for mn in [real_mul1, real_mul2, imag_mul1, imag_mul2]:
        if mn is None or mn.op_type != "Mul" or mn.domain != "":
            return None

    # Each Mul input must come from a Gather node; collect all unique source tensors.
    def gather_sources(mul_node):
        srcs = set()
        for inp in mul_node.input[:2]:
            for idx in (0, 1):
                src = _get_gather_source(g, inp, idx)
                if src is not None:
                    srcs.add(src)
                    break
        return srcs

    all_sources = set()
    for mn in [real_mul1, real_mul2, imag_mul1, imag_mul2]:
        s = gather_sources(mn)
        if len(s) == 0:
            return None
        all_sources |= s

    if len(all_sources) > 2:
        return None

    sources = list(all_sources)
    candidates = [(sources[0], sources[1]), (sources[1], sources[0])] if len(sources) == 2 else []
    if len(sources) == 1:
        # Self-multiplication: A = B = the single source
        candidates = [(sources[0], sources[0])]

    for A, B in candidates:
        if real_sub_imag_add:
            # ComplexMul:
            # real = Sub(rr, ii): real_mul1=(A,0)*(B,0), real_mul2=(A,1)*(B,1)
            # imag = Add(ri, ir): one of {(A,0)*(B,1)} and one of {(A,1)*(B,0)}
            r1 = _classify_mul(g, real_node.input[0], A, B)
            r2 = _classify_mul(g, real_node.input[1], A, B)
            if r1 != (0, 0) or r2 != (1, 1):
                continue
            # imag Add can have either ordering of ri/ir
            i_opts = [
                (
                    _classify_mul(g, imag_node.input[0], A, B),
                    _classify_mul(g, imag_node.input[1], A, B),
                )
            ]
            valid_imag = None
            for ia, ib in i_opts:
                if frozenset([ia, ib]) == frozenset([(0, 1), (1, 0)]):
                    valid_imag = (ia, ib)
                    break
            if valid_imag is None:
                continue
            # Assign roles
            ia, ib = valid_imag
            if ia == (0, 1):
                mul_ri = imag_mul1
                mul_ir = imag_mul2
            else:
                mul_ri = imag_mul2
                mul_ir = imag_mul1
            mul_rr = real_mul1
            mul_ii = real_mul2
        else:
            # ComplexMulConj:
            # real = Add(rr, ii): either ordering of (A,0)*(B,0) and (A,1)*(B,1)
            # imag = Sub(ir, ri): imag_mul1=(A,1)*(B,0), imag_mul2=(A,0)*(B,1)
            r_opts = [
                (
                    _classify_mul(g, real_node.input[0], A, B),
                    _classify_mul(g, real_node.input[1], A, B),
                )
            ]
            valid_real = None
            for ra, rb in r_opts:
                if frozenset([ra, rb]) == frozenset([(0, 0), (1, 1)]):
                    valid_real = (ra, rb)
                    break
            if valid_real is None:
                continue
            ra, rb = valid_real
            if ra == (0, 0):
                mul_rr = real_mul1
                mul_ii = real_mul2
            else:
                mul_rr = real_mul2
                mul_ii = real_mul1

            i1 = _classify_mul(g, imag_node.input[0], A, B)
            i2 = _classify_mul(g, imag_node.input[1], A, B)
            # imag = Sub(ir, ri): first input is ir=(A,1)*(B,0), second is ri=(A,0)*(B,1)
            if i1 != (1, 0) or i2 != (0, 1):
                continue
            mul_ir = imag_mul1
            mul_ri = imag_mul2

        # Retrieve the actual Gather nodes.
        g_a0 = _find_gather_node(g, mul_rr, A, 0)
        g_a1 = _find_gather_node(g, mul_ii, A, 1)
        g_b0 = _find_gather_node(g, mul_rr, B, 0)
        g_b1 = _find_gather_node(g, mul_ii, B, 1)
        if any(gn is None for gn in [g_a0, g_a1, g_b0, g_b1]):
            continue

        return (A, B, g_a0, g_a1, g_b0, g_b1, mul_rr, mul_ii, mul_ri, mul_ir)

    return None


class ComplexMulPattern(PatternOptimization):
    """
    Replaces a decomposed complex multiplication by ``com.microsoft.ComplexMul(A, B)``.

    Complex multiplication is defined as::

        C_r = A_r * B_r - A_i * B_i
        C_i = A_r * B_i + A_i * B_r

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_A(["A FLOAT(a, b, 2)"])
            I_B(["B FLOAT(a, b, 2)"])

            Gather_0[["Gather(., [0], axis=-1)"]]
            Gather_1[["Gather(., [1], axis=-1)"]]
            Gather_2[["Gather(., [0], axis=-1)"]]
            Gather_3[["Gather(., [1], axis=-1)"]]
            Mul_4[["Mul(., .)"]]
            Mul_5[["Mul(., .)"]]
            Mul_6[["Mul(., .)"]]
            Mul_7[["Mul(., .)"]]
            Sub_8[["Sub(., .)"]]
            Add_9[["Add(., .)"]]
            Unsqueeze_10[["Unsqueeze(., [-1])"]]
            Unsqueeze_11[["Unsqueeze(., [-1])"]]
            Concat_12[["Concat(., ., axis=-1)"]]

            I_A -->|"FLOAT(a, b)"| Gather_0
            I_A -->|"FLOAT(a, b)"| Gather_1
            I_B -->|"FLOAT(a, b)"| Gather_2
            I_B -->|"FLOAT(a, b)"| Gather_3
            Gather_0 --> Mul_4
            Gather_2 --> Mul_4
            Gather_1 --> Mul_5
            Gather_3 --> Mul_5
            Gather_0 --> Mul_6
            Gather_3 --> Mul_6
            Gather_1 --> Mul_7
            Gather_2 --> Mul_7
            Mul_4 --> Sub_8
            Mul_5 --> Sub_8
            Mul_6 --> Add_9
            Mul_7 --> Add_9
            Sub_8 --> Unsqueeze_10
            Add_9 --> Unsqueeze_11
            Unsqueeze_10 --> Concat_12
            Unsqueeze_11 --> Concat_12

            O_C(["C FLOAT(a, b, 2)"])
            Concat_12 --> O_C

            class I_A,I_B,O_C ioNode
            class Gather_0,Gather_1,Gather_2,Gather_3 opNode
            class Mul_4,Mul_5,Mul_6,Mul_7,Sub_8,Add_9,Unsqueeze_10,Unsqueeze_11,Concat_12 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_A(["A FLOAT(a, b, 2)"])
            I_B(["B FLOAT(a, b, 2)"])

            ComplexMul_0[["com.microsoft.ComplexMul(., .)"]]

            I_A -->|"FLOAT(a, b, 2)"| ComplexMul_0
            I_B -->|"FLOAT(a, b, 2)"| ComplexMul_0

            O_C(["C FLOAT(a, b, 2)"])
            ComplexMul_0 --> O_C

            class I_A,I_B,O_C ioNode
            class ComplexMul_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Concat" or node.domain != "":
            return self.none()

        atts = g.get_attributes_with_default(node, axis=0)
        if atts["axis"] != -1:
            return self.none(node, inspect.currentframe().f_lineno)
        if len(node.input) != 2:
            return self.none(node, inspect.currentframe().f_lineno)

        unsq_r = g.node_before(node.input[0])
        unsq_i = g.node_before(node.input[1])
        if not _is_unsqueeze_last_dim(g, unsq_r):
            return self.none(node, inspect.currentframe().f_lineno)
        if not _is_unsqueeze_last_dim(g, unsq_i):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(unsq_r.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(unsq_i.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        real_node = g.node_before(unsq_r.input[0])
        imag_node = g.node_before(unsq_i.input[0])
        if real_node is None or real_node.op_type != "Sub" or real_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if imag_node is None or imag_node.op_type != "Add" or imag_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(real_node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(imag_node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        result = _parse_complex_mul_core(g, real_node, imag_node, real_sub_imag_add=True)
        if result is None:
            return self.none(node, inspect.currentframe().f_lineno)

        A, B, g_a0, g_a1, g_b0, g_b1, mul_rr, mul_ii, mul_ri, mul_ir = result

        # Each Mul output must be used only within the Sub or Add node.
        for mul_node, expected_consumer in [
            (mul_rr, real_node),
            (mul_ii, real_node),
            (mul_ri, imag_node),
            (mul_ir, imag_node),
        ]:
            nexts = g.next_nodes(mul_node.output[0])
            if len(nexts) != 1 or nexts[0] is not expected_consumer:
                return self.none(node, inspect.currentframe().f_lineno)

        # Each Gather output must only be consumed by the two expected Mul nodes.
        # Use output names (strings) to compare nodes, since NodeProto is not hashable.
        expected_uses = {
            g_a0.output[0]: {mul_rr.output[0], mul_ri.output[0]},
            g_a1.output[0]: {mul_ii.output[0], mul_ir.output[0]},
            g_b0.output[0]: {mul_rr.output[0], mul_ir.output[0]},
            g_b1.output[0]: {mul_ii.output[0], mul_ri.output[0]},
        }
        # Handle the case where A == B (same tensor used as both operands).
        if A == B:
            combined = {}
            for out, expected in expected_uses.items():
                combined[out] = combined.get(out, set()) | expected
            expected_uses = combined

        for gather_out, expected_mul_outs in expected_uses.items():
            nexts = {n.output[0] for n in g.next_nodes(gather_out)}
            if nexts != expected_mul_outs:
                return self.none(node, inspect.currentframe().f_lineno)

        all_nodes = [
            g_a0,
            g_a1,
            g_b0,
            g_b1,
            mul_rr,
            mul_ii,
            mul_ri,
            mul_ir,
            real_node,
            imag_node,
            unsq_r,
            unsq_i,
            node,
        ]

        # Deduplicate when A == B (same Gather nodes appear for A and B slots).
        seen = []
        for n in all_nodes:
            if n not in seen:
                seen.append(n)

        return MatchResult(self, seen, self.apply, insert_at=node)

    def apply(self, g: "GraphBuilder", *nodes: NodeProto) -> List[NodeProto]:  # noqa: F821
        # The concat node is always last in the deduplicated list.
        concat_node = nodes[-1]
        # Identify Gather nodes (op_type == "Gather") to extract A and B.
        gather_nodes = [n for n in nodes if n.op_type == "Gather"]
        # g_a0 is the first Gather (Gather(A, 0)); its source is A.
        A = gather_nodes[0].input[0]
        # g_b0 is Gather(B, 0); find the Gather with source != A (or same if A==B).
        B = None
        for gn in gather_nodes:
            if gn.input[0] != A:
                B = gn.input[0]
                break
        if B is None:
            B = A  # A == B case

        return [
            g.make_node(
                "ComplexMul",
                [A, B],
                concat_node.output,
                domain="com.microsoft",
                name=f"{self.__class__.__name__}--{concat_node.name}",
            )
        ]


class ComplexMulConjPattern(PatternOptimization):
    """
    Replaces a decomposed complex multiplication with conjugate by
    ``com.microsoft.ComplexMulConj(A, B)``.

    Complex multiplication with conjugate is defined as::

        C_r = A_r * B_r + A_i * B_i
        C_i = A_i * B_r - A_r * B_i

    (Equivalent to multiplying A by the complex conjugate of B.)

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_A(["A FLOAT(a, b, 2)"])
            I_B(["B FLOAT(a, b, 2)"])

            Gather_0[["Gather(., [0], axis=-1)"]]
            Gather_1[["Gather(., [1], axis=-1)"]]
            Gather_2[["Gather(., [0], axis=-1)"]]
            Gather_3[["Gather(., [1], axis=-1)"]]
            Mul_4[["Mul(., .)"]]
            Mul_5[["Mul(., .)"]]
            Mul_6[["Mul(., .)"]]
            Mul_7[["Mul(., .)"]]
            Add_8[["Add(., .)"]]
            Sub_9[["Sub(., .)"]]
            Unsqueeze_10[["Unsqueeze(., [-1])"]]
            Unsqueeze_11[["Unsqueeze(., [-1])"]]
            Concat_12[["Concat(., ., axis=-1)"]]

            I_A -->|"FLOAT(a, b)"| Gather_0
            I_A -->|"FLOAT(a, b)"| Gather_1
            I_B -->|"FLOAT(a, b)"| Gather_2
            I_B -->|"FLOAT(a, b)"| Gather_3
            Gather_0 --> Mul_4
            Gather_2 --> Mul_4
            Gather_1 --> Mul_5
            Gather_3 --> Mul_5
            Gather_1 --> Mul_6
            Gather_2 --> Mul_6
            Gather_0 --> Mul_7
            Gather_3 --> Mul_7
            Mul_4 --> Add_8
            Mul_5 --> Add_8
            Mul_6 --> Sub_9
            Mul_7 --> Sub_9
            Add_8 --> Unsqueeze_10
            Sub_9 --> Unsqueeze_11
            Unsqueeze_10 --> Concat_12
            Unsqueeze_11 --> Concat_12

            O_C(["C FLOAT(a, b, 2)"])
            Concat_12 --> O_C

            class I_A,I_B,O_C ioNode
            class Gather_0,Gather_1,Gather_2,Gather_3 opNode
            class Mul_4,Mul_5,Mul_6,Mul_7,Add_8,Sub_9,Unsqueeze_10,Unsqueeze_11,Concat_12 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_A(["A FLOAT(a, b, 2)"])
            I_B(["B FLOAT(a, b, 2)"])

            ComplexMulConj_0[["com.microsoft.ComplexMulConj(., .)"]]

            I_A -->|"FLOAT(a, b, 2)"| ComplexMulConj_0
            I_B -->|"FLOAT(a, b, 2)"| ComplexMulConj_0

            O_C(["C FLOAT(a, b, 2)"])
            ComplexMulConj_0 --> O_C

            class I_A,I_B,O_C ioNode
            class ComplexMulConj_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Concat" or node.domain != "":
            return self.none()

        atts = g.get_attributes_with_default(node, axis=0)
        if atts["axis"] != -1:
            return self.none(node, inspect.currentframe().f_lineno)
        if len(node.input) != 2:
            return self.none(node, inspect.currentframe().f_lineno)

        unsq_r = g.node_before(node.input[0])
        unsq_i = g.node_before(node.input[1])
        if not _is_unsqueeze_last_dim(g, unsq_r):
            return self.none(node, inspect.currentframe().f_lineno)
        if not _is_unsqueeze_last_dim(g, unsq_i):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(unsq_r.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(unsq_i.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        real_node = g.node_before(unsq_r.input[0])
        imag_node = g.node_before(unsq_i.input[0])
        if real_node is None or real_node.op_type != "Add" or real_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if imag_node is None or imag_node.op_type != "Sub" or imag_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(real_node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(imag_node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        result = _parse_complex_mul_core(g, real_node, imag_node, real_sub_imag_add=False)
        if result is None:
            return self.none(node, inspect.currentframe().f_lineno)

        A, B, g_a0, g_a1, g_b0, g_b1, mul_rr, mul_ii, mul_ri, mul_ir = result

        # Each Mul output must be used only within the Add or Sub node.
        for mul_node, expected_consumer in [
            (mul_rr, real_node),
            (mul_ii, real_node),
            (mul_ir, imag_node),
            (mul_ri, imag_node),
        ]:
            nexts = g.next_nodes(mul_node.output[0])
            if len(nexts) != 1 or nexts[0] is not expected_consumer:
                return self.none(node, inspect.currentframe().f_lineno)

        # Each Gather output must only be consumed by the two expected Mul nodes.
        # Use output names (strings) to compare nodes, since NodeProto is not hashable.
        expected_uses = {
            g_a0.output[0]: {mul_rr.output[0], mul_ri.output[0]},
            g_a1.output[0]: {mul_ii.output[0], mul_ir.output[0]},
            g_b0.output[0]: {mul_rr.output[0], mul_ir.output[0]},
            g_b1.output[0]: {mul_ii.output[0], mul_ri.output[0]},
        }
        if A == B:
            combined = {}
            for out, expected in expected_uses.items():
                combined[out] = combined.get(out, set()) | expected
            expected_uses = combined

        for gather_out, expected_mul_outs in expected_uses.items():
            nexts = {n.output[0] for n in g.next_nodes(gather_out)}
            if nexts != expected_mul_outs:
                return self.none(node, inspect.currentframe().f_lineno)

        all_nodes = [
            g_a0,
            g_a1,
            g_b0,
            g_b1,
            mul_rr,
            mul_ii,
            mul_ri,
            mul_ir,
            real_node,
            imag_node,
            unsq_r,
            unsq_i,
            node,
        ]

        seen = []
        for n in all_nodes:
            if n not in seen:
                seen.append(n)

        return MatchResult(self, seen, self.apply, insert_at=node)

    def apply(self, g: "GraphBuilder", *nodes: NodeProto) -> List[NodeProto]:  # noqa: F821
        concat_node = nodes[-1]
        gather_nodes = [n for n in nodes if n.op_type == "Gather"]
        A = gather_nodes[0].input[0]
        B = None
        for gn in gather_nodes:
            if gn.input[0] != A:
                B = gn.input[0]
                break
        if B is None:
            B = A

        return [
            g.make_node(
                "ComplexMulConj",
                [A, B],
                concat_node.output,
                domain="com.microsoft",
                name=f"{self.__class__.__name__}--{concat_node.name}",
            )
        ]
