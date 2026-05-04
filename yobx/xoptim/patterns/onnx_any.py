import inspect
from typing import Dict, List, Optional, Set
from onnx import AttributeProto, NodeProto, TensorProto
from ...helpers.onnx_helper import make_idn, unary_like_op_types
from ..patterns_api import MatchResult, PatternOptimization


class SameChildrenPattern(PatternOptimization):
    """
    Checks there is no duplicated node doing the same than another one beside.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_Y(["Y FLOAT(a, 1, 3, 4)"])
            I_sh1(["sh1 INT64(4)"])
            I_y1(["y1 FLOAT(a, 2, 3, 4)"])

            Expand_0[["Expand(., .)"]]
            Expand_1[["Expand(., .)"]]

            I_Y -->|"FLOAT(a, 1, 3, 4)"| Expand_0
            I_sh1 -->|"INT64(4)"| Expand_0
            I_Y -->|"FLOAT(a, 1, 3, 4)"| Expand_1
            I_sh1 -->|"INT64(4)"| Expand_1

            O_y2(["y2 FLOAT(a, 2, 3, 4)"])
            Expand_1 --> O_y2
            O_y1(["y1 FLOAT(a, 2, 3, 4)"])
            Expand_0 --> O_y1

            class I_Y,I_sh1,I_y1,O_y2,O_y1 ioNode
            class Expand_0,Expand_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_Y(["Y FLOAT(a, 1, 3, 4)"])
            I_sh1(["sh1 INT64(4)"])
            I_y1(["y1 FLOAT(a, 2, 3, 4)"])

            Expand_0[["Expand(., .)"]]
            Identity_1[["Identity(.)"]]

            I_Y -->|"FLOAT(a, 1, 3, 4)"| Expand_0
            I_sh1 -->|"INT64(4)"| Expand_0
            Expand_0 -->|"FLOAT(a, 2, 3, 4)"| Identity_1

            O_y2(["y2 FLOAT(a, 2, 3, 4)"])
            Identity_1 --> O_y2
            O_y1(["y1 FLOAT(a, 2, 3, 4)"])
            Expand_0 --> O_y1

            class I_Y,I_sh1,I_y1,O_y2,O_y1 ioNode
            class Expand_0,Identity_1 opNode
    """

    @classmethod
    def _cmp(cls, n1: NodeProto, n2: NodeProto) -> bool:
        "Compares two nodes and say if they are the same."
        if len(n1.input) != len(n2.input):
            return False
        if n1.input != n2.input:
            if n1.op_type in {"Add", "Mul"} and (
                (n1.input[0] == n2.input[0] and n1.input[1] == n2.input[1])
                or (n1.input[0] == n2.input[1] and n1.input[1] == n2.input[0])
            ):
                return True
            return False
        if len(n1.output) != len(n2.output):
            return False
        if len(n1.attribute) != len(n2.attribute):
            return False
        for att1, att2 in zip(n1.attribute, n2.attribute):
            if att1.name != att2.name:
                return False
            if att1.type != att2.type:
                return False
            if att1.type == AttributeProto.INT:
                if att1.i != att2.i:
                    return False
                continue
            if att1.type == AttributeProto.FLOAT:
                if att1.f != att2.f:
                    return False
                continue
            if att1.type == AttributeProto.STRING:
                if att1.s != att2.s:
                    return False
                continue
            if att1.SerializeToString() != att2.SerializeToString():
                return False
        assert make_idn(n1) != make_idn(n2), f"Two nodes are the same not identical copies {n1}"
        return True

    @classmethod
    def _cmp_with_alias(cls, n1: NodeProto, n2: NodeProto, sames: Dict[str, Set[str]]) -> bool:
        "Compares two nodes and say if they are the same."
        if len(n1.input) != len(n2.input):
            return False
        if len(n1.output) != len(n2.output):
            return False
        if len(n1.attribute) != len(n2.attribute):
            return False
        for i1, i2 in zip(n1.input, n2.input):
            if i1 != i2 and i1 not in sames.get(i2, set()):
                return False
        for att1, att2 in zip(n1.attribute, n2.attribute):
            if att1.name != att2.name:
                return False
            if att1.type != att2.type:
                return False
            if att1.type == AttributeProto.INT:
                if att1.i != att2.i:
                    return False
                continue
            if att1.type == AttributeProto.FLOAT:
                if att1.f != att2.f:
                    return False
                continue
            if att1.type == AttributeProto.STRING:
                if att1.s != att2.s:
                    return False
                continue
            if att1.SerializeToString() != att2.SerializeToString():
                return False
        assert make_idn(n1) != make_idn(n2), f"Two nodes are the same not identical copies {n1}"
        return True

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        # match results
        match = None
        for i in range(len(node.output)):
            next_nodes = g.next_nodes(node.output[i])
            if len(next_nodes) <= 1:
                continue
            match = self._match_with_nodes(g, node, next_nodes)
            if match is not None:
                break
        if match and len(match.nodes) == 2:
            # Then we continue further
            nodes = match.nodes
            node1, node2 = nodes
            sames = {}
            # Let's continue
            name_used_as_outputs = set(o for o in node1.output if o)
            stack = [(node1, node2)]
            cannot_continue = False
            while stack:
                node1, node2 = stack.pop()
                for o1, o2 in zip(node1.output, node2.output):
                    sames[o1] = {o2}
                    sames[o2] = {o1}
                if len(node1.output) != 1 or len(node2.output) != 1:
                    # Shortcut, another iteration will pick it up.
                    break
                o1, o2 = node1.output[0], node2.output[0]
                next1 = g.next_nodes(o1)
                next2 = g.next_nodes(o2)
                dnext1, dnext2 = {}, {}
                for n in next1:
                    if n.op_type in dnext1:
                        dnext1[n.op_type].append(n)
                    else:
                        dnext1[n.op_type] = [n]
                for n in next2:
                    if n.op_type in dnext2:
                        dnext2[n.op_type].append(n)
                    else:
                        dnext2[n.op_type] = [n]
                common = set(dnext1) & set(dnext2)
                for c in common:
                    if c == "Identity":
                        continue
                    for n1 in dnext1[c]:
                        for n2 in dnext2[c]:
                            if id(n1) == id(n2) or n1.domain != n2.domain:
                                continue
                            if self._cmp_with_alias(n1, n2, sames):
                                if any(o in name_used_as_outputs for o in n1.output if o) or any(
                                    o in name_used_as_outputs for o in n2.output if o
                                ):
                                    cannot_continue = True
                                    break
                                nodes.extend([n1, n2])
                                stack.append((n1, n2))
                                name_used_as_outputs |= {o for o in n1.output if o} | {
                                    o for o in n2.output if o
                                }
                    if cannot_continue:
                        break
                if cannot_continue:
                    break
            match = MatchResult(self, nodes, self.apply)

        if match:
            return match
        return self.none()

    def _match_with_nodes(self, g, node, next_nodes) -> Optional[MatchResult]:
        if len(next_nodes) == 2:
            n1, n2 = next_nodes
            if n1.op_type != n2.op_type or n1.op_type == "Identity":
                return self.none()
            if not self._cmp(n1, n2):
                return self.none(node, inspect.currentframe().f_lineno)
            nodes = [n1, n2]
        else:
            cp = {}
            for n in next_nodes:
                if n.op_type == "Identity":
                    continue
                if n.op_type in cp:
                    cp[n.op_type].append(n)
                else:
                    cp[n.op_type] = [n]
            nodes = []
            for v in cp.values():
                if len(v) <= 1:
                    continue
                if len(v) == 2:
                    n1, n2 = v
                    if not self._cmp(n1, n2):
                        continue
                    nodes.extend([n1, n2])
                    continue
                enough = False
                for i in range(len(v) - 1):
                    for j in range(i + 1, len(v)):
                        if self._cmp(v[i], v[j]):
                            nodes.extend([v[i], v[j]])
                            enough = True
                            break
                    if enough:
                        break
            if len(nodes) == 0:
                return self.none(node, inspect.currentframe().f_lineno)

        for i in range(0, len(nodes), 2):
            n1, n2 = nodes[i : i + 2]
            assert len(n1.output) > 0, "A node should not have no output in this pattern."
            assert (
                not g.has_type(n1.output[0])
                or not g.has_type(n2.output[0])
                or g.get_type(n1.output[0]) == g.get_type(n2.output[0])
            ), (
                f"Nodes n1 and n2 have different output type for outputs "
                f"{n1.output[0]!r}, {n2.output[0]}, and types "
                f"{g.get_type(n1.output[0])} != {g.get_type(n2.output[0])})"
            )
        assert "Identity" not in set(n.op_type for n in nodes), (
            f"Identity nodes should be covered by this pattern "
            f"{set(n.op_type for n in nodes)}, type={node.op_type!r}, "
            f"name={node.name!r}."
        )
        return MatchResult(self, nodes, self.apply)

    def apply(self, g: "GraphBuilder", *nodes: NodeProto) -> List[NodeProto]:  # noqa: F821
        """
        The function receives pairs of nodes. We replace every odd node
        by an identity node.
        """
        assert (
            len(nodes) % 2 == 0
        ), f"Expecting an even number of nodes but len(nodes) == {len(nodes)}"
        new_nodes = []
        already_added = set()
        for i in range(0, len(nodes), 2):
            idn = id(nodes[i])
            if idn not in already_added:
                new_nodes.append(nodes[i])
                already_added.add(idn)

            for o1, o2 in zip(nodes[i].output, nodes[i + 1].output):
                if not o1 and not o2:
                    continue
                assert o1, (
                    f"o1 is empty, this is unlikely to happen, fix it when it does, "
                    f"node1={nodes[i]}, node2={nodes[i+1]}"
                )
                new_nodes.append(
                    g.make_node(
                        "Identity",
                        [o1],
                        [o2],
                        name=f"{self.__class__.__name__}--{nodes[i+1].name}",
                        doc_string=nodes[i + 1].doc_string,
                    )
                )
        return new_nodes


class SameChildrenFromInputPattern(SameChildrenPattern):
    """
    Checks there is no duplicated node doing the same than another
    one beside and taking a model input as input.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_xy1(["xy1 FLOAT16(a, 2, 3, 4)"])
            I_X(["X FLOAT(a, 2, 3, 4)"])

            Cast_0[["Cast(., to=FLOAT16)"]]
            Cast_1[["Cast(., to=FLOAT16)"]]

            I_X -->|"FLOAT(a, 2, 3, 4)"| Cast_0
            I_X -->|"FLOAT(a, 2, 3, 4)"| Cast_1

            O_xy1(["xy1 FLOAT16(a, 2, 3, 4)"])
            Cast_0 --> O_xy1
            O_xy2(["xy2 FLOAT16(a, 2, 3, 4)"])
            Cast_1 --> O_xy2

            class I_xy1,I_X,O_xy1,O_xy2 ioNode
            class Cast_0,Cast_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_xy1(["xy1 FLOAT16(a, 2, 3, 4)"])
            I_X(["X FLOAT(a, 2, 3, 4)"])

            Cast_0[["Cast(., to=FLOAT16)"]]
            Identity_1[["Identity(.)"]]

            I_X -->|"FLOAT(a, 2, 3, 4)"| Cast_0
            Cast_0 -->|"FLOAT16(a, 2, 3, 4)"| Identity_1

            O_xy1(["xy1 FLOAT16(a, 2, 3, 4)"])
            Cast_0 --> O_xy1
            O_xy2(["xy2 FLOAT16(a, 2, 3, 4)"])
            Identity_1 --> O_xy2

            class I_xy1,I_X,O_xy1,O_xy2 ioNode
            class Cast_0,Identity_1 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if not node.input:
            return self.none()
        node_before = g.node_before(node.input[0])
        if node_before is not None:
            return self.none()
        next_nodes = g.next_nodes(node.input[0])
        if len(next_nodes) <= 1:
            return self.none()
        return self._match_with_nodes(g, node, next_nodes)


class ShapeBasedSameChildrenPattern(PatternOptimization):
    """
    Checks there is no duplicated node doing the same than another one beside.
    :class:`yobx.xoptim.patterns.onnx_any.SameChildrenPattern`
    checks it is exactly the same.
    This one assumes it is exactly the same in some cases such
    expand (X, sh1) = expand(X, sh2) if the output shapes are the same.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_y1(["y1 FLOAT(a, 2, 3, 4)"])
            I_sh1(["sh1 INT64(4)"])
            I_Y(["Y FLOAT(a, 1, 3, 4)"])

            Expand_0[["Expand(., .)"]]
            Expand_1[["Expand(., .)"]]

            I_Y -->|"FLOAT(a, 1, 3, 4)"| Expand_0
            I_sh1 -->|"INT64(4)"| Expand_0
            I_Y -->|"FLOAT(a, 1, 3, 4)"| Expand_1
            I_sh1 -->|"INT64(4)"| Expand_1

            O_y1(["y1 FLOAT(a, 2, 3, 4)"])
            Expand_0 --> O_y1
            O_y2(["y2 FLOAT(a, 2, 3, 4)"])
            Expand_1 --> O_y2

            class I_y1,I_sh1,I_Y,O_y1,O_y2 ioNode
            class Expand_0,Expand_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_y1(["y1 FLOAT(a, 2, 3, 4)"])
            I_sh1(["sh1 INT64(4)"])
            I_Y(["Y FLOAT(a, 1, 3, 4)"])

            Expand_0[["Expand(., .)"]]
            Identity_1[["Identity(.)"]]

            I_Y -->|"FLOAT(a, 1, 3, 4)"| Expand_0
            I_sh1 -->|"INT64(4)"| Expand_0
            Expand_0 -->|"FLOAT(a, 2, 3, 4)"| Identity_1

            O_y1(["y1 FLOAT(a, 2, 3, 4)"])
            Expand_0 --> O_y1
            O_y2(["y2 FLOAT(a, 2, 3, 4)"])
            Identity_1 --> O_y2

            class I_y1,I_sh1,I_Y,O_y1,O_y2 ioNode
            class Expand_0,Identity_1 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in {"Expand", "Reshape"} or node.domain != "":
            return self.none()
        next_nodes = g.next_nodes(node.input[0])
        if len(next_nodes) <= 1:
            return self.none()

        op_types = {n.op_type for n in next_nodes}

        for op_type in ["Expand", "Reshape"]:
            if op_type in op_types:
                selected = [
                    n
                    for n in next_nodes
                    if n.op_type == op_type
                    and n.input[0] == node.input[0]
                    and g.has_shape(n.output[0])
                ]
                if len(selected) < 2:
                    continue
                shapes = [g.get_shape(n.output[0]) for n in selected]
                if len(set(shapes)) == 1:
                    return MatchResult(self, selected, self.apply)
        return self.none(node, inspect.currentframe().f_lineno)

    def apply(self, g: "GraphBuilder", *nodes: NodeProto) -> List[NodeProto]:  # noqa: F821
        """
        The function receives multiple nodes of the same type.
        We keep one and replace the other by an Identity node.
        """
        new_nodes = [
            nodes[0],
            *[
                g.make_node(
                    "Identity",
                    [nodes[0].output[0]],
                    [n.output[0]],
                    name=f"{self.__class__.__name__}--{n.name}",
                )
                for n in nodes[1:]
            ],
        ]
        return new_nodes


class IdentityPattern(PatternOptimization):
    """
    Replaces operator such as
    Div(X, 1), Mul(X, 1), Add(X, 0), Sub(X, 0), Transpose(X, [0, 1, 2, ...]),
    Reshape(X, [0, 0, ...])
    by identity nodes. It looks into patterns involving the following operators:

    .. runpython::
        :showcode:

        from yobx.xoptim.patterns.onnx_any import IdentityPattern

        print(IdentityPattern.op_types)

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_x3(["x3 FLOAT(a, b, c)"])

            Transpose_0[["Transpose(., perm=[0, 1, 2])"]]

            I_x3 -->|"FLOAT(a, b, c)"| Transpose_0

            O_Y(["Y FLOAT(a, b, c)"])
            Transpose_0 --> O_Y

            class I_x3,O_Y ioNode
            class Transpose_0 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_x3(["x3 FLOAT(a, b, c)"])

            Identity_0[["Identity(.)"]]

            I_x3 -->|"FLOAT(a, b, c)"| Identity_0

            O_Y(["Y FLOAT(a, b, c)"])
            Identity_0 --> O_Y

            class I_x3,O_Y ioNode
            class Identity_0 opNode
    """

    op_types = {
        "Add",
        "Mul",
        "Div",
        "Sub",
        "Transpose",
        "Slice",
        "And",
        "Or",
        "Expand",
        "BatchNormalization",
        "Reshape",
    }

    @classmethod
    def _any_value_to_scalar(cls, cst):
        try:
            return float(cst)
        except TypeError:
            return complex(cst)

    @classmethod
    def _has_unique_value(cls, cst):
        return cst.min() == cst.max()

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in self.op_types or node.domain != "":
            return self.none()

        if node.op_type == "Slice":
            if len(node.input) == 5:
                steps = node.input[4]
                if (
                    not g.is_constant(steps)
                    or g.get_computed_constant(steps) is None
                    or set(g.get_computed_constant(steps)) != {1}
                ):
                    return self.none(node, inspect.currentframe().f_lineno)
            starts, ends = node.input[1:3]
            if (
                not g.is_constant(starts)
                or g.get_computed_constant(starts) is None
                or set(g.get_computed_constant(starts)) != {0}
            ):
                return self.none(node, inspect.currentframe().f_lineno)
            if (
                not g.is_constant(ends)
                or g.get_computed_constant(ends) is None
                or set(g.get_computed_constant(ends))
                != {9223372036854775807}  # this a value used by torch
            ):
                return self.none(node, inspect.currentframe().f_lineno)
            return MatchResult(self, [node], self.apply, insert_at=node)

        if node.op_type == "Reshape":
            if not g.is_constant(node.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            cst = g.get_computed_constant(node.input[1])
            if cst is None:
                return self.none(node, inspect.currentframe().f_lineno)
            cst = tuple(int(v) for v in cst)
            if len(cst) == 0 or set(cst) != {0}:
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.has_rank(node.input[0]) or g.get_rank(node.input[0]) != len(cst):
                return self.none(node, inspect.currentframe().f_lineno)
            return MatchResult(self, [node], self.apply, insert_at=node)

        if node.op_type == "Transpose":
            perm_attr = g.get_attribute(node, "perm", exc=False)
            if perm_attr is None:
                # perm is optional in ONNX; without it the op reverses all axes,
                # which is never a no-op for rank > 1, so we can safely skip.
                return self.none(node, inspect.currentframe().f_lineno)
            perm = list(perm_attr.ints)
            expected = list(range(len(perm)))
            if perm != expected:
                return self.none(node, inspect.currentframe().f_lineno)
            return MatchResult(self, [node], self.apply, insert_at=node)

        if node.op_type == "Expand":
            if not g.is_constant(node.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            value = g.get_computed_constant(node.input[1])
            if value is None:
                return self.none(node, inspect.currentframe().f_lineno)
            if not g.has_rank(node.input[0]) or value.shape[0] != g.get_rank(node.input[0]):
                return self.none(node, inspect.currentframe().f_lineno)
            with g.builder.maybe_disable_fake_tensor_mode():
                unique = set(value)
            if unique != {1}:
                return self.none(node, inspect.currentframe().f_lineno)
            return MatchResult(self, [node], self.apply, insert_at=node)

        if node.op_type == "BatchNormalization":
            if not g.has_type(node.input[0]):
                return self.none(node, inspect.currentframe().f_lineno)
            if g.get_type(node.input[0]) not in {TensorProto.FLOAT16, TensorProto.BFLOAT16}:
                return self.none(node, inspect.currentframe().f_lineno)
            training_mode = g.get_attribute_with_default(node, "training_mode", 0)
            epsilon = g.get_attribute_with_default(node, "epsilon", 1e-5)
            if training_mode != 0 or epsilon > 1e-5:
                return self.none(node, inspect.currentframe().f_lineno)
            if (
                not g.is_constant(node.input[1])
                or not g.is_constant(node.input[2])
                or not g.is_constant(node.input[3])
                or not g.is_constant(node.input[4])
            ):
                return self.none(node, inspect.currentframe().f_lineno)
            csts = [g.get_computed_constant(i) for i in node.input[1:]]
            if any(c is None for c in csts):
                return self.none(node, inspect.currentframe().f_lineno)
            with g.builder.maybe_disable_fake_tensor_mode():
                minis = [float(c.min()) for c in csts]
                maxis = [float(c.max()) for c in csts]
            if any(mi != ma for mi, ma in zip(minis, maxis)):
                return self.none(node, inspect.currentframe().f_lineno)
            if minis != [1, 0, 0, 1]:
                return self.none(node, inspect.currentframe().f_lineno)
            return MatchResult(self, [node], self.apply, insert_at=node)

        assert len(node.input) == 2, (
            f"Unexpected number of inputs {len(node.input)} "
            f"for node type {node.op_type!r} and name {node.name!r}, "
            f"node.input={node.input}"
        )
        if g.is_constant(node.input[1]):
            if g.has_rank(node.input[1]) and g.get_rank(node.input[1]) > 1:
                # No need to go further.
                return self.none(node, inspect.currentframe().f_lineno)
            shape = g.get_constant_shape(node.input[1], exc=False)
            if shape is None:
                return self.none(node, inspect.currentframe().f_lineno)
            if shape in (tuple(), (1,)):
                # simple case
                if not g.is_constant_scalar(node.input[1]):
                    return self.none(node, inspect.currentframe().f_lineno)
                cst = g.get_constant_scalar(node.input[1])
                val = self._any_value_to_scalar(cst)
                if val == 0:
                    if node.op_type in {"Add", "Sub", "Or"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val is False:
                    if node.op_type in {"Or"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val == 1:
                    if node.op_type in {"Mul", "Div", "And"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val is True:
                    if node.op_type in {"And"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
            elif len(shape) == 1 and node.op_type in {"Add", "Mul", "Sub", "Div", "And", "Or"}:
                # less simple case, the tensor is multiplied on its last dimension.
                cst = g.get_computed_constant(node.input[1])
                if cst is None:
                    return self.none(node, inspect.currentframe().f_lineno)
                if not self._has_unique_value(cst):
                    return self.none(node, inspect.currentframe().f_lineno)
                unique = cst[0]
                if not g.has_shape(node.input[0]):
                    return self.none(node, inspect.currentframe().f_lineno)
                shape = g.get_shape(node.input[0])
                if shape[-1] != cst.shape[0]:
                    return self.none(node, inspect.currentframe().f_lineno)
                if node.op_type in {"Add", "Sub", "And"} and unique != 0:
                    return self.none(node, inspect.currentframe().f_lineno)
                if node.op_type in {"And"} and unique is not True:
                    return self.none(node, inspect.currentframe().f_lineno)
                if node.op_type in {"Mul", "Div", "Or"} and unique != 1:
                    return self.none(node, inspect.currentframe().f_lineno)
                if node.op_type in {"Or"} and unique is not False:
                    return self.none(node, inspect.currentframe().f_lineno)
                return MatchResult(self, [node], self.apply, insert_at=node)
        elif g.is_constant(node.input[0]):
            if g.has_rank(node.input[0]) and g.get_rank(node.input[0]) > 1:
                # No need to go further.
                return self.none(node, inspect.currentframe().f_lineno)
            shape = g.get_constant_shape(node.input[0], exc=False)
            if shape is None:
                return self.none(node, inspect.currentframe().f_lineno)
            if shape in (tuple(), (1,)):
                # simple case
                if not g.is_constant_scalar(node.input[0]):
                    return self.none(node, inspect.currentframe().f_lineno)
                cst = g.get_constant_scalar(node.input[0])
                val = self._any_value_to_scalar(cst)
                if val == 0:
                    if node.op_type in {"Add", "Or"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val is False:
                    if node.op_type in {"Or"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val == 1:
                    if node.op_type in {"Mul", "And"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)
                elif val is True:
                    if node.op_type in {"And"}:
                        return MatchResult(self, [node], self.apply, insert_at=node)

        return self.none(node, inspect.currentframe().f_lineno)

    def apply(self, g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
        name = node.input[
            (
                1
                if node.op_type in {"Add", "Mul", "Div", "Sub", "And", "Or"}
                and g.is_constant(node.input[0])
                and g.has_shape(node.input[0])
                and g.get_shape(node.input[0]) in {(), (1,)}
                else 0
            )
        ]
        return [
            g.make_node(
                "Identity",
                [name],
                [node.output[0]],
                name=f"{self.__class__.__name__}--{node.name}.{node.op_type}",
            )
        ]


class ShapeBasedIdentityPattern(PatternOptimization):
    """
    If a slice leads to the same shape and the step is 1 then it is identity.
    In some cases, just known the same is enough to replace them.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a)"])

            Shape_0[["Shape(., end=1, start=0)"]]
            Slice_1[["Slice(., [0], ., [0])"]]

            I_X -->|"FLOAT(a)"| Shape_0
            I_X -->|"FLOAT(a)"| Slice_1
            Shape_0 --> Slice_1

            O_Y(["Y FLOAT(a)"])
            Slice_1 --> O_Y

            class I_X,O_Y ioNode
            class Shape_0,Slice_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a)"])

            Identity_0[["Identity(.)"]]

            I_X -->|"FLOAT(a)"| Identity_0

            O_Y(["Y FLOAT(a)"])
            Identity_0 --> O_Y

            class I_X,O_Y ioNode
            class Identity_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in {"Transpose", "Slice"} or node.domain != "":
            return self.none()

        if node.op_type == "Slice":
            if len(node.input) == 5:
                steps = node.input[4]
                if (
                    not g.is_constant(steps)
                    or g.get_computed_constant(steps) is None
                    or set(g.get_computed_constant(steps)) != {1}
                ):
                    return self.none(node, inspect.currentframe().f_lineno)

            if not g.has_shape(node.input[0]) or not g.has_shape(node.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)

            shape_i = g.get_shape_renamed(node.input[0])
            shape_o = g.get_shape_renamed(node.output[0])
            if shape_i == shape_o:
                return MatchResult(self, [node], self.apply, insert_at=node)

        return self.none(node, inspect.currentframe().f_lineno)

    def apply(self, g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
        return [
            g.make_node(
                "Identity",
                [node.input[0]],
                [node.output[0]],
                name=f"{self.__class__.__name__}--{node.name}",
            )
        ]


class SwapUnaryPattern(PatternOptimization):
    """
    Tries to move computation nodes before any transpose or reshape.
    That works for unary operator or equivalent to that.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b, c, d)"])
            I_cst(["cst FLOAT(1)"])

            Constant_0[["Constant() -#gt; cst"]]
            Transpose_1[["Transpose(., perm=[0, 2, 1, 3])"]]
            Mul_2[["Mul(., .)"]]

            I_X -->|"FLOAT(a, b, c, d)"| Transpose_1
            Transpose_1 -->|"FLOAT(a, c, b, d)"| Mul_2
            Constant_0 -->|"FLOAT(1)"| Mul_2

            O_Y(["Y FLOAT(a, c, b, d)"])
            Mul_2 --> O_Y

            class I_X,I_cst,O_Y ioNode
            class Constant_0 constNode
            class Transpose_1,Mul_2 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b, c, d)"])
            I_cst(["cst FLOAT(1)"])

            Mul_0[["Mul(., .)"]]
            Transpose_1[["Transpose(., perm=[0, 2, 1, 3])"]]

            I_X -->|"FLOAT(a, b, c, d)"| Mul_0
            I_cst -->|"FLOAT(1)"| Mul_0
            Mul_0 -->|"FLOAT(a, b, c, d)"| Transpose_1

            O_Y(["Y FLOAT(a, c, b, d)"])
            Transpose_1 --> O_Y

            class I_X,I_cst,O_Y ioNode
            class Mul_0,Transpose_1 opNode
    """

    _unary_types = unary_like_op_types()
    _binary_types_scalar_cst = {"Mul", "Add", "Div", "Sub"}

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            node.op_type not in {"Transpose", "Reshape", "Squeeze", "Unsqueeze"}
            or node.domain != ""
        ):
            return self.none()
        if not g.has_rank(node.input[0]) or g.get_rank(node.input[0]) == 0:
            # the unary or binary op changes the shape
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(node.output[0]) or node.op_type in {"Squeeze", "Unsqueeze"}:
            return self.none(node, inspect.currentframe().f_lineno)
        next_node = g.next_node(node.output[0])
        if next_node.op_type not in self._unary_types and (
            next_node.op_type not in self._binary_types_scalar_cst
            or not g.has_shape(next_node.input[1])
            or g.get_shape(next_node.input[1]) != (1,)
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply)

    def apply(
        self, g: "GraphBuilder", node: NodeProto, unary_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        temp_name = g.unique_name(f"{self.__class__.__name__}--{node.output[0]}")
        res = [
            g.make_node(
                unary_node.op_type,
                [node.input[0], *unary_node.input[1:]],
                [temp_name],
                name=f"{self.__class__.__name__}--{unary_node.name}",
            ),
            g.make_node(
                node.op_type,
                [temp_name, *node.input[1:]],
                [unary_node.output[0]],
                name=f"{self.__class__.__name__}--{node.name}",
            ),
        ]
        if unary_node.attribute:
            res[0].attribute.extend(unary_node.attribute)
        if node.attribute:
            res[1].attribute.extend(node.attribute)
        return res


class NotNotPattern(PatternOptimization):
    """
    Fuses Not + Not into Identity.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X BOOL(a, 2)"])

            Not_0[["Not(.)"]]
            Not_1[["Not(.)"]]

            I_X -->|"BOOL(a, 2)"| Not_0
            Not_0 -->|"BOOL(a, 2)"| Not_1

            O_Y(["Y BOOL(a, 2)"])
            Not_1 --> O_Y

            class I_X,O_Y ioNode
            class Not_0,Not_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X BOOL(a, 2)"])

            Identity_0[["Identity(.)"]]

            I_X -->|"BOOL(a, 2)"| Identity_0

            O_Y(["Y BOOL(a, 2)"])
            Identity_0 --> O_Y

            class I_X,O_Y ioNode
            class Identity_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Not" or node.domain != "":
            return self.none()

        not_before = g.node_before(node.input[0])
        if not not_before or not_before.op_type != "Not" or not_before.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [not_before, node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", not_before: NodeProto, not_after: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        pre_nodes = []
        if g.is_used_more_than_once(not_before.output[0]):
            pre_nodes.append(not_before)
        return [
            *pre_nodes,
            g.make_node(
                "Identity",
                [not_before.input[0]],
                [not_after.output[0]],
                name=f"{self.__class__.__name__}--{not_after.name}",
            ),
        ]
