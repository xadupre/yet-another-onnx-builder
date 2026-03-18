import inspect
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
from onnx import NodeProto
from ...helpers.onnx_helper import (
    element_wise_binary_op_types,
    element_wise_op_cmp_types,
    unary_like_op_types,
)
from ...xshape._shape_helper import all_int, DYNAMIC_SHAPE
from ..patterns_api import MatchResult, PatternOptimization


class ExpandPattern(PatternOptimization):
    """
    Checks that a Expand is really needed.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_init7_s4_32_2_10_8(["init7_s4_32_2_10_8 INT64(4)"])
            I_mul(["mul FLOAT(32, 2, 10, 8)"])

            Constant_0[["Constant() -#gt; init7_s4_32_2_10_8"]]
            Expand_1[["Expand(., .)"]]

            I_mul -->|"FLOAT(32, 2, 10, 8)"| Expand_1
            Constant_0 -->|"INT64(4)"| Expand_1

            O_expand(["expand FLOAT(32, 2, 10, 8)"])
            Expand_1 --> O_expand

            class I_init7_s4_32_2_10_8,I_mul,O_expand ioNode
            class Constant_0 constNode
            class Expand_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_init7_s4_32_2_10_8(["init7_s4_32_2_10_8 INT64(4)"])
            I_mul(["mul FLOAT(32, 2, 10, 8)"])

            Identity_0[["Identity(., .)"]]

            I_mul -->|"FLOAT(32, 2, 10, 8)"| Identity_0
            I_init7_s4_32_2_10_8 -->|"INT64(4)"| Identity_0

            O_expand(["expand FLOAT(32, 2, 10, 8)"])
            Identity_0 --> O_expand

            class I_init7_s4_32_2_10_8,I_mul,O_expand ioNode
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
        if node.op_type != "Expand" or node.domain != "":
            return self.none()
        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape = g.get_shape(node.input[0])
        if not all_int(shape):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[1]):
            # It may be a symbolic shape.
            return self.none(node, inspect.currentframe().f_lineno)
        value = g.get_computed_constant(node.input[1])
        if value is None:
            return self.none(node, inspect.currentframe().f_lineno)
        with g.builder.maybe_disable_fake_tensor_mode():
            new_shape = tuple(int(i) for i in value)
        if shape != new_shape:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(self, g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
        new_node = g.make_node(
            "Identity",
            node.input,
            node.output,
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [new_node]


class ExpandBroadcastPattern(PatternOptimization):
    """
    Checks that a Expand is really needed before an element wise operator.
    The objective is to save one allocation and let the next operator
    do the expansion by broadcasting one input.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_mul_25(["mul_25 FLOAT(2, 1024, 1)"])
            I_input66(["input66 FLOAT(2, 1024, 1024)"])

            Expand_0[["Expand(., [2, 1024, 1024])"]]
            Mul_1[["Mul(., .)"]]

            I_mul_25 -->|"FLOAT(2, 1024, 1)"| Expand_0
            Expand_0 -->|"FLOAT(2, 1024, 1024)"| Mul_1
            I_input66 -->|"FLOAT(2, 1024, 1024)"| Mul_1

            O_MulMulMulPattern__mul_27(["MulMulMulPattern--mul_27 FLOAT(2, 1024, 1024)"])
            Mul_1 --> O_MulMulMulPattern__mul_27

            class I_mul_25,I_input66,O_MulMulMulPattern__mul_27 ioNode
            class Expand_0,Mul_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_mul_25(["mul_25 FLOAT(2, 1024, 1)"])
            I_input66(["input66 FLOAT(2, 1024, 1024)"])

            Mul_0[["Mul(., .)"]]

            I_mul_25 -->|"FLOAT(2, 1024, 1)"| Mul_0
            I_input66 -->|"FLOAT(2, 1024, 1024)"| Mul_0

            O_MulMulMulPattern__mul_27(["MulMulMulPattern--mul_27 FLOAT(2, 1024, 1024)"])
            Mul_0 --> O_MulMulMulPattern__mul_27

            class I_mul_25,I_input66,O_MulMulMulPattern__mul_27 ioNode
            class Mul_0 opNode
    """

    _op_types = element_wise_binary_op_types() | element_wise_op_cmp_types()

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return self.none()
        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape = g.get_shape(node.input[0])
        if not all_int(shape):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[1]):
            # It may be a symbolic shape.
            return self.none(node, inspect.currentframe().f_lineno)
        value = g.get_computed_constant(node.input[1])
        if value is None:
            return self.none(node, inspect.currentframe().f_lineno)
        with g.builder.maybe_disable_fake_tensor_mode():
            new_shape = tuple(int(i) for i in value)

        if g.is_used_more_than_once(node.output[0]):
            # More than one output, not handled right now.
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        assert len(next_nodes) == 1, "The previous test should have cleared out this case."
        next_node = next_nodes[0]

        if next_node.op_type not in self._op_types or next_node.domain != "":
            # Not an element wise operator.
            return self.none(node, inspect.currentframe().f_lineno)

        other = next_node.input[1 if next_node.input[0] == node.output[0] else 0]

        if not g.has_shape(other):
            return self.none(node, inspect.currentframe().f_lineno)

        other_shape = g.get_shape(other)
        if new_shape != other_shape:
            # Expand does not expand to the shape of the other element.
            return self.none(node, inspect.currentframe().f_lineno)
        if len(shape) != len(other_shape):
            # Different ranks.
            return self.none(node, inspect.currentframe().f_lineno)
        for a, b in zip(shape, other_shape):
            if not (a == b or a == 1 or b == 1):
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply, insert_at=next_node)

    def apply(
        self, g: "GraphBuilder", node: NodeProto, next_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        if next_node.input[0] == node.output[0]:
            inputs = [node.input[0], next_node.input[1]]
        else:
            inputs = [next_node.input[0], node.input[0]]
        return [
            g.make_node(
                next_node.op_type,
                inputs,
                next_node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=next_node.doc_string,
            )
        ]


class ShapeBasedExpandBroadcastPattern(PatternOptimization):
    """
    Similar to
    :class:`yobx.xoptim.patterns.onnx_expand.ExpandBroadcastPattern`,
    but it allows dynamic shapes as well. It does not look into the second
    argument of Expand, it just infers than an expand is not needed for
    a binary operator following just after.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(1, b, c)"])
            I_Y(["Y FLOAT(a, b, c)"])

            Expand_0[["Expand(., .)"]]
            Add_1[["Add(., .)"]]

            I_X -->|"FLOAT(1, b, c)"| Expand_0
            Expand_0 -->|"FLOAT(a, b, c)"| Add_1
            I_Y -->|"FLOAT(a, b, c)"| Add_1

            O_Z(["Z FLOAT(a, b, c)"])
            Add_1 --> O_Z

            class I_X,I_Y,O_Z ioNode
            class Expand_0,Add_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(1, b, c)"])
            I_Y(["Y FLOAT(a, b, c)"])

            Add_0[["Add(., .)"]]

            I_X -->|"FLOAT(1, b, c)"| Add_0
            I_Y -->|"FLOAT(a, b, c)"| Add_0

            O_Z(["Z FLOAT(a, b, c)"])
            Add_0 --> O_Z

            class I_X,I_Y,O_Z ioNode
            class Add_0 opNode
    """

    _op_types = element_wise_binary_op_types() | element_wise_op_cmp_types()

    @classmethod
    def _is_compatible_shapes_for_expand(
        cls,
        shape_left: DYNAMIC_SHAPE,
        shape_right: DYNAMIC_SHAPE,
        output_shape: Optional[DYNAMIC_SHAPE],
    ) -> bool:
        """
        Checks that the binary operations of the two input shapes returns the output_shape.
        Then no Expand node is needed.
        """
        if output_shape is None:
            return False
        if max(len(shape_left), len(shape_right) if shape_right else 0) < len(output_shape):
            return False
        # Align shapes
        if len(shape_left) < len(shape_right):
            shape_left = (1,) * (len(shape_right) - len(shape_left)) + shape_left
        elif len(shape_left) > len(shape_right):
            shape_right = (1,) * (len(shape_left) - len(shape_right)) + shape_right

        for left, right, out in zip(shape_left, shape_right, output_shape):
            if isinstance(left, int):
                if isinstance(right, int):
                    # static right
                    if left == 1:
                        if right != out:
                            return False
                    elif right == 1:
                        if left != out:
                            return False
                    else:
                        if left != right or left != out or right != out:
                            return False
                else:
                    # dynamic right
                    if left == 1:
                        if right != out:
                            return False
                    else:
                        if left != right or left != out or right != out:
                            return False
            else:
                # dynamic left
                if isinstance(right, int):
                    # static right
                    if right == 1:
                        if left != out:
                            return False
                    else:
                        if left != right or left != out or right != out:
                            return False
                else:
                    # dynamic right
                    if left != right or left != out or right != out:
                        return False
        return True

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in self._op_types or node.domain != "":
            return self.none()
        if (
            not g.has_shape(node.output[0])
            or not g.has_shape(node.input[0])
            or not g.has_shape(node.input[1])
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        node_left = g.node_before(node.input[0])
        node_right = g.node_before(node.input[1])
        before = [
            None if n is None or n.op_type != "Expand" else n for n in [node_left, node_right]
        ]
        if before == [None, None]:
            return self.none(node, inspect.currentframe().f_lineno)

        # At least one expand.
        node_left, node_right = before
        shape_left = g.get_shape_renamed(
            node.input[0] if node_left is None else node_left.input[0]
        )
        shape_right = g.get_shape_renamed(
            node.input[1] if node_right is None else node_right.input[0]
        )
        if self._is_compatible_shapes_for_expand(
            shape_left, shape_right, g.get_shape_renamed(node.output[0])
        ):
            if self.verbose:
                print(
                    f"[{self.__class__.__name__}.match] {shape_left} "
                    f"{node.op_type} {shape_right} -> {g.get_shape_renamed(node.output[0])}"
                )
            return MatchResult(self, [node_left, node_right, node], self.apply)
        # We could end up with the following case.
        # shape_left   = (1, 1, 'seq_length', 'cache_length + seq_length')
        # shape_right  = (1, 1, 'seq_length', 'cache_length + seq_length')
        # output_shape = ('batch', 1, 'seq_length', 'cache_length + seq_length')
        # When this happes, it could also be caught by another pattern.
        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        expand_left: NodeProto,
        expand_right: NodeProto,
        binary_node: NodeProto,
    ) -> List[NodeProto]:
        nodes = []
        if expand_left is not None and g.is_used_more_than_once(expand_left.output[0]):
            nodes.append(expand_left)
        if expand_right is not None and g.is_used_more_than_once(expand_right.output[0]):
            nodes.append(expand_right)
        assert (
            not binary_node.attribute
        ), f"Binary operator should not have any attribute, binary_node={binary_node}"
        return [
            *nodes,
            g.make_node(
                binary_node.op_type,
                [
                    binary_node.input[0] if expand_left is None else expand_left.input[0],
                    binary_node.input[1] if expand_right is None else expand_right.input[0],
                ],
                binary_node.output,
                name=f"{self.__class__.__name__}--{binary_node.name}",
                doc_string=binary_node.doc_string,
            ),
        ]


class ExpandSwapPattern(PatternOptimization):
    """
    Tries to move a node Expand forward in the graph.
    Expand + Exp can be changed into Exp + Expand.
    Then Exp applies on a tensor of a smaller or equal size.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_p(["p INT64(1)"])
            I_X(["X FLOAT(1, 5, 7)"])
            I_shape(["shape INT64(3)"])

            Constant_0[["Constant() -#gt; shape"]]
            Constant_1[["Constant() -#gt; p"]]
            Expand_2[["Expand(., .)"]]
            Pow_3[["Pow(., .)"]]

            I_X -->|"FLOAT(1, 5, 7)"| Expand_2
            Constant_0 -->|"INT64(3)"| Expand_2
            Expand_2 -->|"FLOAT(3, 5, 7)"| Pow_3
            Constant_1 -->|"INT64(1)"| Pow_3

            O_Z(["Z FLOAT(3, 5, 7)"])
            Pow_3 --> O_Z

            class I_p,I_X,I_shape,O_Z ioNode
            class Constant_0,Constant_1 constNode
            class Expand_2,Pow_3 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_p(["p INT64(1)"])
            I_X(["X FLOAT(1, 5, 7)"])
            I_shape(["shape INT64(3)"])

            Pow_0[["Pow(., .)"]]
            Expand_1[["Expand(., .)"]]

            I_X -->|"FLOAT(1, 5, 7)"| Pow_0
            I_p -->|"INT64(1)"| Pow_0
            Pow_0 -->|"FLOAT(1, 5, 7)"| Expand_1
            I_shape -->|"INT64(3)"| Expand_1

            O_Z(["Z FLOAT(3, 5, 7)"])
            Expand_1 --> O_Z

            class I_p,I_X,I_shape,O_Z ioNode
            class Pow_0,Expand_1 opNode
    """

    _op_types = unary_like_op_types()
    _other_types = {"NegXplus1", "ReplaceZero", "Pow"}

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return self.none()
        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        assert g.is_used(node.output[0]), (
            f"The match should not even begin, {node.output[0]!r} "
            f"is not used among {node.output} and type={node.op_type!r}"
        )
        if g.is_used_more_than_once(node.output[0]):
            # More than one output so it probably must be done.
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        assert len(next_nodes) == 1, "The previous test should have cleared out this case."
        next_node = next_nodes[0]

        if next_node.op_type not in self._other_types and (
            next_node.op_type not in self._op_types or next_node.domain != ""
        ):
            # Not an unary wise operator.
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, next_node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", node: NodeProto, next_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        # We need to create a new name for the intermediate results.
        # The optimizer cannot reuse an existing name if the new result
        # has a different shape.
        new_name = g.unique_name(f"{self.__class__.__name__}_{node.input[0]}")
        unary = g.make_node(
            next_node.op_type,
            [node.input[0], *next_node.input[1:]],
            [new_name],
            name=f"{self.__class__.__name__}--{node.name}",
            domain=next_node.domain,
            doc_string=next_node.doc_string,
        )
        unary.attribute.extend(next_node.attribute)
        expand = g.make_node(
            node.op_type,  # Expand
            [new_name, node.input[1]],
            [next_node.output[0]],
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        return [unary, expand]


class ShapeBasedStaticExpandPattern(PatternOptimization):
    """
    Compares input and output shapes to tell if the expand
    can uses a constant as a second input.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(2, 3, d, 1)"])

            Shape_0[["Shape(., end=-1, start=0)"]]
            Concat_1[["Concat(., [2], axis=0)"]]
            Expand_2[["Expand(., .)"]]

            I_X -->|"FLOAT(2, 3, d, 1)"| Shape_0
            Shape_0 -->|"INT64(3)"| Concat_1
            I_X -->|"FLOAT(2, 3, d, 1)"| Expand_2
            Concat_1 -->|"INT64(4)"| Expand_2

            O_Y(["Y FLOAT(2, 3, d, 2)"])
            Expand_2 --> O_Y

            class I_X,O_Y ioNode
            class Shape_0,Concat_1,Expand_2 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(2, 3, d, 1)"])

            Expand_0[["Expand(., [1, 1, 1, 2])"]]

            I_X -->|"FLOAT(2, 3, d, 1)"| Expand_0

            O_Y(["Y FLOAT(2, 3, d, 2)"])
            Expand_0 --> O_Y

            class I_X,O_Y ioNode
            class Expand_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    @classmethod
    def _find_expand_shape(
        cls, sh1: Tuple[Union[str, int], ...], sh2: Tuple[Union[str, int], ...]
    ) -> Tuple[int, ...]:
        expand_shape = []
        for s1, s2 in zip(sh1, sh2):
            if s1 == s2:
                expand_shape.append(1)
                continue
            if not isinstance(s1, int) or not isinstance(s2, int):
                return None
            expand_shape.append(s2)
        return tuple(expand_shape)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return self.none()
        if g.is_constant(node.input[1]):
            # already done
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        sh1 = g.get_shape_renamed(node.input[0])
        sh2 = g.get_shape_renamed(node.output[0])
        if len(sh1) != len(sh2):
            # We ignore that case for the time being.
            return self.none(node, inspect.currentframe().f_lineno)
        expand_shape = self._find_expand_shape(sh1, sh2)
        if expand_shape is None:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(self, g: "GraphBuilder", reshape: NodeProto) -> List[NodeProto]:  # noqa: F821
        expand_shape = self._find_expand_shape(
            g.get_shape_renamed(reshape.input[0]), g.get_shape_renamed(reshape.output[0])
        )
        new_shape = g.make_initializer(
            "", np.array(expand_shape, dtype=np.int64), source=f"{self.__class__.__name__}.m1"
        )
        return [
            g.make_node(
                "Expand",
                [reshape.input[0], new_shape],
                reshape.output,
                name=f"{self.__class__.__name__}--{reshape.name}",
                doc_string=reshape.doc_string,
            )
        ]


class ShapeBasedExpandSwapPattern(PatternOptimization):
    """
    Tries to move a node Expand forward in the graph
    for a binary operator. The code is similar to
    :class:`yobx.xoptim.patterns.onnx_expand.ShapeBasedExpandBroadcastPattern`

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_full_shape(["full_shape INT64(2)"])
            I_Xc(["Xc FLOAT(d, 1)"])
            I_one(["one FLOAT(1)"])

            Constant_0[["Constant() -#gt; one"]]
            Expand_1[["Expand(., .)"]]
            Add_2[["Add(., .)"]]

            I_Xc -->|"FLOAT(d, 1)"| Expand_1
            I_full_shape -->|"INT64(2)"| Expand_1
            Expand_1 --> Add_2
            Constant_0 -->|"FLOAT(1)"| Add_2

            O_Y(["Y FLOAT(d, d)"])
            Add_2 --> O_Y

            class I_full_shape,I_Xc,I_one,O_Y ioNode
            class Constant_0 constNode
            class Expand_1,Add_2 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_full_shape(["full_shape INT64(2)"])
            I_Xc(["Xc FLOAT(d, 1)"])
            I_one(["one FLOAT(1)"])

            Add_0[["Add(., .)"]]
            Expand_1[["Expand(., .)"]]

            I_Xc -->|"FLOAT(d, 1)"| Add_0
            I_one -->|"FLOAT(1)"| Add_0
            Add_0 -->|"FLOAT(d, 1)"| Expand_1
            I_full_shape -->|"INT64(2)"| Expand_1

            O_Y(["Y FLOAT(d, d)"])
            Expand_1 --> O_Y

            class I_full_shape,I_Xc,I_one,O_Y ioNode
            class Add_0,Expand_1 opNode
    """

    _op_types = element_wise_binary_op_types() | element_wise_op_cmp_types()

    @classmethod
    def _broadcast_shape(
        cls,
        before_expand_shape: DYNAMIC_SHAPE,
        other_term_shape: DYNAMIC_SHAPE,
        exc: bool = False,
    ) -> Optional[DYNAMIC_SHAPE]:
        if len(before_expand_shape) != len(other_term_shape):
            d = abs(len(before_expand_shape) - len(other_term_shape))
            if len(before_expand_shape) < len(other_term_shape):
                before_expand_shape = (1,) * d + before_expand_shape
            else:
                other_term_shape = (1,) * d + other_term_shape
        if len(before_expand_shape) != len(other_term_shape):
            assert not exc, (
                f"Unable to produce a broadcasted shape from "
                f"{before_expand_shape} and {other_term_shape}"
            )
            return None
        res = []
        for a, b in zip(before_expand_shape, other_term_shape):
            if a == b:
                res.append(a)
            elif a == 1:
                res.append(b)
            elif b == 1:
                res.append(a)
            else:
                assert not exc, (
                    f"Unable to produce a broadcasted shape from "
                    f"{before_expand_shape} and {other_term_shape}"
                )
                return None
        return tuple(res)

    @classmethod
    def _get_compatible_expand_shape_for_expand_swap(
        cls,
        before_expand_shape: DYNAMIC_SHAPE,
        expanded_shape: DYNAMIC_SHAPE,
        other_term_shape: DYNAMIC_SHAPE,
        other_expanded_shape: Optional[DYNAMIC_SHAPE],
        output_shape: DYNAMIC_SHAPE,
    ) -> Optional[DYNAMIC_SHAPE]:
        """
        Something like that should work.
        The function returns a shape or None is not possible.

        .. code-block:: python

            _get_compatible_expand_shape_for_expand_swap(
                ("batch", 1, 1, 1),
                ("batch", 1, "seq_length", "cache_length+seq_length"),
                (1,),
                None,
                ("batch", 1, "seq_length", "cache_length+seq_length"),
            )

            >>> ("batch", 1, "seq_length", "cache_length+seq_length")
        )

        """
        if other_expanded_shape is not None and (
            other_expanded_shape != expanded_shape
            or expanded_shape != output_shape
            or len(before_expand_shape) != len(other_term_shape)
        ):
            return None
        if before_expand_shape == expanded_shape or expanded_shape == other_term_shape:
            # This pattern is not meant for that.
            return None
        if output_shape != expanded_shape:
            return None
        if (
            other_expanded_shape is None
            and not ShapeBasedExpandBroadcastPattern._is_compatible_shapes_for_expand(
                before_expand_shape,
                other_term_shape,
                cls._broadcast_shape(before_expand_shape, other_term_shape, exc=False),
            )
        ):
            return None
        if (
            other_expanded_shape is not None
            and not ShapeBasedExpandBroadcastPattern._is_compatible_shapes_for_expand(
                before_expand_shape,
                other_term_shape,
                cls._broadcast_shape(before_expand_shape, other_term_shape, exc=False),
            )
        ):
            return None
        if other_expanded_shape is None:
            return "expand_arg"
        max_dim = cls._broadcast_shape(before_expand_shape, other_term_shape)
        if max_dim == output_shape:
            # Expand is not necessary at all.
            return None
        return tuple(1 if a == b else 0 for a, b in zip(max_dim, output_shape))

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in self._op_types or node.domain != "":
            return self.none()
        if (
            not g.has_shape(node.output[0])
            or not g.has_shape(node.input[0])
            or not g.has_shape(node.input[1])
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        node_left = g.node_before(node.input[0])
        node_right = g.node_before(node.input[1])
        before = [
            None if n is None or n.op_type != "Expand" else n for n in [node_left, node_right]
        ]
        if before == [None, None]:
            return self.none(node, inspect.currentframe().f_lineno)

        if None in before:
            # Only one expand
            node_left, node_right = before
            shape_left = g.get_shape_renamed(
                node.input[0] if node_left is None else node_left.input[0]
            )
            shape_right = g.get_shape_renamed(
                node.input[1] if node_right is None else node_right.input[0]
            )
            before_expand_shape = shape_right if node_left is None else shape_left
            expanded_shape = (
                g.get_shape_renamed(node_right.output[0])
                if node_left is None
                else g.get_shape_renamed(node_left.output[0])
            )
            other_term_shape = shape_left if node_left is None else shape_right
            output_shape = g.get_shape_renamed(node.output[0])
            if self._get_compatible_expand_shape_for_expand_swap(
                before_expand_shape, expanded_shape, other_term_shape, None, output_shape
            ):
                if self.verbose:
                    print(
                        f"[{self.__class__.__name__}.match.1] {shape_left} "
                        f"{node.op_type} {shape_right} -> {output_shape}"
                    )
                return MatchResult(self, [node_left, node_right, node], self.apply)
            return self.none(node, inspect.currentframe().f_lineno)

        # Both expand.
        node_left, node_right = before
        if node_left.input[1] != node_right.input[1]:
            # It could work in that case if both expand have different
            # shape argument but the code to make sure it is is not implemented.
            return self.none(node, inspect.currentframe().f_lineno)

        shape_left = g.get_shape_renamed(node_left.input[0])
        shape_right = g.get_shape_renamed(node_right.input[0])
        output_shape = g.get_shape_renamed(node.output[0])
        expand_arg = self._get_compatible_expand_shape_for_expand_swap(
            shape_left,
            g.get_shape_renamed(node.input[0]),
            shape_right,
            g.get_shape_renamed(node.input[1]),
            output_shape,
        )
        if expand_arg:
            if self.verbose:
                print(
                    f"[{self.__class__.__name__}.match.2] {shape_left} "
                    f"{node.op_type} {shape_right} -> {output_shape} with "
                    f"expand_arg={expand_arg}"
                )
            return MatchResult(self, [node_left, node_right, node], self.apply)

        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        expand_left: NodeProto,
        expand_right: NodeProto,
        binary_node: NodeProto,
    ) -> List[NodeProto]:
        nodes = []
        if expand_left is not None and g.is_used_more_than_once(expand_left.output[0]):
            nodes.append(expand_left)
        if expand_right is not None and g.is_used_more_than_once(expand_right.output[0]):
            nodes.append(expand_right)
        assert (
            not binary_node.attribute
        ), f"Binary operator should not have any attribute, binary_node={binary_node}"
        new_name = g.unique_name(f"{self.__class__.__name__}_{binary_node.output[0]}")
        nodes.append(
            g.make_node(
                binary_node.op_type,
                [
                    binary_node.input[0] if expand_left is None else expand_left.input[0],
                    binary_node.input[1] if expand_right is None else expand_right.input[0],
                ],
                [new_name],
                name=f"{self.__class__.__name__}--{binary_node.name}",
                doc_string=binary_node.doc_string,
            )
        )

        # One or two expand, same rewriting as the expand argument is the same.
        return [
            *nodes,
            g.make_node(
                "Expand",
                [
                    new_name,
                    expand_left.input[1] if expand_right is None else expand_right.input[1],
                ],
                binary_node.output,
                name=f"{self.__class__.__name__}--{binary_node.name}",
                doc_string=binary_node.doc_string,
            ),
        ]


class ShapeBasedExpandBroadcastMatMulPattern(PatternOptimization):
    """
    Similar to
    :class:`yobx.xoptim.patterns.onnx_expand.ShapeBasedExpandBroadcastPattern`,
    but works only with MatMul.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_Y(["Y FLOAT(1, c, d)"])
            I_X(["X FLOAT(a, b, c)"])

            Shape_0[["Shape(., end=1, start=0)"]]
            Concat_1[["Concat(., [1, 1], axis=0)"]]
            Expand_2[["Expand(., .)"]]
            MatMul_3[["MatMul(., .)"]]

            I_Y -->|"FLOAT(1, c, d)"| Shape_0
            Shape_0 -->|"INT64(1)"| Concat_1
            I_Y -->|"FLOAT(1, c, d)"| Expand_2
            Concat_1 -->|"INT64(3)"| Expand_2
            I_X -->|"FLOAT(a, b, c)"| MatMul_3
            Expand_2 -->|"FLOAT(1, c, d)"| MatMul_3

            O_Z(["Z FLOAT(a, b, d)"])
            MatMul_3 --> O_Z

            class I_Y,I_X,O_Z ioNode
            class Shape_0,Concat_1,Expand_2,MatMul_3 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_Y(["Y FLOAT(1, c, d)"])
            I_X(["X FLOAT(a, b, c)"])

            MatMul_0[["MatMul(., .)"]]

            I_X -->|"FLOAT(a, b, c)"| MatMul_0
            I_Y -->|"FLOAT(1, c, d)"| MatMul_0

            O_Z(["Z FLOAT(a, b, d)"])
            MatMul_0 --> O_Z

            class I_Y,I_X,O_Z ioNode
            class MatMul_0 opNode
    """

    @classmethod
    def _is_compatible_shapes_for_expand(
        cls,
        shape_left: DYNAMIC_SHAPE,
        shape_right: DYNAMIC_SHAPE,
        output_shape: Optional[DYNAMIC_SHAPE],
    ) -> bool:
        """
        Checks that the binary operations of the two input shapes returns the output_shape.
        Then no Expand node is needed.
        """
        if output_shape is None:
            return False
        if len(shape_left) < 2 or len(shape_right) < 2 or len(output_shape) < 2:
            return False
        return ShapeBasedExpandBroadcastPattern._is_compatible_shapes_for_expand(
            shape_left[:-2], shape_right[:-2], output_shape[:-2]
        )

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "MatMul" or node.domain != "":
            return self.none()
        if not g.has_shape(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        node_left = g.node_before(node.input[0])
        node_right = g.node_before(node.input[1])
        before = [
            None if n is None or n.op_type != "Expand" else n for n in [node_left, node_right]
        ]
        if before == [None, None]:
            return self.none(node, inspect.currentframe().f_lineno)

        # At least one expand.
        node_left, node_right = before
        shape_left = g.get_shape_renamed(
            node.input[0] if node_left is None else node_left.input[0]
        )
        shape_right = g.get_shape_renamed(
            node.input[1] if node_right is None else node_right.input[0]
        )
        if self._is_compatible_shapes_for_expand(
            shape_left, shape_right, g.get_shape_renamed(node.output[0])
        ):
            if self.verbose:
                print(
                    f"[{self.__class__.__name__}.match] {shape_left} "
                    f"{node.op_type} {shape_right} -> {g.get_shape_renamed(node.output[0])}"
                )
            return MatchResult(self, [node_left, node_right, node], self.apply)
        # We could end up with the following case.
        # shape_left   = (1, 1, 'seq_length', 'cache_length + seq_length')
        # shape_right  = (1, 1, 'seq_length', 'cache_length + seq_length')
        # output_shape = ('batch', 1, 'seq_length', 'cache_length + seq_length')
        # When this happes, it could also be caught by another pattern.
        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        expand_left: NodeProto,
        expand_right: NodeProto,
        binary_node: NodeProto,
    ) -> List[NodeProto]:
        nodes = []
        if expand_left is not None and g.is_used_more_than_once(expand_left.output[0]):
            nodes.append(expand_left)
        if expand_right is not None and g.is_used_more_than_once(expand_right.output[0]):
            nodes.append(expand_right)
        assert (
            not binary_node.attribute
        ), f"Binary operator should not have any attribute, binary_node={binary_node}"
        return [
            *nodes,
            g.make_node(
                binary_node.op_type,
                [
                    binary_node.input[0] if expand_left is None else expand_left.input[0],
                    binary_node.input[1] if expand_right is None else expand_right.input[0],
                ],
                binary_node.output,
                name=f"{self.__class__.__name__}--{binary_node.name}",
                doc_string=binary_node.doc_string,
            ),
        ]


class ShapeBasedExpandCastWhereSwapPattern(PatternOptimization):
    """
    Rewrites Where(Cast(X), X, cond).

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(b, c)"])
            I_exp(["exp INT64(3)"])
            I_cst(["cst FLOAT(1)"])

            Constant_0[["Constant() -#gt; cst"]]
            Expand_1[["Expand(., .)"]]
            Cast_2[["Cast(., to=BOOL)"]]
            Where_3[["Where(., ., .)"]]

            I_X -->|"FLOAT(b, c)"| Expand_1
            I_exp -->|"INT64(3)"| Expand_1
            Expand_1 --> Cast_2
            Cast_2 --> Where_3
            Expand_1 --> Where_3
            Constant_0 -->|"FLOAT(1)"| Where_3

            O_Y(["Y FLOAT(b, b, c)"])
            Where_3 --> O_Y

            class I_X,I_exp,I_cst,O_Y ioNode
            class Constant_0 constNode
            class Expand_1,Cast_2,Where_3 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(b, c)"])
            I_exp(["exp INT64(3)"])
            I_cst(["cst FLOAT(1)"])

            Cast_0[["Cast(., to=BOOL)"]]
            Where_1[["Where(., ., .)"]]
            Expand_2[["Expand(., .)"]]

            I_X -->|"FLOAT(b, c)"| Cast_0
            Cast_0 -->|"BOOL(b, c)"| Where_1
            I_X -->|"FLOAT(b, c)"| Where_1
            I_cst -->|"FLOAT(1)"| Where_1
            Where_1 -->|"FLOAT(b, c)"| Expand_2
            I_exp -->|"INT64(3)"| Expand_2

            O_Y(["Y FLOAT(b, b, c)"])
            Expand_2 --> O_Y

            class I_X,I_exp,I_cst,O_Y ioNode
            class Cast_0,Where_1,Expand_2 opNode
    """

    @classmethod
    def _compatible_shapes(
        cls, cond: DYNAMIC_SHAPE, cst: DYNAMIC_SHAPE, output: DYNAMIC_SHAPE, before: DYNAMIC_SHAPE
    ):
        if cond != output:
            return False
        if len(before) < len(output):
            before = (1,) * (len(output) - len(before)) + before
        if len(cst) < len(output):
            cst = (1,) * (len(output) - len(cst)) + cst
        out = ShapeBasedExpandSwapPattern._broadcast_shape(before, cst)
        if len(out) != len(output) or len(out) != len(before):
            return False
        return all(not (o != e and o != b) for b, o, e in zip(before, out, output))

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Where" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.input[0]):
            return self.none()
        cast_node = g.node_before(node.input[0])
        if cast_node is None or cast_node.op_type != "Cast" or cast_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if cast_node.input[0] not in node.input[1:]:
            return self.none(node, inspect.currentframe().f_lineno)
        expand_node = g.node_before(cast_node.input[0])
        if expand_node is None or expand_node.op_type != "Expand" or expand_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        nodes = g.next_nodes(cast_node.input[0])
        if len(nodes) != 2:
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.has_shape(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[2]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(expand_node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        same_index = list(node.input).index(cast_node.input[0])
        if self._compatible_shapes(
            g.get_shape_renamed(node.input[0]),
            g.get_shape_renamed(node.input[3 - same_index]),
            g.get_shape_renamed(node.output[0]),
            g.get_shape_renamed(expand_node.input[0]),
        ):
            return MatchResult(self, [expand_node, cast_node, node], self.apply)
        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        expand_node: NodeProto,
        cast_node: NodeProto,
        where_node: NodeProto,
    ) -> List[NodeProto]:
        to = g.get_attribute(cast_node, "to").i
        pos_index = list(where_node.input).index(expand_node.output[0])
        cast_output = g.unique_name(f"{self.__class__.__name__}_{cast_node.output[0]}")
        where_output = g.unique_name(f"{self.__class__.__name__}_{where_node.output[0]}")
        return [
            g.make_node(
                cast_node.op_type,
                [expand_node.input[0]],
                [cast_output],
                to=to,
                name=f"{self.__class__.__name__}--{cast_node.name}",
                doc_string=cast_node.doc_string,
            ),
            g.make_node(
                where_node.op_type,
                (
                    [cast_output, expand_node.input[0], where_node.input[2]]
                    if pos_index == 1
                    else [cast_output, where_node.input[1], expand_node.input[0]]
                ),
                [where_output],
                name=f"{self.__class__.__name__}--{where_node.name}",
                doc_string=where_node.doc_string,
            ),
            g.make_node(
                expand_node.op_type,
                [where_output, expand_node.input[1]],
                [where_node.output[0]],
                name=f"{self.__class__.__name__}--{expand_node.name}",
                doc_string=expand_node.doc_string,
            ),
        ]


class ShapeBasedConcatExpandPattern(PatternOptimization):
    """
    Rewrites Expand(X, concat(...)) if possible.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, 1)"])
            I_two(["two INT64(1)"])

            Constant_0[["Constant() -#gt; two"]]
            Shape_1[["Shape(., end=1, start=0)"]]
            Concat_2[["Concat(., ., axis=0)"]]
            Expand_3[["Expand(., .)"]]

            I_X -->|"FLOAT(a, 1)"| Shape_1
            Shape_1 -->|"INT64(1)"| Concat_2
            Constant_0 -->|"INT64(1)"| Concat_2
            I_X -->|"FLOAT(a, 1)"| Expand_3
            Concat_2 -->|"INT64(2)"| Expand_3

            O_Y(["Y FLOAT(a, 2)"])
            Expand_3 --> O_Y

            class I_X,I_two,O_Y ioNode
            class Constant_0 constNode
            class Shape_1,Concat_2,Expand_3 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, 1)"])
            I_two(["two INT64(1)"])

            Concat_0[["Concat([1], ., axis=0)"]]
            Expand_1[["Expand(., .)"]]

            I_two -->|"INT64(1)"| Concat_0
            I_X -->|"FLOAT(a, 1)"| Expand_1
            Concat_0 -->|"INT64(2)"| Expand_1

            O_Y(["Y FLOAT(a, 2)"])
            Expand_1 --> O_Y

            class I_X,I_two,O_Y ioNode
            class Concat_0,Expand_1 opNode
    """

    @classmethod
    def _compatible_shapes(
        cls,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        shape: DYNAMIC_SHAPE,
        expanded_shape: DYNAMIC_SHAPE,
        concat_input: Sequence[str],
    ) -> Optional[int]:
        if len(shape) != len(expanded_shape) or len(expanded_shape) != len(concat_input):
            return None
        position = []
        for i, (a, b) in enumerate(zip(shape, expanded_shape)):
            if a == b:
                continue
            position.append(i)
        if len(position) != 1:
            # It might be Identity but this should be caught by another pattern.
            return None
        return position[0]

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return self.none()
        if g.is_used_more_than_once(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_constant(node.input[1]):
            # no need
            return self.none(node, inspect.currentframe().f_lineno)
        concat_node = g.node_before(node.input[1])
        if concat_node is None or concat_node.op_type != "Concat" or concat_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.has_shape(node.input[0]) or not g.has_shape(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape1 = g.get_shape_renamed(node.input[0])
        shape2 = g.get_shape_renamed(node.output[0])
        index = self._compatible_shapes(g, shape1, shape2, concat_node.input)
        if index is None:
            return self.none(node, inspect.currentframe().f_lineno)
        # checking the other values are not 1
        if all(
            (i == index or (g.is_constant(name) and g.get_constant_scalar(name) == 1))
            for i, name in enumerate(concat_node.input)
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [concat_node, node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", concat_node: NodeProto, expand_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        shape1 = g.get_shape_renamed(expand_node.input[0])
        shape2 = g.get_shape_renamed(expand_node.output[0])
        index = self._compatible_shapes(g, shape1, shape2, concat_node.input)
        init1 = g.make_initializer(
            g.unique_name("init7_1"), g.ONE, source="ShapeBasedConcatExpandPattern.1"
        )
        new_input = [
            (iname if i == index else init1) for i, iname in enumerate(concat_node.input)
        ]
        new_name = g.unique_name(concat_node.output[0])
        return [
            g.make_node(
                "Concat",
                new_input,
                [new_name],
                axis=0,
                name=f"{self.__class__.__name__}--{concat_node.name}",
                doc_string=concat_node.doc_string,
            ),
            g.make_node(
                "Expand",
                [expand_node.input[0], new_name],
                expand_node.output,
                name=f"{self.__class__.__name__}--{expand_node.name}",
                doc_string=expand_node.doc_string,
            ),
        ]


class ExpandUnsqueezeExpandPattern(PatternOptimization):
    """
    Fuses the sequence Expand + Unsqueeze + Expand into Unsqueeze + Expand.
    Since Expand does not change the rank of a tensor, the Unsqueeze axes are
    valid for the original tensor as well, and the final Expand can handle
    both the broadcasting of the first Expand and the new dimension added by
    Unsqueeze.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(1, a, b)"])
            I_shape1(["shape1 INT64(3)"])
            I_axes(["axes INT64(1)"])
            I_shape2(["shape2 INT64(4)"])

            Constant_0[["Constant() -#gt; shape1"]]
            Constant_1[["Constant() -#gt; axes"]]
            Constant_2[["Constant() -#gt; shape2"]]
            Expand_3[["Expand(., .)"]]
            Unsqueeze_4[["Unsqueeze(., .)"]]
            Expand_5[["Expand(., .)"]]

            I_X -->|"FLOAT(1, a, b)"| Expand_3
            Constant_0 -->|"INT64(3)"| Expand_3
            Expand_3 -->|"FLOAT(c, a, b)"| Unsqueeze_4
            Constant_1 -->|"INT64(1)"| Unsqueeze_4
            Unsqueeze_4 -->|"FLOAT(c, 1, a, b)"| Expand_5
            Constant_2 -->|"INT64(4)"| Expand_5

            O_Y(["Y FLOAT(c, d, a, b)"])
            Expand_5 --> O_Y

            class I_X,I_shape1,I_axes,I_shape2,O_Y ioNode
            class Constant_0,Constant_1,Constant_2 constNode
            class Expand_3,Unsqueeze_4,Expand_5 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(1, a, b)"])
            I_axes(["axes INT64(1)"])
            I_shape1_new(["shape1_with_new_1s INT64(4)"])
            I_shape2(["shape2 INT64(4)"])

            Constant_0[["Constant() -#gt; axes"]]
            Constant_1[["Constant() -#gt; shape1_with_new_1s"]]
            Max_2[["Max(., .)"]]
            Unsqueeze_3[["Unsqueeze(., .)"]]
            Expand_4[["Expand(., .)"]]

            I_X -->|"FLOAT(1, a, b)"| Unsqueeze_3
            Constant_0 -->|"INT64(1)"| Unsqueeze_3
            Constant_1 -->|"INT64(4)"| Max_2
            I_shape2 -->|"INT64(4)"| Max_2
            Unsqueeze_3 -->|"FLOAT(1, 1, a, b)"| Expand_4
            Max_2 -->|"INT64(4)"| Expand_4

            O_Y(["Y FLOAT(c, d, a, b)"])
            Expand_4 --> O_Y

            class I_X,I_axes,I_shape2,O_Y ioNode
            class Constant_0,Constant_1 constNode
            class Max_2,Unsqueeze_3,Expand_4 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return self.none()

        # The first Expand output must be used only by the Unsqueeze.
        if g.is_used_more_than_once(node.output[0]) or g.main_opset < 13:
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        unsq_node = next_nodes[0]

        if unsq_node.op_type != "Unsqueeze" or unsq_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # The Unsqueeze axes must be provided as an input (opset >= 13).
        if not g.is_constant(unsq_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        next_next_nodes = g.next_nodes(unsq_node.output[0])
        if len(next_next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        expand2_node = next_next_nodes[0]

        if expand2_node.op_type != "Expand" or expand2_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant(node.input[1]):
            # Not implemented in this case but it is possible.
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, unsq_node, expand2_node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        expand_node: NodeProto,
        unsq_node: NodeProto,
        expand2_node: NodeProto,
    ) -> List[NodeProto]:
        exp_shapes = [
            g.get_computed_constant(expand_node.input[1]),
            g.get_computed_constant(expand2_node.input[1]),
        ]
        axes_val = g.get_computed_constant(unsq_node.input[1])

        extra_nodes: List[NodeProto] = []
        if exp_shapes[0] is not None:
            first_expand = exp_shapes[0].tolist()
            for i in axes_val:
                first_expand.insert(i, 1)
            if exp_shapes[1] is not None:
                final = np.maximum(np.array(first_expand), exp_shapes[1])
                combined_shape_input = g.make_initializer(
                    "", np.array(final, dtype=np.int64), source=f"{self.__class__.__name__}.0"
                )
            else:
                first_shape_insert = g.make_initializer(
                    "", np.array(final, dtype=np.int64), source=f"{self.__class__.__name__}.1"
                )
                combined_shape_input = g.unique_name(f"{self.__class__.__name__}_combined_shape")
                extra_nodes = [
                    g.make_node(
                        "Max",
                        [first_shape_insert, expand2_node.input[1]],
                        [combined_shape_input],
                        name=f"{self.__class__.__name__}--combined_shape",
                    )
                ]
        else:
            raise NotImplementedError(
                f"{exp_shapes[0]!r} is not constant, this is not implemented yet."
            )

        # Apply Unsqueeze directly to the original tensor (before first Expand).
        new_unsq_name = g.unique_name(f"{self.__class__.__name__}_{unsq_node.output[0]}")
        new_unsq = g.make_node(
            "Unsqueeze",
            [expand_node.input[0], unsq_node.input[1]],
            [new_unsq_name],
            name=f"{self.__class__.__name__}--{unsq_node.name}",
            doc_string=unsq_node.doc_string,
        )
        # The second Expand broadcasts the unsqueezed original tensor to the combined shape.
        new_expand2 = g.make_node(
            "Expand",
            [new_unsq_name, combined_shape_input],
            expand2_node.output,
            name=f"{self.__class__.__name__}--{expand2_node.name}",
            doc_string=expand2_node.doc_string,
        )
        return [*extra_nodes, new_unsq, new_expand2]


class SwapExpandReshapePattern(PatternOptimization):
    """
    Checks if Expand + Reshape can be swapped.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_weight(["weight FLOAT(1, 4, 1)"])
            I_stat(["stat INT64(3)"])
            I_shape(["shape INT64(3)"])

            Constant_0[["Constant() -#gt; weight"]]
            Constant_1[["Constant() -#gt; stat"]]
            Expand_2[["Expand(., .)"]]
            Reshape_3[["Reshape(., .)"]]

            Constant_0 -->|"FLOAT(1, 4, 1)"| Expand_2
            I_shape -->|"INT64(3)"| Expand_2
            Expand_2 --> Reshape_3
            Constant_1 -->|"INT64(3)"| Reshape_3

            O_Y(["Y FLOAT(a, 1, 4)"])
            Reshape_3 --> O_Y

            class I_weight,I_stat,I_shape,O_Y ioNode
            class Constant_0,Constant_1 constNode
            class Expand_2,Reshape_3 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_weight(["weight FLOAT(1, 4, 1)"])
            I_stat(["stat INT64(3)"])
            I_shape(["shape INT64(3)"])

            Reshape_0[["Reshape(., .)"]]
            Expand_1[["Expand(., .)"]]

            I_weight -->|"FLOAT(1, 4, 1)"| Reshape_0
            I_stat -->|"INT64(3)"| Reshape_0
            Reshape_0 --> Expand_1
            I_shape -->|"INT64(3)"| Expand_1

            O_Y(["Y FLOAT(a, 1, 4)"])
            Expand_1 --> O_Y

            class I_weight,I_stat,I_shape,O_Y ioNode
            class Reshape_0,Expand_1 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Reshape" or node.domain != "":
            return self.none()
        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        expand_node = g.node_before(node.input[0])
        if expand_node is None or expand_node.op_type != "Expand" or expand_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_rank(expand_node.input[0]) or g.get_rank(expand_node.input[0]) != 3:
            return self.none(node, inspect.currentframe().f_lineno)

        cst = g.get_computed_constant(node.input[1])
        if cst is None:
            return self.none(node, inspect.currentframe().f_lineno)
        shape = g.builder.value_as_shape(expand_node.input[1])
        if shape is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if tuple(cst) != (0, 1, -1) or shape[1:] != (1, 1):
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [expand_node, node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", expand_node: NodeProto, reshape_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        new_name = g.unique_name(reshape_node.output[0])
        return [
            g.make_node(
                "Reshape",
                [expand_node.input[0], reshape_node.input[1]],
                [new_name],
                name=f"{self.__class__.__name__}--{reshape_node.name}",
                doc_string=reshape_node.doc_string,
            ),
            g.make_node(
                "Expand",
                [new_name, expand_node.input[1]],
                reshape_node.output,
                name=f"{self.__class__.__name__}--{expand_node.name}",
                doc_string=expand_node.doc_string,
            ),
        ]


class SwapExpandUnsqueezePattern(PatternOptimization):
    """
    Swaps Expand and Unsqueeze when Unsqueeze directly follows Expand.
    ``Expand(X, shape) → Unsqueeze(expanded, axes)`` is rewritten as
    ``Unsqueeze(X, axes) → Expand(unsqueezed, new_shape)`` where
    ``new_shape`` is obtained by inserting ``1`` at every position listed in
    ``axes`` into the original expand shape.  Performing the Unsqueeze before
    the Expand means the Unsqueeze operates on the smaller (pre-expanded)
    tensor, which is more efficient.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(1, 5, 7)"])
            I_shape(["shape INT64(3)"])
            I_axes(["axes INT64(1)"])

            Constant_0[["Constant() -#gt; shape"]]
            Constant_1[["Constant() -#gt; axes"]]
            Expand_2[["Expand(., .)"]]
            Unsqueeze_3[["Unsqueeze(., .)"]]

            I_X -->|"FLOAT(1, 5, 7)"| Expand_2
            Constant_0 -->|"INT64(3)"| Expand_2
            Expand_2 -->|"FLOAT(3, 5, 7)"| Unsqueeze_3
            Constant_1 -->|"INT64(1)"| Unsqueeze_3

            O_Y(["Y FLOAT(3, 1, 5, 7)"])
            Unsqueeze_3 --> O_Y

            class I_X,I_shape,I_axes,O_Y ioNode
            class Constant_0,Constant_1 constNode
            class Expand_2,Unsqueeze_3 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(1, 5, 7)"])
            I_axes(["axes INT64(1)"])
            I_new_shape(["new_shape INT64(4)"])

            Constant_0[["Constant() -#gt; axes"]]
            Constant_1[["Constant() -#gt; new_shape"]]
            Unsqueeze_2[["Unsqueeze(., .)"]]
            Expand_3[["Expand(., .)"]]

            I_X -->|"FLOAT(1, 5, 7)"| Unsqueeze_2
            Constant_0 -->|"INT64(1)"| Unsqueeze_2
            Unsqueeze_2 -->|"FLOAT(1, 1, 5, 7)"| Expand_3
            Constant_1 -->|"INT64(4)"| Expand_3

            O_Y(["Y FLOAT(3, 1, 5, 7)"])
            Expand_3 --> O_Y

            class I_X,I_axes,I_new_shape,O_Y ioNode
            class Constant_0,Constant_1 constNode
            class Unsqueeze_2,Expand_3 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Expand" or node.domain != "":
            return self.none()

        # The Expand output must be used only by the Unsqueeze (opset >= 13).
        if g.is_used_more_than_once(node.output[0]) or g.main_opset < 13:
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        unsq_node = next_nodes[0]

        if unsq_node.op_type != "Unsqueeze" or unsq_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # The Unsqueeze axes must be a constant so we can rewrite the Expand shape.
        if not g.is_constant(unsq_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        # We must be able to determine the new Expand shape: either the original
        # Expand shape is a constant, or the Expand output shape is fully known.
        expand_shape_cst = g.get_computed_constant(node.input[1])
        if expand_shape_cst is None:
            if not g.has_shape(node.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)
            shape = g.get_shape(node.output[0])
            if not all_int(shape):
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, unsq_node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", expand_node: NodeProto, unsq_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        axes_val = g.get_computed_constant(unsq_node.input[1])

        expand_shape_cst = g.get_computed_constant(expand_node.input[1])
        if expand_shape_cst is not None:
            base_shape = expand_shape_cst.tolist()
        else:
            base_shape = list(g.get_shape(expand_node.output[0]))

        # Normalise negative axes (relative to the output rank after unsqueeze).
        rank_out = len(base_shape) + len(axes_val)
        axes = sorted(int(a) if a >= 0 else int(a) + rank_out for a in axes_val)

        # Insert 1s at the axes positions to build the new Expand target shape.
        new_shape_list = list(base_shape)
        for a in axes:
            new_shape_list.insert(a, 1)

        new_shape = g.make_initializer(
            "",
            np.array(new_shape_list, dtype=np.int64),
            source=f"{self.__class__.__name__}.apply.new_shape",
        )

        new_name = g.unique_name(f"{self.__class__.__name__}_{expand_node.input[0]}")
        return [
            g.make_node(
                "Unsqueeze",
                [expand_node.input[0], unsq_node.input[1]],
                [new_name],
                name=f"{self.__class__.__name__}--{unsq_node.name}",
                doc_string=unsq_node.doc_string,
            ),
            g.make_node(
                "Expand",
                [new_name, new_shape],
                unsq_node.output,
                name=f"{self.__class__.__name__}--{expand_node.name}",
                doc_string=expand_node.doc_string,
            ),
        ]
