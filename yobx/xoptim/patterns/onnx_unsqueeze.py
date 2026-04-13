import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class SqueezeUnsqueezePattern(PatternOptimization):
    """
    Replaces the sequence Squeeze, Unsqueeze by Identity or the other ways around.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, 1, 1, d)"])

            Unsqueeze_0[["Unsqueeze(., [1, 2])"]]
            Squeeze_1[["Squeeze(., [1, 2])"]]

            I_X -->|"FLOAT(a, 1, 1, d)"| Unsqueeze_0
            Unsqueeze_0 -->|"FLOAT(a, 1, 1, 1, 1, d)"| Squeeze_1

            O_Y(["Y FLOAT(a, 1, 1, d)"])
            Squeeze_1 --> O_Y

            class I_X,O_Y ioNode
            class Unsqueeze_0,Squeeze_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, 1, 1, d)"])

            Identity_0[["Identity(.)"]]

            I_X -->|"FLOAT(a, 1, 1, d)"| Identity_0

            O_Y(["Y FLOAT(a, 1, 1, d)"])
            Identity_0 --> O_Y

            class I_X,O_Y ioNode
            class Identity_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def _diff_axes(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        first_node: NodeProto,
        second_node: NodeProto,
    ):
        if first_node.op_type == "Unsqueeze" and len(second_node.input) == 1:
            return "Squeeze", None
        axes1 = (
            None if len(first_node.input) == 1 else g.get_computed_constant(first_node.input[1])
        )
        axes2 = (
            None if len(second_node.input) == 1 else g.get_computed_constant(second_node.input[1])
        )

        if axes1 is None and first_node.op_type == "Squeeze" and g.has_shape(first_node.input[0]):
            axes1 = tuple(i for i, a in enumerate(g.get_shape(first_node.input[0])) if a == 1)
        if (
            axes2 is None
            and second_node.op_type == "Squeeze"
            and g.has_shape(second_node.input[0])
        ):
            axes2 = tuple(i for i, a in enumerate(g.get_shape(second_node.input[0])) if a == 1)

        if len(first_node.input) == 2 and axes1 is None:
            return self.none(second_node, inspect.currentframe().f_lineno)
        if len(second_node.input) == 2 and axes2 is None:
            return self.none(second_node, inspect.currentframe().f_lineno)
        tax1 = tuple(map(int, axes1))
        tax2 = tuple(map(int, axes2))
        if tax1 == tax2:
            if len(axes1) > 1 and tuple(map(int, axes1)) != tuple(
                range(min(axes1), max(axes1) + 1)
            ):
                return self.none(second_node, inspect.currentframe().f_lineno)
            return "Identity", None
        if first_node.op_type == "Unsqueeze" and set(tax1) < set(tax2):
            keep_axes = sorted(set(tax2) - set(tax1))
            for i in range(len(keep_axes)):
                m = len([t for t in tax1 if t < keep_axes[i]])
                keep_axes[i] -= m
            return "Squeeze", tuple(keep_axes)
        return self.none(second_node, inspect.currentframe().f_lineno)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type not in {"Squeeze", "Unsqueeze"} or node.domain != "":
            return self.none()
        node_before = g.node_before(node.input[0])
        if (
            node_before is None
            or node_before.op_type not in {"Squeeze", "Unsqueeze"}
            or node_before.op_type == node.op_type
            or node_before.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        diff = self._diff_axes(g, node_before, node)
        if diff is None:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(
            self,
            [node_before, node],
            self.apply,
            insert_at=node_before if g.is_used_more_than_once(node.input[0]) else node,
        )

    def apply(
        self, g: "GraphBuilder", node_first: NodeProto, node_second: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        diff = self._diff_axes(g, node_first, node_second)
        assert diff is not None, "Match should not have happened then."
        op_type, args = diff
        if args is None:
            new_node = g.make_node(
                op_type,
                [node_first.input[0]],
                [node_second.output[0]],
                name=f"{self.__class__.__name__}--{node_first.name}",
                doc_string=node_first.doc_string,
            )
        else:
            new_axes = g.make_initializer(
                "",
                np.array(args, dtype=np.int64),
                source="SqueezeUnsqueezePattern.apply.new_axes",
            )
            new_node = g.make_node(
                op_type,
                [node_first.input[0], new_axes],
                [node_second.output[0]],
                name=f"{self.__class__.__name__}--{node_first.name}",
                doc_string=node_first.doc_string,
            )
        return (
            [node_first, new_node]
            if g.is_used_more_than_once(node_second.input[0])
            else [new_node]
        )


class UnsqueezeUnsqueezePattern(PatternOptimization):
    """
    Replaces the sequence Unsqueeze, Unsqueeze by Unsqueeze.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b)"])

            Unsqueeze_0[["Unsqueeze(., [2])"]]
            Unsqueeze_1[["Unsqueeze(., [3])"]]

            I_X -->|"FLOAT(a, b)"| Unsqueeze_0
            Unsqueeze_0 -->|"FLOAT(a, b, 1)"| Unsqueeze_1

            O_Y(["Y FLOAT(1, 1, a, b)"])
            Unsqueeze_1 --> O_Y

            class I_X,O_Y ioNode
            class Unsqueeze_0,Unsqueeze_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b)"])

            Unsqueeze_0[["Unsqueeze(., [2, 3])"]]

            I_X -->|"FLOAT(a, b)"| Unsqueeze_0

            O_Y(["Y FLOAT(1, 1, a, b)"])
            Unsqueeze_0 --> O_Y

            class I_X,O_Y ioNode
            class Unsqueeze_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Unsqueeze" or node.domain != "":
            return self.none()
        next_nodes = [n for n in g.next_nodes(node.output[0]) if n.op_type == "Unsqueeze"]
        if not next_nodes:
            return self.none(node, inspect.currentframe().f_lineno)
        next_node = next_nodes[0]
        if next_node.op_type != "Unsqueeze" or node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if next_node.input[0] != node.output[0]:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[1]) or not g.is_constant(next_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_rank(node.input[1]) or not g.has_rank(next_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_rank(node.input[1]) != 1 or g.get_rank(next_node.input[1]) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        axis1 = g.get_constant_or_attribute(node, "axis", 1)
        axis2 = g.get_constant_or_attribute(next_node, "axis", 1)
        if (len(axis1) > 1 or len(axis2) > 1) and not g.has_rank(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node, next_node], self.apply, insert_at=node)

    @classmethod
    def _unsqueeze(cls, current, axes):
        if len(axes) == 1:
            current.insert(axes[0], True)
        else:
            for a in axes:
                current.insert(a, True)
        return current

    def apply(
        self, g: "GraphBuilder", node: NodeProto, next_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        axis1 = g.get_constant_or_attribute(node, "axis", 1)
        axis2 = g.get_constant_or_attribute(next_node, "axis", 1)
        if len(axis1) == 1 and len(axis2) == 1:
            # No need to be clever.
            if axis2[0] == axis1[0]:
                new_axis_value = [*axis2, *(axis1 + 1)]
            elif axis1[0] < axis2[0]:
                new_axis_value = [*axis1, *axis2]
            else:
                new_axis_value = [*axis2, *(axis1 + 1)]
            new_axis = g.make_initializer(
                "",
                np.array(new_axis_value, dtype=np.int64),
                source="UnsqueezeUnsqueezePattern.apply.new_axis.0",
            )
            new_node = g.make_node(
                "Unsqueeze",
                [node.input[0], new_axis],
                next_node.output,
                name=f"{self.__class__.__name__}--{node.name}",
                doc_string=next_node.doc_string,
            )
            if g.is_used_more_than_once(node.output[0]):
                return [node, new_node]
            return [new_node]

        rk = g.get_rank(node.input[0])
        existing = [False for i in range(rk)]
        existing = self._unsqueeze(existing, axis1)
        assert axis1.min() < 0 or [i for i, a in enumerate(existing) if a] == list(axis1), (
            f"Something is wrong: rk={rk}, axis={axis1}, existing={existing}, shapes:"
            f"{g.get_shape(node.input[0]) if g.has_shape(node.input[0]) else '?'}, "
            f"{g.get_shape(node.output[0]) if g.has_shape(node.output[0]) else '?'}"
        )
        existing = self._unsqueeze(existing, axis2)
        new_axes = [i for i, a in enumerate(existing) if a]
        new_axis = g.make_initializer(
            "",
            np.array(new_axes, dtype=np.int64),
            source="UnsqueezeUnsqueezePattern.apply.new_axis.1",
        )
        new_node = g.make_node(
            "Unsqueeze",
            [node.input[0], new_axis],
            next_node.output,
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=next_node.doc_string,
        )
        if g.is_used_more_than_once(node.output[0]):
            return [node, new_node]
        return [new_node]


class SqueezeAddPattern(PatternOptimization):
    """
    Replaces the sequence Add(Squeeze, Squeeze) by Squeeze(Add).

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_S2(["S2 INT64(1)"])
            I_S1(["S1 INT64(1)"])

            Squeeze_0[["Squeeze(.)"]]
            Squeeze_1[["Squeeze(.)"]]
            Add_2[["Add(., .)"]]

            I_S1 -->|"INT64(1)"| Squeeze_0
            I_S2 -->|"INT64(1)"| Squeeze_1
            Squeeze_0 -->|"INT64()"| Add_2
            Squeeze_1 -->|"INT64()"| Add_2

            O_s(["s INT64()"])
            Add_2 --> O_s

            class I_S2,I_S1,O_s ioNode
            class Squeeze_0,Squeeze_1,Add_2 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_S2(["S2 INT64(1)"])
            I_S1(["S1 INT64(1)"])

            Add_0[["Add(., .)"]]
            Squeeze_1[["Squeeze(.)"]]

            I_S1 -->|"INT64(1)"| Add_0
            I_S2 -->|"INT64(1)"| Add_0
            Add_0 -->|"INT64(1)"| Squeeze_1

            O_s(["s INT64()"])
            Squeeze_1 --> O_s

            class I_S2,I_S1,O_s ioNode
            class Add_0,Squeeze_1 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Add" or node.domain != "" or g.builder.main_opset < 13:
            return self.none()
        node_before = [g.node_before(node.input[0]), g.node_before(node.input[1])]
        if (
            not node_before[0]
            or not node_before[1]
            or node_before[0].op_type != "Squeeze"
            or node_before[1].op_type != "Squeeze"
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if len(node_before[0].input) == 2:
            s1 = g.builder.value_as_shape(node_before[0].input[1])
        else:
            if not g.has_shape(node_before[0].input[0]) or g.get_shape(
                node_before[0].input[0]
            ) != (1,):
                return self.none(node, inspect.currentframe().f_lineno)
            s1 = (0,)

        if len(node_before[1].input) == 2:
            s2 = g.builder.value_as_shape(node_before[1].input[1])
        else:
            if not g.has_shape(node_before[1].input[0]) or g.get_shape(
                node_before[1].input[0]
            ) != (1,):
                return self.none(node, inspect.currentframe().f_lineno)
            s2 = (0,)

        if s1 is None or s2 is None or s1 != s2:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [*node_before, node], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        squeeze1: NodeProto,
        squeeze2: NodeProto,
        add: NodeProto,
    ) -> List[NodeProto]:
        new_name = g.unique_name(f"{self.__class__.__name__}_{add.output[0]}")
        new_nodes = [
            g.make_node(
                "Add",
                [squeeze1.input[0], squeeze2.input[0]],
                [new_name],
                name=f"{self.__class__.__name__}--{add.name}",
                doc_string=add.doc_string,
            ),
            g.make_node(
                "Squeeze",
                [new_name, *squeeze1.input[1:]],
                add.output,
                name=f"{self.__class__.__name__}--{squeeze1.name}",
                doc_string=squeeze1.doc_string,
            ),
        ]
        if g.is_used_more_than_once(add.input[1]):
            new_nodes = [squeeze2, *new_nodes]
        if g.is_used_more_than_once(add.input[0]):
            new_nodes = [squeeze1, *new_nodes]
        return new_nodes


class MulUnsqueezeUnsqueezePattern(PatternOptimization):
    """
    Replaces ``Mul(Unsqueeze(x, axes), Unsqueeze(y, axes))`` by
    ``Unsqueeze(Mul(x, y), axes)`` when both inputs are unsqueezed
    with the same axes.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b)"])
            I_Y(["Y FLOAT(a, b)"])

            Unsqueeze_0[["Unsqueeze(., [2])"]]
            Unsqueeze_1[["Unsqueeze(., [2])"]]
            Mul_2[["Mul(., .)"]]

            I_X -->|"FLOAT(a, b)"| Unsqueeze_0
            I_Y -->|"FLOAT(a, b)"| Unsqueeze_1
            Unsqueeze_0 -->|"FLOAT(a, b, 1)"| Mul_2
            Unsqueeze_1 -->|"FLOAT(a, b, 1)"| Mul_2

            O_Z(["Z FLOAT(a, b, 1)"])
            Mul_2 --> O_Z

            class I_X,I_Y,O_Z ioNode
            class Unsqueeze_0,Unsqueeze_1,Mul_2 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b)"])
            I_Y(["Y FLOAT(a, b)"])

            Mul_0[["Mul(., .)"]]
            Unsqueeze_1[["Unsqueeze(., [2])"]]

            I_X -->|"FLOAT(a, b)"| Mul_0
            I_Y -->|"FLOAT(a, b)"| Mul_0
            Mul_0 -->|"FLOAT(a, b)"| Unsqueeze_1

            O_Z(["Z FLOAT(a, b, 1)"])
            Unsqueeze_1 --> O_Z

            class I_X,I_Y,O_Z ioNode
            class Mul_0,Unsqueeze_1 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Mul" or node.domain != "" or g.builder.main_opset < 13:
            return self.none()
        node_before = [g.node_before(node.input[0]), g.node_before(node.input[1])]
        if (
            not node_before[0]
            or not node_before[1]
            or node_before[0].op_type != "Unsqueeze"
            or node_before[1].op_type != "Unsqueeze"
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # Both Unsqueeze nodes must have a constant axes input (second input).
        if len(node_before[0].input) < 2 or len(node_before[1].input) < 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node_before[0].input[1]) or not g.is_constant(
            node_before[1].input[1]
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        axes0 = g.get_constant_or_attribute(node_before[0], "axis", 1)
        axes1 = g.get_constant_or_attribute(node_before[1], "axis", 1)
        if axes0 is None or axes1 is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if list(axes0) != list(axes1):
            return self.none(node, inspect.currentframe().f_lineno)

        # The unsqueezed outputs must not be used elsewhere.
        if g.is_used_more_than_once(node.input[0]) or g.is_used_more_than_once(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [*node_before, node], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        unsqueeze1: NodeProto,
        unsqueeze2: NodeProto,
        mul: NodeProto,
    ) -> List[NodeProto]:
        new_name = g.unique_name(f"{self.__class__.__name__}_{mul.output[0]}")
        new_nodes = [
            g.make_node(
                "Mul",
                [unsqueeze1.input[0], unsqueeze2.input[0]],
                [new_name],
                name=f"{self.__class__.__name__}--{mul.name}",
                doc_string=mul.doc_string,
            ),
            g.make_node(
                "Unsqueeze",
                [new_name, *unsqueeze1.input[1:]],
                mul.output,
                name=f"{self.__class__.__name__}--{unsqueeze1.name}",
                doc_string=unsqueeze1.doc_string,
            ),
        ]
        return new_nodes


class SqueezeBinaryUnsqueezePattern(PatternOptimization):
    """
    Replaces the sequence Squeeze Binary Unsqueeze) by Binary.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_zero(["zero INT64(1)"])
            I_d(["d INT64(1)"])
            I_two(["two INT64()"])

            Constant_0[["Constant() -#gt; two"]]
            Constant_1[["Constant() -#gt; zero"]]
            Squeeze_2[["Squeeze(.)"]]
            Div_3[["Div(., .)"]]
            Unsqueeze_4[["Unsqueeze(., .)"]]

            I_d -->|"INT64(1)"| Squeeze_2
            Squeeze_2 -->|"INT64()"| Div_3
            Constant_0 -->|"INT64()"| Div_3
            Div_3 -->|"INT64()"| Unsqueeze_4
            Constant_1 -->|"INT64(1)"| Unsqueeze_4

            O_e(["e INT64(1)"])
            Unsqueeze_4 --> O_e

            class I_zero,I_d,I_two,O_e ioNode
            class Constant_0,Constant_1 constNode
            class Squeeze_2,Div_3,Unsqueeze_4 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_zero(["zero INT64(1)"])
            I_d(["d INT64(1)"])
            I_two(["two INT64()"])

            Unsqueeze_0[["Unsqueeze(., .)"]]
            Div_1[["Div(., .)"]]

            I_two -->|"INT64()"| Unsqueeze_0
            I_zero -->|"INT64(1)"| Unsqueeze_0
            I_d -->|"INT64(1)"| Div_1
            Unsqueeze_0 --> Div_1

            O_e(["e INT64(1)"])
            Div_1 --> O_e

            class I_zero,I_d,I_two,O_e ioNode
            class Unsqueeze_0,Div_1 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Unsqueeze" or node.domain != "" or g.builder.main_opset < 13:
            return self.none()
        if not g.is_constant_scalar(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        scalar = g.get_constant_scalar(node.input[1])
        if scalar != 0:
            return self.none(node, inspect.currentframe().f_lineno)

        binary = g.node_before(node.input[0])
        if binary is None or binary.op_type not in {"Add", "Div", "Mul", "Sub"}:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(binary.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(binary.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_rank(binary.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_rank(binary.input[1]) != 0:
            return self.none(node, inspect.currentframe().f_lineno)

        squeeze = g.node_before(binary.input[0])
        if squeeze is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if len(squeeze.input) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_rank(squeeze.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_rank(squeeze.input[0]) != 1:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [squeeze, binary, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        squeeze_node: NodeProto,
        binary_node: NodeProto,
        unsqueeze_node: NodeProto,
    ) -> List[NodeProto]:
        new_name = g.unique_name(f"{self.__class__.__name__}_{binary_node.input[1]}")
        new_nodes = [
            g.make_node(
                "Unsqueeze",
                [binary_node.input[1], unsqueeze_node.input[1]],
                [new_name],
                name=f"{self.__class__.__name__}--{unsqueeze_node.name}",
                doc_string=unsqueeze_node.doc_string,
            ),
            g.make_node(
                binary_node.op_type,
                [squeeze_node.input[0], new_name],
                [unsqueeze_node.output[0]],
                name=f"{self.__class__.__name__}--{binary_node.name}",
                doc_string=binary_node.doc_string,
            ),
        ]
        return new_nodes
