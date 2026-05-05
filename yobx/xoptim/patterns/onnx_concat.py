import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ...helpers.onnx_helper import unary_like_op_types
from ..patterns_api import MatchResult, PatternOptimization


class ConcatGatherPattern(PatternOptimization):
    """
    Simplifies ``Gather(Concat(...), cst_index)`` when the index is a constant.

    The Concat inputs may each have one or more elements along axis 0.  The
    pattern locates which Concat input contains the element addressed by the
    constant index and replaces the Gather with:

    * an ``Identity`` node when the located input has exactly one element, or
    * a ``Gather`` node on that input with an adjusted (local) index when the
      input has more than one element.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_D1(["D1 INT64(1)"])
            I_D2(["D2 INT64(1)"])

            Concat_0[["Concat(., ., axis=0)"]]
            Gather_1[["Gather(., [1])"]]

            I_D1 -->|"INT64(1)"| Concat_0
            I_D2 -->|"INT64(1)"| Concat_0
            Concat_0 -->|"INT64(2)"| Gather_1

            O_Y(["Y INT64(1)"])
            Gather_1 --> O_Y

            class I_D1,I_D2,O_Y ioNode
            class Concat_0,Gather_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_D2(["D2 INT64(1)"])

            Identity_0[["Identity(.)"]]

            I_D2 -->|"INT64(1)"| Identity_0

            O_Y(["Y INT64(1)"])
            Identity_0 --> O_Y

            class I_D2,O_Y ioNode
            class Identity_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    @staticmethod
    def _concat_input_bucket(
        g: "GraphBuilderPatternOptimization", concat_node: NodeProto, idx: int  # noqa: F821
    ) -> Optional[tuple]:
        """Locates which Concat input holds element at position ``idx``.

        Returns ``(input_name, local_index, input_size)`` where ``input_name``
        is the Concat input that contains position ``idx``, ``local_index`` is
        the offset of ``idx`` within that input (``idx - cumulative_offset``),
        and ``input_size`` is the number of elements in that input.

        Returns ``None`` when any input size is non-concrete (symbolic) or
        ``idx`` is out of range.
        """
        offset = 0
        for inp in concat_node.input:
            shape = g.get_shape(inp)
            n = shape[0]
            if not isinstance(n, int):
                return None
            if offset + n > idx:
                return inp, idx - offset, n
            offset += n
        return None

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Gather" or node.domain != "":
            return self.none()
        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(node.input[1])
        if cst is None or cst.dtype != np.int64 or cst.shape != (1,):
            return self.none(node, inspect.currentframe().f_lineno)
        before = g.node_before(node.input[0])
        if not before or before.op_type != "Concat":
            return self.none(node, inspect.currentframe().f_lineno)
        if any(not g.has_shape(i) for i in before.input):
            return self.none(node, inspect.currentframe().f_lineno)
        # All inputs must be 1D with concrete integer sizes.
        if any(len(g.get_shape(i)) != 1 for i in before.input):
            return self.none(node, inspect.currentframe().f_lineno)
        if self._concat_input_bucket(g, before, int(cst[0])) is None:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [before, node], self.apply)

    def apply(
        self, g: "GraphBuilder", concat_node: NodeProto, gather_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        cst = g.get_computed_constant(gather_node.input[1])
        idx = int(cst[0])
        inp, local_idx, inp_size = self._concat_input_bucket(g, concat_node, idx)
        if inp_size == 1:
            new_node = g.make_node(
                "Identity",
                [inp],
                gather_node.output,
                name=f"{self.__class__.__name__}--{gather_node.name}",
                doc_string=gather_node.doc_string,
            )
        else:
            # Multi-element bucket: if the input comes from a ranged Shape node,
            # emit a narrower Shape directly so downstream patterns (e.g.
            # ConcatReshapePattern) still see a Shape-derived value.
            inp_producer = g.node_before(inp)
            if (
                inp_producer is not None
                and inp_producer.op_type == "Shape"
                and inp_producer.domain == ""
            ):
                shape_start = next(
                    (int(a.i) for a in inp_producer.attribute if a.name == "start"), 0
                )
                new_node = g.make_node(
                    "Shape",
                    [inp_producer.input[0]],
                    gather_node.output,
                    start=shape_start + local_idx,
                    end=shape_start + local_idx + 1,
                    name=f"{self.__class__.__name__}--{gather_node.name}",
                    doc_string=gather_node.doc_string,
                )
            else:
                new_idx_name = g.make_initializer(
                    "",
                    np.array([local_idx], dtype=np.int64),
                    source=f"{self.__class__.__name__}.apply.idx",
                )
                new_node = g.make_node(
                    "Gather",
                    [inp, new_idx_name],
                    gather_node.output,
                    name=f"{self.__class__.__name__}--{gather_node.name}",
                    doc_string=gather_node.doc_string,
                )
        return (
            [concat_node, new_node]
            if g.is_used_more_than_once(concat_node.output[0])
            else [new_node]
        )


class _CommonConcatPattern(PatternOptimization):
    def remove_set(self, g, node):
        att = g.get_attribute(node, "axis")
        axis = att.i
        rem = set()
        for idi, i in enumerate(node.input):
            if not g.has_shape(i):
                continue
            shape = g.get_shape(i)
            if axis < len(shape) and shape[axis] == 0:
                rem.add(idi)
        return rem


class ConcatEmptyPattern(_CommonConcatPattern):
    """
    Checks if one of the concatenated values is empty.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_Y(["Y INT64(b)"])
            I_X(["X INT64(a)"])

            Concat_0[["Concat(., ., [], axis=0)"]]

            I_X -->|"INT64(a)"| Concat_0
            I_Y -->|"INT64(b)"| Concat_0

            O_Z(["Z INT64(c)"])
            Concat_0 --> O_Z

            class I_Y,I_X,O_Z ioNode
            class Concat_0 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_Y(["Y INT64(b)"])
            I_X(["X INT64(a)"])

            Concat_0[["Concat(., ., axis=0)"]]

            I_X -->|"INT64(a)"| Concat_0
            I_Y -->|"INT64(b)"| Concat_0

            O_Z(["Z INT64(c)"])
            Concat_0 --> O_Z

            class I_Y,I_X,O_Z ioNode
            class Concat_0 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Concat" or node.domain != "":
            return self.none()
        rem = self.remove_set(g, node)
        if not rem:
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [node], self.apply, insert_at=node)

    def apply(self, g: "GraphBuilder", node: NodeProto) -> List[NodeProto]:  # noqa: F821
        rem = self.remove_set(g, node)
        assert rem, f"rem is empty for node={node}"
        new_inputs = [n for i, n in enumerate(node.input) if i not in rem]
        if len(rem) == len(node.input) - 1:
            # Identity
            return [
                g.make_node(
                    "Identity",
                    new_inputs,
                    node.output,
                    name=f"{self.__class__.__name__}--{node.name}",
                    doc_string=node.doc_string,
                )
            ]
        new_node = g.make_node(
            "Concat",
            new_inputs,
            node.output,
            name=f"{self.__class__.__name__}--{node.name}",
            doc_string=node.doc_string,
        )
        new_node.attribute.extend(node.attribute)
        return [new_node]


class ConcatTwiceUnaryPattern(_CommonConcatPattern):
    """
    Sin(Concat(x,x)) -> Concat(Sin(x), Sin(x)).

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(b, c)"])

            Concat_0[["Concat(., ., axis=0)"]]
            Sin_1[["Sin(.)"]]

            I_X -->|"FLOAT(b, c)"| Concat_0
            Concat_0 -->|"FLOAT(2*b, c)"| Sin_1

            O_xsin(["xsin FLOAT(2*b, c)"])
            Sin_1 --> O_xsin

            class I_X,O_xsin ioNode
            class Concat_0,Sin_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(b, c)"])

            Sin_0[["Sin(.)"]]
            Concat_1[["Concat(., ., axis=0)"]]

            I_X -->|"FLOAT(b, c)"| Sin_0
            Sin_0 -->|"FLOAT(b, c)"| Concat_1

            O_xsin(["xsin FLOAT(2*b, c)"])
            Concat_1 --> O_xsin

            class I_X,O_xsin ioNode
            class Sin_0,Concat_1 opNode
    """

    _unary_types = unary_like_op_types()
    _binary_types_scalar_cst = {"Mul", "Add", "Div", "Sub"}

    @classmethod
    def _valid_node(
        cls,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        concat: NodeProto,
        unary: NodeProto,
    ):
        if unary.op_type in cls._unary_types:
            return True
        if unary.op_type == "Unsqueeze" and unary.domain == "":
            if g.is_constant_scalar(unary.input[1]):
                cst = g.get_constant_scalar(unary.input[1])
                axis = g.get_attribute(concat, "axis").i
                if axis == -1 and cst != -1 and cst < g.get_rank(unary.input[0]):
                    return True
        if unary.op_type in cls._binary_types_scalar_cst and g.is_constant_scalar(unary.input[1]):
            return True
        return False

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            g.main_opset < 18
            or node.op_type != "Concat"
            or node.domain != ""
            or len(node.input) != 2
            or node.input[0] != node.input[1]
        ):
            return self.none()

        # Let's check what follows.
        nodes = [n for n in g.next_nodes(node.output[0]) if self._valid_node(g, node, n)]
        if nodes:
            return MatchResult(self, [node, nodes[0]], self.apply)
        return self.none(node, inspect.currentframe().f_lineno)

    def apply(
        self, g: "GraphBuilder", concat: NodeProto, unary: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        new_name = g.unique_name(f"u{unary.output[0]}")
        nodes = [
            g.make_node(
                unary.op_type,
                [concat.input[0], *unary.input[1:]],
                [new_name],
                name=f"{self.__class__.__name__}--{unary.name}",
                doc_string=unary.doc_string,
            ),
            g.make_node(
                concat.op_type,
                [new_name, new_name],
                [unary.output[0]],
                name=f"{self.__class__.__name__}--{concat.name}",
                doc_string=concat.doc_string,
            ),
        ]
        if unary.attribute:
            nodes[0].attribute.extend(unary.attribute)
        if concat.attribute:
            nodes[1].attribute.extend(concat.attribute)

        if g.is_used_more_than_once(concat.output[0]):
            return [concat, *nodes]
        return nodes
