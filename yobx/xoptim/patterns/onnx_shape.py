import inspect
from typing import List, Optional
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class ShapeBasedShapeShapeAddPattern(PatternOptimization):
    """
    Tries to find another way to get a dimension obtained with the addition of two.
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Add" or node.domain != "":
            return self.none()
        shape1 = g.node_before(node.input[0])
        if shape1 is None or shape1.op_type != "Shape" or shape1.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        shape2 = g.node_before(node.input[1])
        if shape2 is None or shape2.op_type != "Shape" or shape2.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        # ishape1 = g.get_shape_renamed(shape1.input[0])
        # ishape2 = g.get_shape_renamed(shape2.input[0])
        # value1 = g.builder.value_as_shape(node.input[0])
        # value2 = g.builder.value_as_shape(node.input[1])
        # input_shapes = [g.get_shape_renamed(i) for i in g.builder.input_names]
        # g.builder._known_value_shape
        # g.builder.constraints_)
        # g.builder.replacements_dimensions_
        return self.none(node, inspect.currentframe().f_lineno)
        # return MatchResult(self, [shape1, shape2, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        shape1_node: NodeProto,
        shape2_node: NodeProto,
        add_node: NodeProto,
    ) -> List[NodeProto]:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented yet.")


class ShapeUnsqueezePattern(PatternOptimization):
    """
    Replaces ``Gather(Shape(Unsqueeze(X, axes)), indices)`` by
    ``Gather(Shape(X), remapped_indices)`` when none of the ``indices``
    fall on an axis inserted by the ``Unsqueeze``.

    The key observation is that ``Shape(Unsqueeze(X, axes))`` produces a
    shape vector that is identical to ``Shape(X)`` except for the inserted
    ``1`` entries at the ``axes`` positions.  When a downstream ``Gather``
    selects only positions that are *not* among those axes, the inserted
    ``1`` values are never read, so the Unsqueeze is unnecessary.  The
    selected indices are remapped to the original tensor by subtracting, for
    each gathered position ``i``, the number of ``axes`` values that are
    strictly less than ``i``.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b, c)"])
            I_axes(["axes INT64(1)"])
            I_idx(["idx INT64(2)"])

            Constant_0[["Constant() -#gt; axes"]]
            Constant_1[["Constant() -#gt; idx"]]
            Unsqueeze_2[["Unsqueeze(., .)"]]
            Shape_3[["Shape(.)"]]
            Gather_4[["Gather(., .)"]]

            I_X -->|"FLOAT(a, b, c)"| Unsqueeze_2
            Constant_0 -->|"INT64(1)"| Unsqueeze_2
            Unsqueeze_2 -->|"FLOAT(a, 1, b, c)"| Shape_3
            Shape_3 -->|"INT64(4)"| Gather_4
            Constant_1 -->|"INT64(2)"| Gather_4

            O_Y(["Y INT64(2)"])
            Gather_4 --> O_Y

            class I_X,I_axes,I_idx,O_Y ioNode
            class Constant_0,Constant_1 constNode
            class Unsqueeze_2,Shape_3,Gather_4 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b, c)"])
            I_new_idx(["new_idx INT64(2)"])

            Constant_0[["Constant() -#gt; new_idx"]]
            Shape_1[["Shape(.)"]]
            Gather_2[["Gather(., .)"]]

            I_X -->|"FLOAT(a, b, c)"| Shape_1
            Shape_1 -->|"INT64(3)"| Gather_2
            Constant_0 -->|"INT64(2)"| Gather_2

            O_Y(["Y INT64(2)"])
            Gather_2 --> O_Y

            class I_X,I_new_idx,O_Y ioNode
            class Constant_0 constNode
            class Shape_1,Gather_2 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Shape" or node.domain != "":
            return self.none()

        # Input to Shape must come from an Unsqueeze node.
        unsq_node = g.node_before(node.input[0])
        if unsq_node is None or unsq_node.op_type != "Unsqueeze" or unsq_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # Unsqueeze output must only be consumed by this Shape node.
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        # Unsqueeze axes must be a constant second input (opset >= 13 form).
        if len(unsq_node.input) < 2 or not g.is_constant(unsq_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        axes = g.get_computed_constant(unsq_node.input[1])
        if axes is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # Need the rank of the Unsqueeze input to normalise negative axes.
        if not g.has_rank(unsq_node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        input_rank = g.get_rank(unsq_node.input[0])
        output_rank = input_rank + len(axes)

        # Shape output must be consumed by exactly one Gather node.
        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        gather_node = g.next_node(node.output[0])
        if gather_node is None or gather_node.op_type != "Gather" or gather_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # Gather must use the Shape output as its data input (input[0]).
        if gather_node.input[0] != node.output[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        # Gather indices must be a constant.
        if not g.is_constant(gather_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        indices = g.get_computed_constant(gather_node.input[1])
        if indices is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # Normalise both axes and indices to non-negative values.
        axes_norm = set(int(a) % output_rank for a in axes.flatten())
        indices_norm = set(int(i) % output_rank for i in indices.flatten())

        # Gather indices must not overlap with the Unsqueeze axes.
        if axes_norm & indices_norm:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [unsq_node, node, gather_node], self.apply, insert_at=unsq_node)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        unsq_node: NodeProto,
        shape_node: NodeProto,
        gather_node: NodeProto,
    ) -> List[NodeProto]:
        axes = g.get_computed_constant(unsq_node.input[1])
        input_rank = g.get_rank(unsq_node.input[0])
        output_rank = input_rank + len(axes)
        axes_norm = sorted(int(a) % output_rank for a in axes.flatten())

        indices = g.get_computed_constant(gather_node.input[1])
        flat_indices = [int(i) % output_rank for i in indices.flatten()]

        # Remap each index: subtract the count of Unsqueeze axes strictly less than it.
        remapped = [i - sum(1 for a in axes_norm if a < i) for i in flat_indices]
        new_indices = np.array(remapped, dtype=np.int64).reshape(indices.shape)

        new_indices_name = g.make_initializer(
            "", new_indices, source=f"{self.__class__.__name__}.apply.remapped_indices"
        )

        new_shape_name = g.unique_name(f"{self.__class__.__name__}_{shape_node.output[0]}")
        new_shape_node = g.make_node(
            "Shape",
            [unsq_node.input[0]],
            [new_shape_name],
            name=f"{self.__class__.__name__}--{shape_node.name}",
            doc_string=shape_node.doc_string,
        )

        gather_axis = gather_node.attribute[0].i if gather_node.attribute else 0
        new_gather_node = g.make_node(
            "Gather",
            [new_shape_name, new_indices_name],
            gather_node.output,
            axis=gather_axis,
            name=f"{self.__class__.__name__}--{gather_node.name}",
            doc_string=gather_node.doc_string,
        )

        return [new_shape_node, new_gather_node]
