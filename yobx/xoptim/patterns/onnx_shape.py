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


class UnsqueezeShapePattern(PatternOptimization):
    """
    Replaces ``Shape(Unsqueeze(X, axes))`` by
    ``Gather(Shape(X), remapped_indices)`` where ``remapped_indices``
    are all positions in the unsqueezed shape that are *not* inserted by
    the ``Unsqueeze``, mapped back to indices in the original tensor's shape.

    The key observation is that ``Shape(Unsqueeze(X, axes))`` produces a
    shape vector that is identical to ``Shape(X)`` except for the inserted
    ``1`` entries at the ``axes`` positions.  Gathering only the non-inserted
    positions from ``Shape(X)`` avoids materialising the Unsqueeze on the
    (potentially large) data tensor entirely; the same values are retrieved
    by a cheap Gather on the tiny shape vector.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b, c)"])
            I_axes(["axes INT64(1)"])

            Constant_0[["Constant() -#gt; axes"]]
            Unsqueeze_1[["Unsqueeze(., .)"]]
            Shape_2[["Shape(.)"]]

            I_X -->|"FLOAT(a, b, c)"| Unsqueeze_1
            Constant_0 -->|"INT64(1)"| Unsqueeze_1
            Unsqueeze_1 -->|"FLOAT(a, 1, b, c)"| Shape_2

            O_Y(["Y INT64(4)"])
            Shape_2 --> O_Y

            class I_X,I_axes,O_Y ioNode
            class Constant_0 constNode
            class Unsqueeze_1,Shape_2 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b, c)"])
            I_indices(["indices INT64(3)"])

            Constant_0[["Constant() -#gt; indices"]]
            Shape_1[["Shape(.)"]]
            Gather_2[["Gather(., .)"]]

            I_X -->|"FLOAT(a, b, c)"| Shape_1
            Shape_1 -->|"INT64(3)"| Gather_2
            Constant_0 -->|"INT64(3)"| Gather_2

            O_Y(["Y INT64(3)"])
            Gather_2 --> O_Y

            class I_X,I_indices,O_Y ioNode
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
        if node.op_type != "Unsqueeze" or node.domain != "":
            return self.none()

        # Unsqueeze axes must be a constant second input (opset >= 13 form).
        if len(node.input) < 2 or not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        axes = g.get_computed_constant(node.input[1])
        if axes is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # Need the rank of the Unsqueeze input to normalise negative axes.
        if not g.has_rank(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        # Unsqueeze output must only be consumed by one Shape node.
        if g.is_used_more_than_once(node.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape_node = g.next_node(node.output[0])
        if shape_node is None or shape_node.op_type != "Shape" or shape_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, shape_node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", unsq_node: NodeProto, shape_node: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        axes = g.get_computed_constant(unsq_node.input[1])
        input_rank = g.get_rank(unsq_node.input[0])
        output_rank = input_rank + len(axes)
        axes_norm = set(int(a) % output_rank for a in axes.flatten())

        # All positions in the unsqueezed shape that are NOT inserted by Unsqueeze,
        # remapped to coordinate space of the original tensor X.
        non_axes = [i for i in range(output_rank) if i not in axes_norm]
        remapped = [i - sum(1 for a in sorted(axes_norm) if a < i) for i in non_axes]
        new_indices = np.array(remapped, dtype=np.int64)

        new_indices_name = g.make_initializer(
            "", new_indices, source=f"{self.__class__.__name__}.apply.remapped_indices"
        )

        # The replacement Gather will output a shorter vector than the original
        # Shape(Unsqueeze(...)) node.  The graph builder caches the "value shape"
        # (symbolic element values) of every result.  If the cache already holds
        # the old (longer) value for shape_node.output[0], the assertion inside
        # set_value_shape would fire when the new Gather node is inserted.
        # Clearing the stale entry lets the graph builder recompute the correct
        # value shape from the replacement nodes.
        stale = shape_node.output[0]
        if stale in g.builder._known_value_shape:
            del g.builder._known_value_shape[stale]

        new_shape_name = g.unique_name(f"{self.__class__.__name__}_{shape_node.output[0]}")
        new_shape_node = g.make_node(
            "Shape",
            [unsq_node.input[0]],
            [new_shape_name],
            name=f"{self.__class__.__name__}--{shape_node.name}",
            doc_string=shape_node.doc_string,
        )

        new_gather_node = g.make_node(
            "Gather",
            [new_shape_name, new_indices_name],
            shape_node.output,
            name=f"{self.__class__.__name__}--{unsq_node.name}",
            doc_string=unsq_node.doc_string,
        )

        return [new_shape_node, new_gather_node]
