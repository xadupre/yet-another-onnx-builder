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
    Replaces ``Shape(Unsqueeze(X, axes))`` by a ``Concat`` of
    ``Shape(X, start=s, end=e)`` slices interleaved with constant ``[1]``
    tensors at the inserted axis positions.

    The key observation is that ``Shape(Unsqueeze(X, axes))`` produces a
    shape vector that is identical to ``Shape(X)`` with ``1`` entries
    inserted at the ``axes`` positions.  By splitting ``Shape(X)`` into
    segments and concatenating them with the constant ``1`` values, the
    Unsqueeze on the (potentially large) data tensor is avoided entirely
    while the output shape vector remains bit-for-bit identical.

    For ``X`` of shape ``(a, b, c)`` and ``axes=[1]`` the transformation
    is::

        # Before
        xu = Unsqueeze(X, [1])        # (a, 1, b, c)
        Y  = Shape(xu)                # [a, 1, b, c]

        # After
        s0 = Shape(X, start=0, end=1) # [a]
        s1 = Shape(X, start=1, end=3) # [b, c]
        Y  = Concat([s0, [1], s1])    # [a, 1, b, c]

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
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b, c)"])
            C1(["const INT64[1]"])

            Shape_s0[["Shape(., start=0, end=1)"]]
            Shape_s1[["Shape(., start=1, end=3)"]]
            Concat_2[["Concat(axis=0)"]]

            I_X -->|"FLOAT(a, b, c)"| Shape_s0
            I_X -->|"FLOAT(a, b, c)"| Shape_s1
            Shape_s0 -->|"INT64(1)"| Concat_2
            C1 -->|"INT64(1)"| Concat_2
            Shape_s1 -->|"INT64(2)"| Concat_2

            O_Y(["Y INT64(4)"])
            Concat_2 --> O_Y

            class I_X,O_Y ioNode
            class C1 constNode
            class Shape_s0,Shape_s1,Concat_2 opNode
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
        if axes is None or len(axes) == 0:
            return self.none(node, inspect.currentframe().f_lineno)

        # Need the rank of the Unsqueeze input to normalise negative axes.
        if not g.has_rank(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        input_rank = g.get_rank(node.input[0])

        # Find a Shape consumer for the Unsqueeze output.  Even when the
        # Unsqueeze output is consumed by other nodes as well, we can still
        # eliminate the Shape(Unsqueeze(…)) sub-expression; apply() will
        # preserve the Unsqueeze itself if it is still needed.
        shape_node = next(
            (n for n in g.next_nodes(node.output[0]) if n.op_type == "Shape" and n.domain == ""),
            None,
        )
        if shape_node is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # Shape (opset ≥ 15) may carry start/end attributes that select a
        # sub-range of the output shape.  Resolve them now so we can guard
        # against a degenerate (empty) range, which would require a 0-input
        # Concat (invalid ONNX).
        output_rank = input_rank + len(axes)
        shape_start = next(
            (int(attr.i) for attr in shape_node.attribute if attr.name == "start"), 0
        )
        shape_end = next(
            (int(attr.i) for attr in shape_node.attribute if attr.name == "end"), output_rank
        )
        if shape_start < 0:
            shape_start += output_rank
        if shape_end < 0:
            shape_end += output_rank
        shape_start = max(0, min(shape_start, output_rank))
        shape_end = max(0, min(shape_end, output_rank))
        if shape_start >= shape_end:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [node, shape_node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        unsq_node: NodeProto,
        shape_node: NodeProto,
    ) -> List[NodeProto]:
        axes = g.get_computed_constant(unsq_node.input[1])
        input_rank = g.get_rank(unsq_node.input[0])
        output_rank = input_rank + len(axes)
        sorted_axes = sorted(int(a) % output_rank for a in axes.flatten())

        # Resolve any start/end attributes on shape_node.  The Shape operator
        # (opset ≥ 15) uses these to select a sub-range of the output shape
        # instead of returning all dimensions.  Defaults match the full range.
        shape_start = next(
            (int(attr.i) for attr in shape_node.attribute if attr.name == "start"), 0
        )
        shape_end = next(
            (int(attr.i) for attr in shape_node.attribute if attr.name == "end"), output_rank
        )
        if shape_start < 0:
            shape_start += output_rank
        if shape_end < 0:
            shape_end += output_rank
        shape_start = max(0, min(shape_start, output_rank))
        shape_end = max(0, min(shape_end, output_rank))

        # Reconstruct Shape(Unsqueeze(X, axes))[shape_start:shape_end] as a
        # Concat of Shape(X, start, end) slices interleaved with constant [1]
        # tensors, restricted to positions that fall inside [shape_start, shape_end).
        #
        # Example: X shape (a, b, c), axes=[1], shape_start=0, shape_end=4
        #   Shape(X, start=0, end=1)  →  [a]
        #   const [1]
        #   Shape(X, start=1, end=3)  →  [b, c]
        #   Concat([a], [1], [b, c], axis=0)  →  [a, 1, b, c]
        concat_inputs = []
        extra_nodes = []
        x_cursor = 0  # running position in X's dimension space
        out_cursor = 0  # running position in the unsqueezed output

        for axis in sorted_axes:
            # Run of original dims: out positions [out_cursor, axis),
            # corresponding to X dims [x_cursor, x_cursor + run_len).
            run_len = axis - out_cursor
            if run_len > 0:
                seg_out_start = max(out_cursor, shape_start)
                seg_out_end = min(axis, shape_end)
                if seg_out_start < seg_out_end:
                    x_seg_start = x_cursor + (seg_out_start - out_cursor)
                    x_seg_end = x_cursor + (seg_out_end - out_cursor)
                    seg_name = g.unique_name(
                        f"{self.__class__.__name__}_{shape_node.output[0]}_s{x_seg_start}"
                    )
                    extra_nodes.append(
                        g.make_node(
                            "Shape",
                            [unsq_node.input[0]],
                            [seg_name],
                            start=x_seg_start,
                            end=x_seg_end,
                            name=f"{self.__class__.__name__}--shape{x_seg_start}",
                        )
                    )
                    concat_inputs.append(seg_name)
                x_cursor += run_len
            # Inserted axis at out position `axis`.
            if shape_start <= axis < shape_end:
                one_name = g.make_initializer(
                    "",
                    np.array([1], dtype=np.int64),
                    source=f"{self.__class__.__name__}.apply.one",
                )
                concat_inputs.append(one_name)
            out_cursor = axis + 1

        # Trailing run of original dims: out positions [out_cursor, output_rank),
        # corresponding to X dims [x_cursor, input_rank).
        if x_cursor < input_rank:
            seg_out_start = max(out_cursor, shape_start)
            seg_out_end = min(output_rank, shape_end)
            if seg_out_start < seg_out_end:
                x_seg_start = x_cursor + (seg_out_start - out_cursor)
                x_seg_end = x_cursor + (seg_out_end - out_cursor)
                seg_name = g.unique_name(
                    f"{self.__class__.__name__}_{shape_node.output[0]}_s{x_seg_start}"
                )
                extra_nodes.append(
                    g.make_node(
                        "Shape",
                        [unsq_node.input[0]],
                        [seg_name],
                        start=x_seg_start,
                        end=x_seg_end,
                        name=f"{self.__class__.__name__}--shape_tail",
                    )
                )
                concat_inputs.append(seg_name)

        concat_node = g.make_node(
            "Concat",
            concat_inputs,
            shape_node.output,
            axis=0,
            name=f"{self.__class__.__name__}--{shape_node.name}",
            doc_string=shape_node.doc_string,
        )

        # Both unsq_node and shape_node are in match.nodes and will be removed.
        # If the Unsqueeze output is still consumed by nodes other than the Shape
        # we just eliminated (or is a graph output / used by a subgraph), we must
        # re-emit the Unsqueeze so those consumers remain valid.
        result = []
        if g.is_used_more_than_once(unsq_node.output[0]):
            result.append(unsq_node)
        result.extend(extra_nodes)
        result.append(concat_node)
        return result
