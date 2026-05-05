import inspect
from typing import List, Optional, Tuple
import numpy as np
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class GatherShapePattern(PatternOptimization):
    """
    Simplifies ``Gather(Shape(X), indices)`` into ``Shape(X, start=s, end=e)``
    when *indices* is a constant 1-D ``int64`` array that forms a contiguous
    ascending range ``[s, s+1, ..., e-1]``.

    This avoids materialising the full shape vector only to slice it immediately
    afterwards.  The Shape node may already carry ``start`` / ``end`` attributes
    (ONNX opset ≥ 15); those are taken into account when computing the absolute
    indices in ``X``'s dimension space.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b, c, d)"])
            C_idx(["idx INT64[3]"])

            Shape_0[["Shape(.)"]]
            Gather_1[["Gather(., [0, 1, 2])"]]

            I_X -->|"FLOAT(a, b, c, d)"| Shape_0
            C_idx -->|"INT64(3)"| Gather_1
            Shape_0 -->|"INT64(4)"| Gather_1

            O_Y(["Y INT64(3)"])
            Gather_1 --> O_Y

            class I_X,O_Y ioNode
            class C_idx constNode
            class Shape_0,Gather_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b, c, d)"])

            Shape_0[["Shape(., start=0, end=3)"]]

            I_X -->|"FLOAT(a, b, c, d)"| Shape_0

            O_Y(["Y INT64(3)"])
            Shape_0 --> O_Y

            class I_X,O_Y ioNode
            class Shape_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    @staticmethod
    def _get_attr_i(node: NodeProto, name: str, default):
        """Returns the integer value of a named attribute, or *default* if absent."""
        for attr in node.attribute:
            if attr.name == name:
                return int(attr.i)
        return default

    @staticmethod
    def _clamp_dim(value: int, rank: int) -> int:
        """Normalises a possibly negative dimension index and clamps it to ``[0, rank]``."""
        if value < 0:
            value += rank
        return max(0, min(value, rank))

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Gather" or node.domain != "":
            return self.none()

        # Only axis=0 is meaningful here (Shape outputs a 1-D vector).
        if self._get_attr_i(node, "axis", 0) != 0:
            return self.none(node, inspect.currentframe().f_lineno)

        # Indices must be a constant 0-D (scalar) or 1-D int64 array.
        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        indices = g.get_computed_constant(node.input[1])
        if (
            indices is None
            or indices.ndim not in (0, 1)
            or (indices.ndim == 1 and len(indices) < 1)
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if indices.dtype != np.int64:
            return self.none(node, inspect.currentframe().f_lineno)

        # Indices must form a strictly contiguous ascending range (only for 1-D).
        if indices.ndim == 1 and len(indices) > 1 and not np.all(np.diff(indices) == 1):
            return self.none(node, inspect.currentframe().f_lineno)

        # Data must come from a Shape node.
        shape_node = g.node_before(node.input[0])
        if shape_node is None or shape_node.op_type != "Shape" or shape_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        x_name = shape_node.input[0]

        # If the Shape node carries negative start/end attrs, or the gather
        # indices are negative, we need the rank of X to normalise them.
        shape_start_raw = self._get_attr_i(shape_node, "start", None)
        shape_end_raw = self._get_attr_i(shape_node, "end", None)
        has_negative_shape_attr = (shape_start_raw is not None and shape_start_raw < 0) or (
            shape_end_raw is not None and shape_end_raw < 0
        )
        has_negative_index = (
            bool(indices.item() < 0) if indices.ndim == 0 else bool(indices[0] < 0)
        )
        if has_negative_shape_attr or has_negative_index:
            if not g.has_rank(x_name):
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [shape_node, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        shape_node: NodeProto,
        gather_node: NodeProto,
    ) -> List[NodeProto]:
        indices = g.get_computed_constant(gather_node.input[1])
        x_name = shape_node.input[0]
        is_scalar = indices.ndim == 0

        # Resolve the Shape node's start/end against the rank of X.
        rank = g.get_rank(x_name) if g.has_rank(x_name) else None
        shape_start_raw = self._get_attr_i(shape_node, "start", None)
        shape_end_raw = self._get_attr_i(shape_node, "end", None)

        s0 = shape_start_raw if shape_start_raw is not None else 0
        if rank is not None:
            e0 = shape_end_raw if shape_end_raw is not None else rank
            s0 = self._clamp_dim(s0, rank)
            e0 = self._clamp_dim(e0, rank)
            L = e0 - s0
        else:
            # No rank available; shape attrs must be non-negative (guaranteed by match).
            if shape_end_raw is not None:
                e0 = shape_end_raw
                L = e0 - s0
            else:
                L = None  # unknown

        # Normalise gather indices (may be negative when rank is known, per match guard).
        gather_first = indices.item() if is_scalar else int(indices[0])
        gather_last = indices.item() if is_scalar else int(indices[-1])
        if gather_first < 0:
            gather_first += L
        if gather_last < 0:
            gather_last += L
        new_start = s0 + gather_first
        new_end = s0 + gather_last + 1

        if is_scalar:
            # Gather with a 0-D scalar index returns a scalar (0-D output); the
            # replacement Shape(X, start, end) returns a 1-D tensor of length 1.
            # Insert Squeeze(axes=[0]) to recover the scalar.
            shape_out_name = g.unique_name(
                f"{self.__class__.__name__}_{gather_node.output[0]}_1d"
            )
            new_shape_node = g.make_node(
                "Shape",
                [x_name],
                [shape_out_name],
                start=new_start,
                end=new_end,
                name=f"{self.__class__.__name__}--{gather_node.name}",
                doc_string=gather_node.doc_string,
            )
            axes_name = g.make_initializer(
                "",
                np.array([0], dtype=np.int64),
                source=f"{self.__class__.__name__}.apply.squeeze_axes",
            )
            squeeze_node = g.make_node(
                "Squeeze",
                [shape_out_name, axes_name],
                gather_node.output,
                name=f"{self.__class__.__name__}--squeeze--{gather_node.name}",
                doc_string=gather_node.doc_string,
            )
            if g.is_used_more_than_once(shape_node.output[0]):
                return [shape_node, new_shape_node, squeeze_node]
            return [new_shape_node, squeeze_node]

        new_shape_node = g.make_node(
            "Shape",
            [x_name],
            gather_node.output,
            start=new_start,
            end=new_end,
            name=f"{self.__class__.__name__}--{gather_node.name}",
            doc_string=gather_node.doc_string,
        )

        if g.is_used_more_than_once(shape_node.output[0]):
            return [shape_node, new_shape_node]
        return [new_shape_node]


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

    @staticmethod
    def _resolve_shape_start_end(shape_node: NodeProto, output_rank: int) -> tuple:
        """Returns the effective (start, end) range from a Shape node's attributes.

        Reads ``start`` and ``end`` attributes (both optional, defaulting to
        ``0`` and ``output_rank`` respectively), normalises negative values
        against ``output_rank``, and clamps the result to ``[0, output_rank]``.
        """
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
        return shape_start, shape_end

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
        shape_start, shape_end = self._resolve_shape_start_end(shape_node, output_rank)
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
        shape_start, shape_end = self._resolve_shape_start_end(shape_node, output_rank)

        # Reconstruct Shape(Unsqueeze(X, axes))[shape_start:shape_end] as a
        # Concat of Shape(X, start, end) slices interleaved with constant [1]
        # tensors.  Only output positions that fall inside [shape_start, shape_end)
        # contribute to the result: a run of original (non-inserted) dimensions
        # is clipped to the requested window and extracted via a ranged Shape node,
        # while each inserted axis within the window contributes a constant [1].
        #
        # Example: X shape (a, b, c), axes=[1], shape_start=0, shape_end=4
        #   Shape(X, start=0, end=1)  →  [a]
        #   const [1]
        #   Shape(X, start=1, end=3)  →  [b, c]
        #   Concat([a], [1], [b, c], axis=0)  →  [a, 1, b, c]
        concat_inputs = []
        extra_nodes = []
        x_cursor = 0  # next un-consumed position in X's dimension space
        out_cursor = 0  # next un-consumed position in the unsqueezed output

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


class ShapeTransposePattern(PatternOptimization):
    """
    Replaces ``Shape(Transpose(X, perm))`` by ``Gather(Shape(X), perm_indices)``
    so that the expensive Transpose on the full data tensor is avoided.

    The key observation is that the shape of ``Transpose(X, perm)`` is simply
    a permuted view of the shape of ``X``.  The permutation indices are known
    at optimisation time (they are an attribute of the Transpose node), so we
    can extract the desired dimensions directly from ``Shape(X)`` using a
    ``Gather`` with the (sub-)permutation as the index tensor.

    For ``X`` of shape ``(a, b, c)`` and ``perm=[2, 0, 1]`` the transformation
    is::

        # Before
        xt = Transpose(X, perm=[2, 0, 1])   # (c, a, b)
        Y  = Shape(xt)                       # [c, a, b]

        # After
        sx   = Shape(X)                      # [a, b, c]
        perm = Initializer([2, 0, 1])
        Y    = Gather(sx, perm, axis=0)      # [c, a, b]

    Shape's optional ``start``/``end`` attributes are respected: the
    permutation slice ``perm[start:end]`` is used as the Gather indices.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b, c)"])

            Transpose_0[["Transpose(., perm=[2, 0, 1])"]]
            Shape_1[["Shape(.)"]]

            I_X -->|"FLOAT(a, b, c)"| Transpose_0
            Transpose_0 -->|"FLOAT(c, a, b)"| Shape_1

            O_Y(["Y INT64(3)"])
            Shape_1 --> O_Y

            class I_X,O_Y ioNode
            class Transpose_0,Shape_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, b, c)"])
            C_perm(["perm INT64[2, 0, 1]"])

            Shape_s[["Shape(.)"]]
            Gather_0[["Gather(., ., axis=0)"]]

            I_X -->|"FLOAT(a, b, c)"| Shape_s
            Shape_s -->|"INT64(3)"| Gather_0
            C_perm -->|"INT64(3)"| Gather_0

            O_Y(["Y INT64(3)"])
            Gather_0 --> O_Y

            class I_X,O_Y ioNode
            class C_perm constNode
            class Shape_s,Gather_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 0):
        super().__init__(verbose, priority)

    @staticmethod
    def _resolve_shape_start_end(shape_node: NodeProto, output_rank: int) -> Tuple[int, int]:
        """Returns the effective (start, end) range from a Shape node's attributes.

        Reads ``start`` and ``end`` attributes (both optional, defaulting to
        ``0`` and ``output_rank`` respectively), normalises negative values
        against ``output_rank``, and clamps the result to ``[0, output_rank]``.
        """
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
        return shape_start, shape_end

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "Shape" or node.domain != "":
            return self.none()

        tr_node = g.node_before(node.input[0])
        if tr_node is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if tr_node.op_type != "Transpose" or tr_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # The permutation must be present as an attribute.
        perm_attr = next((a for a in tr_node.attribute if a.name == "perm"), None)
        if perm_attr is None:
            return self.none(node, inspect.currentframe().f_lineno)

        perm = list(perm_attr.ints)
        rank = len(perm)

        # Guard against a degenerate (empty) perm slice after applying start/end.
        shape_start, shape_end = self._resolve_shape_start_end(node, rank)
        if shape_start >= shape_end:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, [tr_node, node], self.apply, insert_at=node)

    def apply(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        tr_node: NodeProto,
        shape_node: NodeProto,
    ) -> List[NodeProto]:
        perm = list(next(a for a in tr_node.attribute if a.name == "perm").ints)
        rank = len(perm)

        shape_start, shape_end = self._resolve_shape_start_end(shape_node, rank)
        perm_subset = perm[shape_start:shape_end]

        result = []

        # Re-emit the Transpose if its output is still needed by other consumers.
        if g.is_used_more_than_once(tr_node.output[0]):
            result.append(tr_node)

        # Shape(X) – no start/end needed since we index via Gather.
        shape_out = g.unique_name(f"{self.__class__.__name__}_{shape_node.output[0]}_sx")
        result.append(
            g.make_node(
                "Shape", [tr_node.input[0]], [shape_out], name=f"{self.__class__.__name__}--shape"
            )
        )

        # Gather indices = the (sub-)permutation.
        perm_init = g.make_initializer(
            "",
            np.array(perm_subset, dtype=np.int64),
            source=f"{self.__class__.__name__}.apply.perm",
        )
        result.append(
            g.make_node(
                "Gather",
                [shape_out, perm_init],
                shape_node.output,
                axis=0,
                name=f"{self.__class__.__name__}--{shape_node.name}",
                doc_string=shape_node.doc_string,
            )
        )
        return result
