import inspect
from typing import List, Optional, Sequence, Tuple
import numpy as np
from onnx import NodeProto, TensorProto
from ...helpers.onnx_helper import tensor_dtype_to_np_dtype
from ..patterns_api import MatchResult, PatternOptimization
from ..patterns.onnx_attention import FunctionAttentionPattern, FunctionAttentionGQAPattern
from ..patterns.onnx_rotary import FunctionHalfRotaryEmbeddingPattern


class ContribRotaryEmbeddingPattern(PatternOptimization):
    """
    Very similar to
    :class:`yobx.xoptim.patterns.onnx_rotary.RotaryEmbeddingPattern`.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, 2, c, 2*e)"])
            I_m1(["m1 FLOAT(1, 1, c, e)"])
            I_m2(["m2 FLOAT(1, 1, c, e)"])

            Concat_0[["Concat(., ., axis=-1)"]]
            Concat_1[["Concat(., ., axis=-1)"]]
            HalfRotaryEmbedding_2[["intermediate.HalfRotaryEmbedding(., ., .)"]]

            I_m2 -->|"FLOAT(1, 1, c, e)"| Concat_0
            I_m1 -->|"FLOAT(1, 1, c, e)"| Concat_1
            I_X -->|"FLOAT(a, 2, c, 2*e)"| HalfRotaryEmbedding_2
            Concat_0 --> HalfRotaryEmbedding_2
            Concat_1 --> HalfRotaryEmbedding_2

            O_Y(["Y FLOAT(a, b, c, 2*e)"])
            HalfRotaryEmbedding_2 --> O_Y

            class I_X,I_m1,I_m2,O_Y ioNode
            class Concat_0,Concat_1,HalfRotaryEmbedding_2 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_X(["X FLOAT(a, 2, c, 2*e)"])
            I_m1(["m1 FLOAT(1, 1, c, e)"])
            I_m2(["m2 FLOAT(1, 1, c, e)"])

            Squeeze_0[["Squeeze(., [0, 1])"]]
            Squeeze_1[["Squeeze(., [0, 1])"]]
            Shape_2[["Shape(., end=1, start=0)"]]
            Shape_3[["Shape(., end=3, start=2)"]]
            Squeeze_4[["Squeeze(.)"]]
            Range_5[["Range(0, ., 1)"]]
            Concat_6[["Concat(., [1], axis=0)"]]
            Expand_7[["Expand(., .)"]]
            RotaryEmbedding_8[["com.microsoft.RotaryEmbedding(., ., ., .)"]]

            I_m2 -->|"FLOAT(1, 1, c, e)"| Squeeze_0
            I_m1 -->|"FLOAT(1, 1, c, e)"| Squeeze_1
            I_X -->|"FLOAT(a, 2, c, 2*e)"| Shape_2
            I_X -->|"FLOAT(a, 2, c, 2*e)"| Shape_3
            Shape_3 -->|"INT64(1)"| Squeeze_4
            Squeeze_4 -->|"INT64()"| Range_5
            Shape_2 -->|"INT64(1)"| Concat_6
            Range_5 -->|"INT64(NEWDIM_range_0)"| Expand_7
            Concat_6 -->|"INT64(2)"| Expand_7
            I_X -->|"FLOAT(a, 2, c, 2*e)"| RotaryEmbedding_8
            Expand_7 -->|"INT64(a, NEWDIM_range_0)"| RotaryEmbedding_8
            Squeeze_0 -->|"FLOAT(c, e)"| RotaryEmbedding_8
            Squeeze_1 -->|"FLOAT(c, e)"| RotaryEmbedding_8

            O_Y(["Y FLOAT(a, b, c, 2*e)"])
            RotaryEmbedding_8 --> O_Y

            class I_X,I_m1,I_m2,O_Y ioNode
            class Squeeze_0,Squeeze_1,Shape_2,Shape_3,Squeeze_4,Range_5,Concat_6,Expand_7 opNode
            class RotaryEmbedding_8 opNode
    """

    _operator_name = FunctionHalfRotaryEmbeddingPattern._operator_name
    _domain_name = FunctionHalfRotaryEmbeddingPattern._domain_name

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)
        self._info = []

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != self._operator_name or node.domain != self._domain_name:
            return self.none()
        if not g.has_shape(node.input[0]) or g.get_rank(node.input[0]) != 4:
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(node.input[1]) or not g.has_shape(node.input[2]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape_cos = g.get_shape(node.input[1])
        shape_sin = g.get_shape(node.input[2])
        if shape_cos != shape_sin:
            return self.none(node, inspect.currentframe().f_lineno)
        if len(shape_cos) != 4:
            return self.none(node, inspect.currentframe().f_lineno)
        if shape_cos[1] != 1 or shape_sin[1] != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        # if shape_cos[0] != 1 or shape_sin[0] != 1:
        #    return self.none(node, inspect.currentframe().f_lineno)
        # if shape_cos[0] != 1 or shape_sin[0] != 1:
        # batch size is not 1 because position_ids was involved in the
        # computation of cos/sin caches.
        #    return self.none(node, inspect.currentframe().f_lineno)

        concat_cos = g.node_before(node.input[1])
        if concat_cos is None or concat_cos.op_type != "Concat" or concat_cos.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if concat_cos.input[0] != concat_cos.input[1]:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_attribute(concat_cos, "axis").i != -1:
            return self.none(node, inspect.currentframe().f_lineno)

        concat_sin = g.node_before(node.input[2])
        if concat_sin is None or concat_sin.op_type != "Concat" or concat_sin.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if concat_sin.input[0] != concat_sin.input[1]:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_attribute(concat_sin, "axis").i != -1:
            return self.none(node, inspect.currentframe().f_lineno)

        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        # If cos_cache[-1] + sin_cache[-1] == X.shape[-1],
        # then there is no split before.
        split_node = g.node_before(node.input[0])
        if split_node is None or split_node.op_type != "Split" or split_node.domain != "":
            if not g.has_shape(concat_cos.input[0]) or not g.has_shape(concat_sin.input[0]):
                return self.none(node, inspect.currentframe().f_lineno)
            cos_shape = g.get_shape(concat_cos.input[0])
            sin_shape = g.get_shape(concat_sin.input[0])
            input_shape = g.get_shape(node.input[0])
            if g.builder.evaluate_dimension_equality_with_constraints(
                input_shape[-1], cos_shape[-1], "+", sin_shape[-1]
            ):
                shape = g.get_shape_renamed(node.input[0])
                self._info.append((node.input[0], shape))
                if not isinstance(shape[1], int):
                    # Number of heads is not fixed"
                    return self.none(
                        node,
                        inspect.currentframe().f_lineno,
                        msg=lambda: (
                            f"number of head (shape[1]) is not fixed for {node.input[0]!r}, "
                            f"{shape=}, renamed shape={g.get_shape_renamed(node.input[0])}"
                        ),
                    )
                # No split before, no concat after but there could be still position ids
                return self._match_last_part(
                    g,
                    concat_cos,
                    concat_sin,
                    None,
                    node,
                    None,
                    comment="path with no split before, no concat after",
                )

        if split_node is None or split_node.op_type != "Split" or split_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(split_node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape_input = g.get_shape(split_node.input[0])
        if not isinstance(shape_input[1], int):
            # Not a fixed number of heads.
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(split_node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(split_node.input[1])
        if cst.shape != (2,):
            return self.none(node, inspect.currentframe().f_lineno)

        next_nodes = g.next_nodes(node.output[0])
        if len(next_nodes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        concat_node = next_nodes[0]
        if concat_node.op_type != "Concat" or concat_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if split_node.output[1] != concat_node.input[1]:
            return self.none(node, inspect.currentframe().f_lineno)
        axis = g.get_attribute(concat_node, "axis").i
        if axis != -1:
            return self.none(node, inspect.currentframe().f_lineno)

        input_name = node.input[0] if split_node is None else split_node.input[0]
        shape = g.get_shape(input_name)
        self._info.append((input_name, shape))
        if not isinstance(shape[1], int):
            # Number of heads is not fixed"
            return self.none(node, inspect.currentframe().f_lineno)

        return self._match_last_part(
            g,
            concat_cos,
            concat_sin,
            split_node,
            node,
            concat_node,
            comment="path with split before, concat after",
        )

    def _match_last_part(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        concat_cos: NodeProto,
        concat_sin: NodeProto,
        split_node: Optional[NodeProto],
        node: NodeProto,
        concat_node: Optional[NodeProto],
        comment: str,
    ) -> Optional[MatchResult]:
        # Finally, we need to check if position_ids exists or it is given
        # a default value.
        common = self._find_common_ancestor(g, concat_cos, concat_sin)
        if common is not None and not common:
            # cos/sin are switched. The pattern cannot match.
            return self.none(node, inspect.currentframe().f_lineno)
        if (
            common
            and common[0].op_type == "Mul"
            and {"Sin", "Cos"} & set(n.op_type for n in common)
        ):
            # pattern FunctionCosSinCache has yet to be triggered first.
            return self.none(node, inspect.currentframe().f_lineno)

        if (
            common
            and common[0].op_type.startswith("CosSinCache")
            and not common[0].op_type.startswith("CosSinCacheWithRange")
            and common[0].domain == self._domain_name
        ):
            # Finally, we need to check if position_ids exists or if it is given
            # a default value.
            cos_sin = common[0]
            if not g.has_shape(cos_sin.input[0]) or not g.has_shape(cos_sin.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            expand_node = g.node_before(cos_sin.input[1])
            if expand_node is not None:
                # position_ids is expanded first
                shape_expand = g.builder.value_as_shape(expand_node.input[1])
                if shape_expand is None or len(shape_expand) != 3 or shape_expand[1:] != (1, 1):
                    # maybe position_ids is not given
                    return self.none(
                        node,
                        inspect.currentframe().f_lineno,
                        msg=lambda: (
                            f"op_type={expand_node.op_type!r} name={expand_node.input[1]!r} "
                            f"shape_expand={shape_expand}"
                        ),
                    )
                if not g.has_shape(expand_node.input[0]):
                    return self.none(node, inspect.currentframe().f_lineno)
                wei_shape = g.get_shape(expand_node.input[0])
                if wei_shape[0] != 1:
                    return self.none(node, inspect.currentframe().f_lineno)

            position_ids_shape = g.get_shape_renamed(cos_sin.input[0])
            weights_shape = g.get_shape_renamed(cos_sin.input[1])
            if (
                len(position_ids_shape) != 2
                or len(weights_shape) != 3
                or (position_ids_shape[0] != weights_shape[0] and weights_shape[0] != 1)
            ):
                return self.none(node, inspect.currentframe().f_lineno)

            # Then we need to add those nodes to the matched nodes.
            return MatchResult(
                self,
                [expand_node, concat_cos, concat_sin, split_node, node, concat_node, *common],
                self.apply,
                comment=f"{comment} / with CosSinCache",
            )

        return MatchResult(
            self,
            [None, concat_cos, concat_sin, split_node, node, concat_node],
            self.apply,
            insert_at=None if g.is_used_more_than_once(concat_cos.output[0]) else concat_node,
            comment=f"{comment} / without CosSinCache",
        )

    def _find_common_ancestor(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        concat_cos: NodeProto,
        concat_sin: NodeProto,
    ) -> Optional[List[NodeProto]]:
        anc_cos, anc_sin = concat_cos, concat_sin
        nodes = []
        for _it in range(5):
            cos_name, sin_name = anc_cos.input[0], anc_sin.input[0]
            anc_cos = g.node_before(cos_name)
            anc_sin = g.node_before(sin_name)
            if anc_cos is None or anc_sin is None:
                return self.none(concat_cos, inspect.currentframe().f_lineno)
            if anc_cos.input[0] == anc_sin.input[0] and id(anc_cos) == id(anc_sin):
                if len(anc_cos.output) == 2:
                    if (
                        cos_name != anc_cos.output[0]
                        or sin_name != anc_cos.output[1]
                        or not anc_cos.op_type.startswith("CosSinCache")
                    ):
                        # cos/sin were switched, the pattern should not match at all.
                        return []
                    nodes.append(anc_cos)
                    return nodes[::-1]
                # cos/sin are not produced the usual way (CosSinCache)
                return []
            nodes.extend([anc_cos, anc_sin])
        return self.none(concat_cos, inspect.currentframe().f_lineno)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        expand_node: Optional[NodeProto],
        concat_cos: NodeProto,
        concat_sin: NodeProto,
        split_node: NodeProto,
        half_node: NodeProto,
        concat_node: NodeProto,
        *prefix_nodes: Sequence[NodeProto],
    ) -> List[NodeProto]:
        if split_node is None:
            rotary_dim = None
            shape = g.get_shape_renamed(half_node.input[0])
            main_input = half_node.input[0]
            main_output = half_node.output[0]
        else:
            cst = g.get_computed_constant(split_node.input[1])
            rotary_dim = int(cst[0])
            shape = g.get_shape_renamed(split_node.input[0])
            main_input = split_node.input[0]
            main_output = concat_node.output[0]

        assert isinstance(shape[1], int), (
            f"Number of heads is not fixed, shape("
            f"{split_node.input[0] if split_node is not None else half_node.input[0]}"
            f")={shape}, info={self._info}"
        )
        num_heads = shape[1]

        used_twice_cos = g.is_used_more_than_once(concat_cos.output[0])
        used_twice_sin = g.is_used_more_than_once(concat_sin.output[0])
        rotary_nodes = [expand_node, *prefix_nodes] if used_twice_cos or used_twice_sin else []
        if used_twice_cos:
            rotary_nodes.append(concat_cos)
        if used_twice_sin:
            rotary_nodes.append(concat_sin)

        batch_name = g.unique_name(f"{self.__class__.__name__}--{half_node.input[0]}--batch")
        zeroone = g.make_initializer(
            "", np.array([0, 1], dtype=np.int64), source=f"{self.__class__.__name__}.01"
        )
        one = g.make_initializer("", g.ONE, source=f"{self.__class__.__name__}.1")
        one_no_dim = g.make_initializer("", g.ONE_NO_DIM, source=f"{self.__class__.__name__}.1d")

        # position_ids
        zero_no_dim = g.make_initializer(
            "", g.ZERO_NO_DIM, source=f"{self.__class__.__name__}.0d"
        )

        added_nodes = []
        if prefix_nodes:
            assert prefix_nodes[0].op_type.startswith(
                "CosSinCache"
            ), f"Unexpected first node {prefix_nodes[0]}"
            cos_sin = prefix_nodes[0]
            position_ids = cos_sin.input[0]
            max_ids, max_ids_1, new_positions_ids, cos_out, sin_out, range_ids = [
                g.unique_name(f"{self.__class__.__name__}--{position_ids}") for i in range(6)
            ]
            zero = g.make_initializer("", g.ZERO, source=f"{self.__class__.__name__}.0")
            added_nodes = [
                g._make_node("ReduceMax", [position_ids], [max_ids], keepdims=0),
                g._make_node("Add", [max_ids, one_no_dim], [max_ids_1]),
                g._make_node("Range", [zero_no_dim, max_ids_1, one_no_dim], [range_ids]),
                g._make_node("Unsqueeze", [range_ids, zero], [new_positions_ids]),
                g._make_node(
                    cos_sin.op_type,
                    [
                        new_positions_ids,
                        expand_node.input[0] if expand_node is not None else cos_sin.input[1],
                    ],
                    [cos_out, sin_out],
                    domain=cos_sin.domain,
                ),
            ]
            cos_cur, sin_cur = cos_out, sin_out
            for i in range(1, len(prefix_nodes), 2):
                ncos, nsin = prefix_nodes[i : i + 2]
                if ncos.op_type == "Concat":
                    break
                rcos, rsin = [
                    g.unique_name(f"{self.__class__.__name__}--{position_ids}") for i in range(2)
                ]
                added_nodes.extend(
                    [
                        g._make_node(ncos.op_type, [cos_cur, *ncos.input[1:]], [rcos]),
                        g._make_node(ncos.op_type, [sin_cur, *nsin.input[1:]], [rsin]),
                    ]
                )
                if ncos.attribute:
                    added_nodes[-2].attribute.extend(ncos.attribute)
                if nsin.attribute:
                    added_nodes[-1].attribute.extend(nsin.attribute)
                cos_cur, sin_cur = rcos, rsin
            cos_input, sin_input = cos_cur, sin_cur
            range_nodes = []
        else:
            assert expand_node is None, f"Unexpected expand node {expand_node}"
            position_ids = g.unique_name(
                f"{self.__class__.__name__}--{half_node.input[0]}_position_ids"
            )
            seq_length = g.unique_name(
                f"{self.__class__.__name__}--{half_node.input[0]}_seq_length"
            )
            seq_length_squeezed = g.unique_name(
                f"{self.__class__.__name__}--{half_node.input[0]}_seqsq"
            )
            exp_shape = g.unique_name(f"{self.__class__.__name__}_{half_node.input[0]}_pshape")
            flat_pids = g.unique_name(
                f"{self.__class__.__name__}--{half_node.input[0]}_flat_pids"
            )
            cos_input, sin_input = concat_cos.input[0], concat_sin.input[0]
            range_nodes = [
                g._make_node("Shape", [main_input], [batch_name], start=0, end=1),
                g._make_node("Shape", [main_input], [seq_length], start=2, end=3),
                g._make_node("Squeeze", [seq_length], [seq_length_squeezed]),
                g._make_node(
                    "Range", [zero_no_dim, seq_length_squeezed, one_no_dim], [flat_pids]
                ),
                g._make_node("Concat", [batch_name, one], [exp_shape], axis=0),
                g._make_node("Expand", [flat_pids, exp_shape], [position_ids]),
            ]

        cos_name = g.unique_name(f"{self.__class__.__name__}--{half_node.input[1]}")
        sin_name = g.unique_name(f"{self.__class__.__name__}--{half_node.input[2]}")
        rotary_nodes.extend(
            [
                *added_nodes,
                g._make_node("Squeeze", [cos_input, zeroone], [cos_name]),
                g._make_node("Squeeze", [sin_input, zeroone], [sin_name]),
                *range_nodes,
            ]
        )
        rotary_nodes = [n for n in rotary_nodes if n]
        for node in rotary_nodes:
            if not node.name:
                node.name = g.builder.unique_node_name(
                    f"{self.__class__.__name__}--{half_node.name}"
                )

        kwargs = {} if rotary_dim is None else {"rotary_embedding_dim": rotary_dim}
        rotary_node = g.make_node(
            "RotaryEmbedding",
            [main_input, position_ids, cos_name, sin_name],
            [main_output],
            name=f"{self.__class__.__name__}--{half_node.name}",
            num_heads=num_heads,
            domain="com.microsoft",
            **kwargs,
        )
        rotary_nodes.append(rotary_node)
        return rotary_nodes


class ContribRotaryEmbedding3DPattern(PatternOptimization):
    """
    Extension to
    :class:`yobx.xoptim.patterns_ort.llm_optim.ContribRotaryEmbeddingPattern`,
    turn the operator into a 3D operator including the transpose.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            icrote_m2x2(["ContribRotaryEmbeddingPattern--m2x2 FLOAT(NEWDIM_range, 2)"])
            I_position_ids(["position_ids INT64(a, e)"])
            icrote_m1x2(["ContribRotaryEmbeddingPattern--m1x2 FLOAT(NEWDIM_range, 2)"])
            I_X(["X FLOAT(a, c, 2, d)"])

            Transpose_0[["Transpose(., perm=[0, 2, 1, 3])"]]
            RotaryEmbedding_1[["com.microsoft.RotaryEmbedding(., ., ., .)"]]

            I_X -->|"FLOAT(a, c, 2, d)"| Transpose_0
            Transpose_0 -->|"FLOAT(a, 2, c, d)"| RotaryEmbedding_1
            I_position_ids -->|"INT64(a, e)"| RotaryEmbedding_1
            icrote_m1x2 -->|"FLOAT(NEWDIM_range, 2)"| RotaryEmbedding_1
            icrote_m2x2 -->|"FLOAT(NEWDIM_range, 2)"| RotaryEmbedding_1

            O_Y(["Y FLOAT(a, b, c, d)"])
            RotaryEmbedding_1 --> O_Y

            class icrote_m2x2,I_position_ids,icrote_m1x2,I_X,O_Y ioNode
            class Transpose_0,RotaryEmbedding_1 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            icrote_m2x2(["ContribRotaryEmbeddingPattern--m2x2 FLOAT(NEWDIM_range, 2)"])
            I_position_ids(["position_ids INT64(a, e)"])
            icrote_m1x2(["ContribRotaryEmbeddingPattern--m1x2 FLOAT(NEWDIM_range, 2)"])
            I_X(["X FLOAT(a, c, 2, d)"])

            Reshape_0[["Reshape(., [0, 0, -1])"]]
            RotaryEmbedding_1[["com.microsoft.RotaryEmbedding(., ., ., .)"]]
            Shape_2[["Shape(., start=3)"]]
            Concat_3[["Concat([0, 0, -1], ., axis=0)"]]
            Reshape_4[["Reshape(., .)"]]
            Transpose_5[["Transpose(., perm=[0, 2, 1, 3])"]]

            I_X -->|"FLOAT(a, c, 2, d)"| Reshape_0
            Reshape_0 -->|"FLOAT(a, c, 2*d)"| RotaryEmbedding_1
            I_position_ids -->|"INT64(a, e)"| RotaryEmbedding_1
            icrote_m1x2 -->|"FLOAT(NEWDIM_range, 2)"| RotaryEmbedding_1
            icrote_m2x2 -->|"FLOAT(NEWDIM_range, 2)"| RotaryEmbedding_1
            I_X -->|"FLOAT(a, c, 2, d)"| Shape_2
            Shape_2 -->|"INT64(1)"| Concat_3
            RotaryEmbedding_1 -->|"FLOAT(a, c, 2*d)"| Reshape_4
            Concat_3 -->|"INT64(4)"| Reshape_4
            Reshape_4 -->|"FLOAT(a, c, 2, d)"| Transpose_5

            O_Y(["Y FLOAT(a, b, c, d)"])
            Transpose_5 --> O_Y

            class icrote_m2x2,I_position_ids,icrote_m1x2,I_X,O_Y ioNode
            class Reshape_0,RotaryEmbedding_1,Shape_2,Concat_3,Reshape_4,Transpose_5 opNode
    """

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "RotaryEmbedding" or node.domain != "com.microsoft":
            return self.none()
        transpose = g.node_before(node.input[0])
        if transpose is None or transpose.op_type != "Transpose" or transpose.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        perm = tuple(g.get_attribute(transpose, "perm").ints)
        if perm != (0, 2, 1, 3):
            return self.none(node, inspect.currentframe().f_lineno)
        if g.is_used_more_than_once(node.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [transpose, node], self.apply, insert_at=node)

    def apply(
        self, g: "GraphBuilder", transpose: NodeProto, rotary: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        last_dim = g.unique_name(f"{transpose.input[0]}::Shape3")
        new_shape2 = g.unique_name(f"{transpose.input[0]}::Shape+1")
        new_shape = g.make_initializer(
            "", np.array([0, 0, -1], dtype=np.int64), source=f"{self.__class__.__name__}.00_1"
        )
        reshaped = g.unique_name(f"{transpose.input[0]}::3D")
        rot_name = g.unique_name(f"{transpose.input[0]}::3Dr")
        reshaped2 = g.unique_name(f"{transpose.input[0]}::4D")
        nodes = [
            g._make_node("Reshape", [transpose.input[0], new_shape], [reshaped]),
            g._make_node(
                rotary.op_type, [reshaped, *rotary.input[1:]], [rot_name], domain=rotary.domain
            ),
            g._make_node("Shape", [transpose.input[0]], [last_dim], start=3),
            g._make_node("Concat", [new_shape, last_dim], [new_shape2], axis=0),
            g._make_node("Reshape", [rot_name, new_shape2], [reshaped2]),
            g._make_node("Transpose", [reshaped2], [rotary.output[0]], perm=[0, 2, 1, 3]),
        ]
        if rotary.attribute:
            nodes[1].attribute.extend(rotary.attribute)
        for node in nodes:
            if not node.name:
                node.name = g.builder.unique_node_name(
                    f"{self.__class__.__name__}--{rotary.name}"
                )
        return nodes


class ContribGemmaRotaryEmbeddingPattern(PatternOptimization):
    """
    Fuses two :class:`intermediate.HalfRotaryEmbedding
    <yobx.xoptim.patterns.onnx_rotary.FunctionHalfRotaryEmbeddingPattern>` nodes
    that share cos/sin inputs traced back through ``Unsqueeze([Cast(]Cos/Sin(emb)[)])``
    into a single ``com.microsoft.GemmaRotaryEmbedding`` node.

    Model with nodes to be fused (after
    :class:`yobx.xoptim.patterns.onnx_rotary.FunctionHalfRotaryEmbeddingPattern`):

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_emb(["emb FLOAT(a, b, c)"])
            I_q(["q FLOAT(a, h1, b, c)"])
            I_k(["k FLOAT(a, h2, b, c)"])

            Sin_0[["Sin(.)"]]
            Cos_1[["Cos(.)"]]
            Unsqueeze_2[["Unsqueeze(., [1])"]]
            Unsqueeze_3[["Unsqueeze(., [1])"]]
            HalfRotaryEmbedding_4[["intermediate.HalfRotaryEmbedding(., ., .)"]]
            HalfRotaryEmbedding_5[["intermediate.HalfRotaryEmbedding(., ., .)"]]

            I_emb -->|"FLOAT(a, b, c)"| Sin_0
            I_emb -->|"FLOAT(a, b, c)"| Cos_1
            Sin_0 -->|"FLOAT(a, b, c)"| Unsqueeze_2
            Cos_1 -->|"FLOAT(a, b, c)"| Unsqueeze_3
            I_q -->|"FLOAT(a, h1, b, c)"| HalfRotaryEmbedding_4
            Unsqueeze_3 -->|"FLOAT(a, 1, b, c)"| HalfRotaryEmbedding_4
            Unsqueeze_2 -->|"FLOAT(a, 1, b, c)"| HalfRotaryEmbedding_4
            I_k -->|"FLOAT(a, h2, b, c)"| HalfRotaryEmbedding_5
            Unsqueeze_3 -->|"FLOAT(a, 1, b, c)"| HalfRotaryEmbedding_5
            Unsqueeze_2 -->|"FLOAT(a, 1, b, c)"| HalfRotaryEmbedding_5

            O_q_embed(["q_embed FLOAT(a, h1, b, c)"])
            HalfRotaryEmbedding_4 --> O_q_embed
            O_k_embed(["k_embed FLOAT(a, h2, b, c)"])
            HalfRotaryEmbedding_5 --> O_k_embed

            class I_emb,I_q,I_k,O_q_embed,O_k_embed ioNode
            class Sin_0,Cos_1,Unsqueeze_2,Unsqueeze_3 opNode
            class HalfRotaryEmbedding_4,HalfRotaryEmbedding_5 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_emb(["emb FLOAT(a, b, c)"])
            I_q(["q FLOAT(a, h1, b, c)"])
            I_k(["k FLOAT(a, h2, b, c)"])

            Split_0[["Split(q, num_outputs=2, axis=-1)"]]
            Neg_1[["Neg(.)"]]
            Concat_2[["Concat(., ., axis=-1)"]]
            Split_3[["Split(k, num_outputs=2, axis=-1)"]]
            Neg_4[["Neg(.)"]]
            Concat_5[["Concat(., ., axis=-1)"]]
            GemmaRotaryEmbedding_6[["com.microsoft.GemmaRotaryEmbedding(., ., ., ., .)"]]

            I_q --> Split_0
            Split_0 --> Neg_1
            Neg_1 --> Concat_2
            Split_0 --> Concat_2
            I_k --> Split_3
            Split_3 --> Neg_4
            Neg_4 --> Concat_5
            Split_3 --> Concat_5
            I_emb -->|"FLOAT(a, b, c)"| GemmaRotaryEmbedding_6
            I_q -->|"FLOAT(a, h1, b, c)"| GemmaRotaryEmbedding_6
            Concat_2 -->|"FLOAT(a, h1, b, c)"| GemmaRotaryEmbedding_6
            I_k -->|"FLOAT(a, h2, b, c)"| GemmaRotaryEmbedding_6
            Concat_5 -->|"FLOAT(a, h2, b, c)"| GemmaRotaryEmbedding_6

            O_q_embed(["q_embed FLOAT(a, h1, b, c)"])
            GemmaRotaryEmbedding_6 --> O_q_embed
            O_k_embed(["k_embed FLOAT(a, h2, b, c)"])
            GemmaRotaryEmbedding_6 --> O_k_embed

            class I_emb,I_q,I_k,O_q_embed,O_k_embed ioNode
            class Split_0,Neg_1,Concat_2,Split_3,Neg_4,Concat_5 opNode
            class GemmaRotaryEmbedding_6 opNode
    """

    _operator_name = FunctionHalfRotaryEmbeddingPattern._operator_name
    _domain_name = FunctionHalfRotaryEmbeddingPattern._domain_name

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def _trace_to_emb(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        unsq_node: "NodeProto",  # noqa: F821
        expected_trig_op: str,
    ) -> Tuple[Optional[str], Optional["NodeProto"], Optional["NodeProto"]]:  # noqa: F821
        """Traces ``Unsqueeze([Cast(]trig_op(emb)[)])`` and returns
        ``(emb, unsq_node, cast_node)``.

        Returns ``(None, None, None)`` if the pattern does not match.
        """
        if unsq_node is None or unsq_node.op_type != "Unsqueeze" or unsq_node.domain != "":
            return None, None, None
        before_unsq = g.node_before(unsq_node.input[0])
        if before_unsq is None:
            return None, None, None
        cast_node = None
        if before_unsq.op_type == "Cast" and before_unsq.domain == "":
            cast_node = before_unsq
            trig_node = g.node_before(cast_node.input[0])
        else:
            trig_node = before_unsq
        if trig_node is None or trig_node.op_type != expected_trig_op or trig_node.domain != "":
            return None, None, None
        return trig_node.input[0], unsq_node, cast_node

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != self._operator_name or node.domain != self._domain_name:
            return self.none()
        if not g.has_rank(node.input[0]) or g.get_rank(node.input[0]) != 4:
            return self.none(node, inspect.currentframe().f_lineno)

        # node.input[1] must trace back to Unsqueeze([Cast(]Cos(emb)[)])
        cos_4d_node = g.node_before(node.input[1])
        emb_from_cos, _cos_unsq, _cos_cast = self._trace_to_emb(g, cos_4d_node, "Cos")
        if emb_from_cos is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # node.input[2] must trace back to Unsqueeze([Cast(]Sin(emb)[)])
        sin_4d_node = g.node_before(node.input[2])
        emb_from_sin, _sin_unsq, _sin_cast = self._trace_to_emb(g, sin_4d_node, "Sin")
        if emb_from_sin is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # Both must trace back to the same emb tensor
        if emb_from_cos != emb_from_sin:
            return self.none(node, inspect.currentframe().f_lineno)

        # emb must be 3D
        if not g.has_rank(emb_from_cos) or g.get_rank(emb_from_cos) != 3:
            return self.none(node, inspect.currentframe().f_lineno)

        # Find the second HalfRotaryEmbedding using the same cos/sin tensors
        other_node = None
        for user in g.next_nodes(node.input[1]):
            if (
                user is not node
                and user.op_type == self._operator_name
                and user.domain == self._domain_name
                and user.input[1] == node.input[1]
                and user.input[2] == node.input[2]
            ):
                other_node = user
                break

        if other_node is None:
            return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self, [node, other_node], self.apply, comment="GemmaRotaryEmbedding fusion"
        )

    def apply(
        self, g: "GraphBuilder", node_first: NodeProto, node_second: NodeProto  # noqa: F821
    ) -> List[NodeProto]:
        q = node_first.input[0]
        k = node_second.input[0]
        q_embed = node_first.output[0]
        k_embed = node_second.output[0]

        # Recover emb by tracing through cos_4d → Unsqueeze → [Cast] → Cos(emb)
        cos_unsq = g.node_before(node_first.input[1])
        before_unsq = g.node_before(cos_unsq.input[0])
        if before_unsq.op_type == "Cast":
            cos_trig = g.node_before(before_unsq.input[0])
        else:
            cos_trig = before_unsq
        emb = cos_trig.input[0]

        name_prefix = f"{self.__class__.__name__}--{node_first.name}"

        # Compute q_rot = rotate_half(q): Split → Neg → Concat
        q1 = g.unique_name(f"{self.__class__.__name__}--{q}--q1")
        q2 = g.unique_name(f"{self.__class__.__name__}--{q}--q2")
        neg_q2 = g.unique_name(f"{self.__class__.__name__}--{q}--neg_q2")
        q_rot = g.unique_name(f"{self.__class__.__name__}--{q}--q_rot")

        # Compute k_rot = rotate_half(k): Split → Neg → Concat
        k1 = g.unique_name(f"{self.__class__.__name__}--{k}--k1")
        k2 = g.unique_name(f"{self.__class__.__name__}--{k}--k2")
        neg_k2 = g.unique_name(f"{self.__class__.__name__}--{k}--neg_k2")
        k_rot = g.unique_name(f"{self.__class__.__name__}--{k}--k_rot")

        nodes = [
            g._make_node("Split", [q], [q1, q2], axis=-1, num_outputs=2, name=name_prefix),
            g._make_node("Neg", [q2], [neg_q2], name=name_prefix),
            g._make_node("Concat", [neg_q2, q1], [q_rot], axis=-1, name=name_prefix),
            g._make_node("Split", [k], [k1, k2], axis=-1, num_outputs=2, name=name_prefix),
            g._make_node("Neg", [k2], [neg_k2], name=name_prefix),
            g._make_node("Concat", [neg_k2, k1], [k_rot], axis=-1, name=name_prefix),
        ]
        gemma_node = g.make_node(
            "GemmaRotaryEmbedding",
            [emb, q, q_rot, k, k_rot],
            [q_embed, k_embed],
            name=name_prefix,
            domain="com.microsoft",
        )
        nodes.append(gemma_node)
        for n in nodes:
            if not n.name:
                n.name = g.builder.unique_node_name(name_prefix)
        return nodes


class MultiHeadAttention3DPattern(PatternOptimization):
    """
    Merges multiple nodes into MultiHeadAttention. It assumes pattern
    :class:`yobx.xoptim.patterns.onnx_attention.FunctionAttentionPattern`
    was triggered before.

    Model with nodes to be fused:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_mask(["mask BOOL(am, 1, cm, dm)"])
            I_past_values(["past_values FLOAT(pav, 8, pcv, 64)"])
            I_values(["values FLOAT(av, bv, 8, 64)"])
            I_query(["query FLOAT(aq, bq, 8, 64)"])
            I_past_keys(["past_keys FLOAT(pak, 8, pck, 64)"])
            I_keys(["keys FLOAT(ak, bk, 8, 64)"])

            Transpose_0[["Transpose(., perm=[0, 2, 1, 3])"]]
            Transpose_1[["Transpose(., perm=[0, 2, 1, 3])"]]
            Concat_2[["Concat(., ., axis=-2)"]]
            Transpose_3[["Transpose(., perm=[0, 2, 1, 3])"]]
            Concat_4[["Concat(., ., axis=-2)"]]
            LocalAttention_to1_5[["intermediate.LocalAttention_to1(., ., ., ., [0.31622776])"]]
            Transpose_6[["Transpose(., perm=[0, 2, 1, 3])"]]

            I_query -->|"FLOAT(aq, bq, 8, 64)"| Transpose_0
            I_keys -->|"FLOAT(ak, bk, 8, 64)"| Transpose_1
            I_past_keys -->|"FLOAT(pak, 8, pck, 64)"| Concat_2
            Transpose_1 --> Concat_2
            I_values -->|"FLOAT(av, bv, 8, 64)"| Transpose_3
            I_past_values -->|"FLOAT(pav, 8, pcv, 64)"| Concat_4
            Transpose_3 --> Concat_4
            Transpose_0 --> LocalAttention_to1_5
            Concat_2 --> LocalAttention_to1_5
            Concat_4 --> LocalAttention_to1_5
            I_mask -->|"BOOL(am, 1, cm, dm)"| LocalAttention_to1_5
            LocalAttention_to1_5 --> Transpose_6

            O_ct_values(["ct_values FLOAT(pav, 8, pcv+bv, 64)"])
            Concat_4 --> O_ct_values
            O_Y(["Y FLOAT(ay, by, cy, dy)"])
            Transpose_6 --> O_Y
            O_ct_keys(["ct_keys FLOAT(pak, 8, pck+bk, 64)"])
            Concat_2 --> O_ct_keys

            class I_mask,I_past_values,I_values,I_query,I_past_keys,I_keys ioNode
            class O_ct_values,O_Y,O_ct_keys ioNode
            class Transpose_0,Transpose_1,Concat_2,Transpose_3,Concat_4 opNode
            class LocalAttention_to1_5,Transpose_6 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_mask(["mask BOOL(am, 1, cm, dm)"])
            I_past_values(["past_values FLOAT(pav, 8, pcv, 64)"])
            I_values(["values FLOAT(av, bv, 8, 64)"])
            I_query(["query FLOAT(aq, bq, 8, 64)"])
            I_past_keys(["past_keys FLOAT(pak, 8, pck, 64)"])
            I_keys(["keys FLOAT(ak, bk, 8, 64)"])

            Reshape_0[["Reshape(., [0, 0, -1])"]]
            Reshape_1[["Reshape(., [0, 0, -1])"]]
            Reshape_2[["Reshape(., [0, 0, -1])"]]
            Where_3[["Where(., [0.0], [-inf])"]]
            MultiHeadAttention_4[["com.microsoft.MultiHeadAttention(., ., ., , , ., ., .)"]]
            Reshape_5[["Reshape(., [0, 0, -1, 64])"]]

            I_query -->|"FLOAT(aq, bq, 8, 64)"| Reshape_0
            I_keys -->|"FLOAT(ak, bk, 8, 64)"| Reshape_1
            I_values -->|"FLOAT(av, bv, 8, 64)"| Reshape_2
            I_mask -->|"BOOL(am, 1, cm, dm)"| Where_3
            Reshape_0 -->|"FLOAT(aq, bq, 512)"| MultiHeadAttention_4
            Reshape_1 -->|"FLOAT(ak, bk, 512)"| MultiHeadAttention_4
            Reshape_2 -->|"FLOAT(av, bv, 512)"| MultiHeadAttention_4
            Where_3 -->|"FLOAT(am, 1, cm, dm)"| MultiHeadAttention_4
            I_past_keys -->|"FLOAT(pak, 8, pck, 64)"| MultiHeadAttention_4
            I_past_values -->|"FLOAT(pav, 8, pcv, 64)"| MultiHeadAttention_4
            MultiHeadAttention_4 -->|"FLOAT(aq, bq, 512)"| Reshape_5

            O_ct_values(["ct_values FLOAT(pav, 8, pcv+bv, 64)"])
            MultiHeadAttention_4 --> O_ct_values
            O_Y(["Y FLOAT(ay, by, cy, dy)"])
            Reshape_5 --> O_Y
            O_ct_keys(["ct_keys FLOAT(pak, 8, pck+bk, 64)"])
            MultiHeadAttention_4 --> O_ct_keys

            class I_mask,I_past_values,I_values,I_query,I_past_keys,I_keys ioNode
            class O_ct_values,O_Y,O_ct_keys ioNode
            class Reshape_0,Reshape_1,Reshape_2,Where_3,MultiHeadAttention_4,Reshape_5 opNode
    """

    _prefixes_operator_name = (
        f"{FunctionAttentionPattern._operator_name}_to",
        f"{FunctionAttentionPattern._operator_name}sQ_to",
        f"{FunctionAttentionPattern._operator_name}SW_to",
        f"{FunctionAttentionPattern._operator_name}SWsQ_to",
        f"{FunctionAttentionPattern._operator_name}NoT_to",
    )

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            not node.op_type.startswith(self._prefixes_operator_name)
            or node.domain != FunctionAttentionPattern._domain_name
            or len(node.input) != 5
        ):
            return self.none()
        if not g.is_constant_scalar(node.input[4]):
            return self.none(node, inspect.currentframe().f_lineno)

        q_transpose = g.node_before(node.input[0])
        expected_perm = (0, 2, 1, 3)
        if (
            q_transpose is None
            or q_transpose.op_type != "Transpose"
            or tuple(g.get_attribute(q_transpose, "perm").ints) != expected_perm
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.has_shape(q_transpose.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape = g.get_shape(q_transpose.input[0])
        if not isinstance(shape[2], int):
            return self.none(node, inspect.currentframe().f_lineno)

        k_concat = g.node_before(node.input[1])
        if (
            k_concat is None
            or k_concat.op_type != "Concat"
            or g.get_attribute(k_concat, "axis").i != -2
            or len(k_concat.input) != 2
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        k_transpose = g.node_before(k_concat.input[1])
        if (
            k_transpose is None
            or k_transpose.op_type != "Transpose"
            or tuple(g.get_attribute(k_transpose, "perm").ints) != expected_perm
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        v_concat = g.node_before(node.input[2])
        if (
            v_concat is None
            or v_concat.op_type != "Concat"
            or g.get_attribute(v_concat, "axis").i != -2
            or len(v_concat.input) != 2
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        v_transpose = g.node_before(v_concat.input[1])
        if (
            v_transpose is None
            or v_transpose.op_type != "Transpose"
            or tuple(g.get_attribute(v_transpose, "perm").ints) != expected_perm
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        transposes = g.next_nodes(node.output[0])
        if len(transposes) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        transpose = transposes[0]
        if (
            transpose is None
            or transpose.op_type != "Transpose"
            or tuple(g.get_attribute(transpose, "perm").ints) != expected_perm
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        if (
            not g.has_shape(q_transpose.input[0])
            or g.get_rank(q_transpose.input[0]) != 4
            or not isinstance(g.get_shape(q_transpose.input[0])[-1], int)
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        for n in [q_transpose, k_transpose, v_transpose, node]:
            if n and g.is_used_more_than_once(n.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(
            self,
            [q_transpose, k_transpose, k_concat, v_transpose, v_concat, node, transpose],
            self.apply,
        )

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        q_transpose: NodeProto,
        k_transpose: NodeProto,
        k_concat: NodeProto,
        v_transpose: NodeProto,
        v_concat: NodeProto,
        attention: NodeProto,
        transpose: NodeProto,
    ) -> List[NodeProto]:
        query = q_transpose.input[0]
        keys = k_transpose.input[0]
        values = v_transpose.input[0]
        mask = attention.input[3]
        past_keys = k_concat.input[0]
        past_values = v_concat.input[0]
        num_heads = g.get_shape(query)[2]

        scale = float(g.get_constant_scalar(attention.input[4])) ** 2
        dtype = tensor_dtype_to_np_dtype(g.get_type(query))
        zero = g.make_initializer(
            "", np.array([0], dtype=dtype), source=f"{self.__class__.__name__}.0"
        )
        minfty = g.make_initializer(
            "", np.array([-np.inf], dtype=dtype), source=f"{self.__class__.__name__}._inf"
        )
        init_00_1 = g.make_initializer(
            "", np.array([0, 0, -1], dtype=np.int64), source=f"{self.__class__.__name__}.00_1"
        )
        last = g.get_shape(query)[-1]
        init_00_1l = g.make_initializer(
            "",
            np.array([0, 0, -1, last], dtype=np.int64),
            source=f"{self.__class__.__name__}.00_1l",
        )

        r_query = g.unique_name(f"{self.__class__.__name__}--{query}")
        r_keys = g.unique_name(f"{self.__class__.__name__}--{keys}")
        r_values = g.unique_name(f"{self.__class__.__name__}--{values}")
        attention_bias = g.unique_name(f"{self.__class__.__name__}--{mask}")
        r_output = g.unique_name(f"{self.__class__.__name__}--{transpose.output[0]}")
        switch_where = "SW" in attention.op_type

        nodes = [
            g._make_node("Reshape", [query, init_00_1], [r_query]),
            g._make_node("Reshape", [keys, init_00_1], [r_keys]),
            g._make_node("Reshape", [values, init_00_1], [r_values]),
            g._make_node(
                "Where",
                [mask, minfty, zero] if switch_where else [mask, zero, minfty],
                [attention_bias],
            ),
            g._make_node(
                "MultiHeadAttention",
                [r_query, r_keys, r_values, "", "", attention_bias, past_keys, past_values],
                [r_output, k_concat.output[0], v_concat.output[0]],
                num_heads=num_heads,
                scale=scale,
                domain="com.microsoft",
            ),
            g._make_node("Reshape", [r_output, init_00_1l], [transpose.output[0]]),
        ]
        for node in nodes:
            if node.name:
                continue
            node.name = g.builder.unique_node_name(f"{self.__class__.__name__}--{attention.name}")
        return nodes


class GroupQueryAttention3DPattern(PatternOptimization):
    """
    Fuse LocalAttention into GroupQueryAttention.
    ``bias`` is not supported by this kernel on CUDA.

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_query(["query FLOAT(batch, 8, seq_length, 32)"])
            I_past_value(["past_value FLOAT(batch, 4, past_length, 32)"])
            I_key(["key FLOAT(batch, 4, seq_length, 32)"])
            I_value(["value FLOAT(batch, 4, seq_length, 32)"])
            I_past_key(["past_key FLOAT(batch, 4, past_length, 32)"])
            I_bitwise_not(["bitwise_not BOOL(seq_length, total_length)"])

            Concat_0[["Concat(., ., axis=2)"]]
            Concat_1[["Concat(., ., axis=2)"]]
            locatt2[["intermediate.LocalAttentionGQASW_to1(
            ., ., ., ., [0.4204482], [1, 1, 2, 1, 1], [0, 8, -1, 32])"]]

            I_past_key -->|"FLOAT(batch, 4, past_length, 32)"| Concat_0
            I_key -->|"FLOAT(batch, 4, seq_length, 32)"| Concat_0
            I_past_value -->|"FLOAT(batch, 4, past_length, 32)"| Concat_1
            I_value -->|"FLOAT(batch, 4, seq_length, 32)"| Concat_1
            I_query -->|"FLOAT(batch, 8, seq_length, 32)"| locatt2
            Concat_0 --> locatt2
            Concat_1 --> locatt2
            I_bitwise_not -->|"BOOL(seq_length, total_length)"| locatt2

            O_output_0(["output_0 FLOAT(batch, 8, seq_length, 32)"])
            locatt2 --> O_output_0
            O_cat_1(["cat_1 FLOAT(batch, 4, past_length+seq_length, 32)"])
            Concat_1 --> O_cat_1
            O_cat(["cat FLOAT(batch, 4, past_length+seq_length, 32)"])
            Concat_0 --> O_cat

            class I_query,I_past_value,I_key,I_value,I_past_key,I_bitwise_not ioNode
            class O_output_0,O_cat_1,O_cat ioNode
            class Concat_0,Concat_1,locatt2 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_query(["query FLOAT(batch, 8, seq_length, 32)"])
            I_past_value(["past_value FLOAT(batch, 4, past_length, 32)"])
            I_key(["key FLOAT(batch, 4, seq_length, 32)"])
            I_value(["value FLOAT(batch, 4, seq_length, 32)"])
            I_past_key(["past_key FLOAT(batch, 4, past_length, 32)"])
            I_bitwise_not(["bitwise_not BOOL(seq_length, total_length)"])

            Where_0[["Where(., [-3.4028235e+38], [0.0])"]]
            Shape_1[["Shape(., end=1, start=0)"]]
            Unsqueeze_2[["Unsqueeze(., [0, 1])"]]
            Shape_3[["Shape(., start=-1)"]]
            Cast_4[["Cast(., to=INT32)"]]
            Sub_5[["Sub(., [1])"]]
            Expand_6[["Expand(., .)"]]
            Transpose_7[["Transpose(., perm=[0, 2, 1, 3])"]]
            Transpose_8[["Transpose(., perm=[0, 2, 1, 3])"]]
            Transpose_9[["Transpose(., perm=[0, 2, 1, 3])"]]
            Reshape_10[["Reshape(., [0, 0, -1])"]]
            Reshape_11[["Reshape(., [0, 0, -1])"]]
            Reshape_12[["Reshape(., [0, 0, -1])"]]
            gqa13[["com.microsoft.GroupQueryAttention(., ., ., ., ., ., ., , , , .)"]]
            Reshape_14[["Reshape(., [0, 0, -1, 32])"]]
            Transpose_15[["Transpose(., perm=[0, 2, 1, 3])"]]

            I_bitwise_not -->|"BOOL(seq_length, total_length)"| Where_0
            I_query -->|"FLOAT(batch, 8, seq_length, 32)"| Shape_1
            Where_0 --> Unsqueeze_2
            Where_0 --> Shape_3
            Shape_3 --> Cast_4
            Cast_4 --> Sub_5
            Sub_5 --> Expand_6
            Shape_1 --> Expand_6
            I_query -->|"FLOAT(batch, 8, seq_length, 32)"| Transpose_7
            I_key -->|"FLOAT(batch, 4, seq_length, 32)"| Transpose_8
            I_value -->|"FLOAT(batch, 4, seq_length, 32)"| Transpose_9
            Transpose_7 --> Reshape_10
            Transpose_8 --> Reshape_11
            Transpose_9 --> Reshape_12
            Reshape_10 --> gqa13
            Reshape_11 --> gqa13
            Reshape_12 --> gqa13
            I_past_key -->|"FLOAT(batch, 4, past_length, 32)"| gqa13
            I_past_value -->|"FLOAT(batch, 4, past_length, 32)"| gqa13
            Expand_6 --> gqa13
            Cast_4 --> gqa13
            Unsqueeze_2 --> gqa13
            gqa13 --> Reshape_14
            Reshape_14 --> Transpose_15

            O_output_0(["output_0 FLOAT(batch, 8, seq_length, 32)"])
            Transpose_15 --> O_output_0
            O_cat_1(["cat_1 FLOAT(batch, 4, past_length+seq_length, 32)"])
            gqa13 --> O_cat_1
            O_cat(["cat FLOAT(batch, 4, past_length+seq_length, 32)"])
            gqa13 --> O_cat

            class I_query,I_past_value,I_key,I_value,I_past_key,I_bitwise_not ioNode
            class O_output_0,O_cat_1,O_cat ioNode
            class Where_0,Shape_1,Unsqueeze_2,Shape_3,Cast_4,Sub_5,Expand_6,Transpose_7 opNode
            class Transpose_8,Transpose_9,Reshape_10,Reshape_11,Reshape_12 opNode
            class gqa13,Reshape_14,Transpose_15 opNode
    """

    _prefixes_operator_name = (
        f"{FunctionAttentionGQAPattern._operator_gqa_name}SW_to",
        f"{FunctionAttentionGQAPattern._operator_gqa_name}SWsQ_to",
        f"{FunctionAttentionGQAPattern._operator_gqa_name}_to",
        f"{FunctionAttentionGQAPattern._operator_gqa_name}sQ_to",
    )

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            not node.op_type.startswith(self._prefixes_operator_name)
            or node.domain != FunctionAttentionGQAPattern._domain_name
            or len(node.input) != 7
        ):
            return self.none()
        keys, values = node.input[1:3]
        concats = g.node_before(keys), g.node_before(values)
        if None in concats:
            return self.none(node, inspect.currentframe().f_lineno)
        if len(concats[0].input) != 2 or len(concats[1].input) != 2:
            return self.none(node, inspect.currentframe().f_lineno)
        if g.get_attribute_with_default(concats[0], "axis", 0) != g.get_attribute_with_default(
            concats[1], "axis", 0
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        if len(node.input) > 3 and node.input[3] and g.has_processor("CUDA"):
            # GroupQueryAttention does not work with a bias.
            return self.none()

        if len(node.input) > 3 and (
            not g.has_rank(node.input[3]) or g.get_rank(node.input[3]) < 2
        ):
            # Only 2D ranks allowed.
            return self.none(node, inspect.currentframe().f_lineno)

        if not g.is_constant_scalar(node.input[4]):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[5]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(node.input[5])
        if cst is None:
            return self.none(node, inspect.currentframe().f_lineno)
        cst = tuple(cst)
        if len(cst) < 4:
            return self.none(node, inspect.currentframe().f_lineno)
        if cst[:2] != cst[3:] or cst[:2] != (1, 1):
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(node.input[6]):
            return self.none(node, inspect.currentframe().f_lineno)
        shape_or_axis = g.get_computed_constant(node.input[6])
        if shape_or_axis is None:
            return self.none(node, inspect.currentframe().f_lineno)
        if "sQ_to" in node.op_type:
            # This is an axis for a Squeeze node.
            if not g.get_shape(node.input[1]):
                # We need that shape to get kv_num_heads.
                return self.none(node, inspect.currentframe().f_lineno)
        else:
            # This is a shape for a Reshape node.
            if shape_or_axis[1] <= 0:
                return self.none(node, inspect.currentframe().f_lineno)
        return MatchResult(self, [*concats, node], self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        keys_concat_node: NodeProto,
        values_concat_node: NodeProto,
        local_attention_gqa: NodeProto,
    ) -> List[NodeProto]:
        query, _keys, _values, mask = local_attention_gqa.input[:4]
        scale = g.get_constant_scalar(local_attention_gqa.input[4])  # this scale ** 0.5
        expand_shape = g.get_computed_constant(local_attention_gqa.input[5])
        repeat = int(expand_shape[2])
        rk_mask = g.get_rank(mask)

        if "sQ_" in local_attention_gqa.op_type:
            k_shape = g.get_shape(local_attention_gqa.input[1])
            kv_num_heads = k_shape[1]
        else:
            reshape_shape = g.get_computed_constant(local_attention_gqa.input[6])
            kv_num_heads = reshape_shape[1] // repeat

        num_heads = kv_num_heads * repeat
        head_size = g.get_shape(query)[-1]

        shape00 = g.make_initializer(
            "", np.array([0, 0, -1], dtype=np.int64), source=f"{self.__class__.__name__}.00"
        )
        shape0000 = g.make_initializer(
            "",
            np.array([0, 0, -1, head_size], dtype=np.int64),
            source=f"{self.__class__.__name__}.00_1",
        )

        query3D = g.unique_name(f"{self.__class__.__name__}--{query}")
        query4D = g.unique_name(f"{self.__class__.__name__}--{query}")
        keys3D = g.unique_name(f"{self.__class__.__name__}--{_keys}")
        keys4D = g.unique_name(f"{self.__class__.__name__}--{_keys}")
        values3D = g.unique_name(f"{self.__class__.__name__}--{_values}")
        values4D = g.unique_name(f"{self.__class__.__name__}--{_values}")
        attn3D = g.unique_name(f"{self.__class__.__name__}--{local_attention_gqa.output[0]}")
        attn4D = g.unique_name(f"{self.__class__.__name__}--{local_attention_gqa.output[0]}")

        seqlensk = g.unique_name(f"{self.__class__.__name__}--sl")
        total_length64 = g.unique_name(f"{self.__class__.__name__}--tl")
        total_length = g.unique_name(f"{self.__class__.__name__}--tl")
        seq_length64 = g.unique_name(f"{self.__class__.__name__}--seq64")
        seq_length32 = g.unique_name(f"{self.__class__.__name__}--seq32")
        past_length = g.unique_name(f"{self.__class__.__name__}--pl")

        batch_shape = g.unique_name(f"{self.__class__.__name__}--{query}")

        nodes = []
        # mask is not mask if SW
        switch_where = "SW" in local_attention_gqa.op_type
        if g.get_type(mask) == TensorProto.BOOL:
            itype = g.get_type(query)
            dtype = tensor_dtype_to_np_dtype(itype)
            zero = g.make_initializer(
                "", np.array([0], dtype=dtype), source=f"{self.__class__.__name__}.0"
            )
            infty = g.make_initializer(
                "",
                np.array([np.finfo(dtype).min], dtype=dtype),
                source=f"{self.__class__.__name__}.inf{itype}",
            )
            float_mask = g.unique_name(f"{self.__class__.__name__}--{mask}")
            nodes.append(
                g._make_node(
                    "Where",
                    [mask, infty, zero] if switch_where else [mask, zero, infty],
                    [float_mask],
                )
            )
        else:
            raise NotImplementedError(
                f"float mask is not implemented yet for pattern {self.__class__.__name__!r}"
            )

        if rk_mask == 2:
            expanded_mask = g.unique_name(f"{self.__class__.__name__}--{mask}")
            cst01 = g.make_initializer(
                "", np.array([0, 1], dtype=np.int64), source=f"{self.__class__.__name__}.01"
            )
            nodes.append(g._make_node("Unsqueeze", [float_mask, cst01], [expanded_mask]))
        elif rk_mask == 3:
            expanded_mask = g.unique_name(f"{self.__class__.__name__}--{mask}")
            cst0 = g.make_initializer(
                "", np.array([0], dtype=np.int64), source=f"{self.__class__.__name__}.0"
            )
            nodes.append(g._make_node("Unsqueeze", [float_mask, cst0], [expanded_mask]))
        else:
            expanded_mask = float_mask

        attention_node = g._make_node(
            "GroupQueryAttention",
            [
                query3D,
                keys3D,
                values3D,
                keys_concat_node.input[0],
                values_concat_node.input[0],
                seqlensk,
                total_length,
                "",
                "",
                "",
                expanded_mask,
            ],
            [attn3D, keys_concat_node.output[0], values_concat_node.output[0]],
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            scale=scale**2,
            do_rotary=0,
            rotary_interleaved=0,
            domain="com.microsoft",
            doc_string="This operator only accepts batch_size=1 "
            "and (past_length==0 or seq_length==1).",
        )

        nodes.extend(
            [
                g._make_node("Shape", [query], [batch_shape], start=0, end=1),
                g._make_node("Shape", [float_mask], [total_length64], start=-1),
                g._make_node("Cast", [total_length64], [total_length], to=TensorProto.INT32),
                g._make_node("Shape", [float_mask], [seq_length64], start=0, end=1),
                g._make_node("Cast", [seq_length64], [seq_length32], to=TensorProto.INT32),
                g._make_node("Sub", [total_length, seq_length32], [past_length]),
                g._make_node("Expand", [past_length, batch_shape], [seqlensk]),
                g._make_node("Transpose", [query], [query4D], perm=[0, 2, 1, 3]),
                g._make_node(
                    "Transpose", [keys_concat_node.input[1]], [keys4D], perm=[0, 2, 1, 3]
                ),
                g._make_node(
                    "Transpose", [values_concat_node.input[1]], [values4D], perm=[0, 2, 1, 3]
                ),
                g._make_node("Reshape", [query4D, shape00], [query3D]),
                g._make_node("Reshape", [keys4D, shape00], [keys3D]),
                g._make_node("Reshape", [values4D, shape00], [values3D]),
                attention_node,
                g._make_node("Reshape", [attn3D, shape0000], [attn4D]),
                g._make_node(
                    "Transpose", [attn4D], [local_attention_gqa.output[0]], perm=[0, 2, 1, 3]
                ),
            ]
        )
        for node in nodes:
            if not node.name:
                node.name = g.builder.unique_node_name(
                    f"{self.__class__.__name__}--{local_attention_gqa.name}"
                )
        return nodes


class Attention3DPattern(PatternOptimization):
    """
    Fuses nodes into Attention from `com.microsoft` domain.
    In progress.
    """

    _prefixes_operator_name = (f"{FunctionAttentionPattern._operator_name}_to",)

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def _match_above(
        self, g: "GraphBuilderPatternOptimization", node: NodeProto, name: str  # noqa: F821
    ) -> Optional[Tuple[NodeProto, NodeProto, NodeProto]]:
        transpose = g.node_before(name)
        if not transpose or transpose.op_type != "Transpose":
            return self.none(node, inspect.currentframe().f_lineno)
        if tuple(g.get_attribute(transpose, "perm").ints) != (0, 2, 1, 3):
            return self.none(node, inspect.currentframe().f_lineno)
        reshape = g.node_before(transpose.input[0])
        if not reshape or reshape.op_type != "Reshape":
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant(reshape.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(reshape.input[1])
        if cst is None:
            return self.none(node, inspect.currentframe().f_lineno)
        matmul = g.node_before(reshape.input[0])
        if matmul is None or matmul.op_type != "MatMul":
            return self.none(node, inspect.currentframe().f_lineno)
        return matmul, reshape, transpose

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if (
            not node.op_type.startswith(self._prefixes_operator_name)
            or node.domain != FunctionAttentionGQAPattern._domain_name
            or len(node.input) != 5
        ):
            return self.none()
        if len(node.input) > 3 and node.input[3] and not g.has_type(node.input[3]):
            # mask type is unknown
            return self.none(node, inspect.currentframe().f_lineno)
        if not g.is_constant_scalar(node.input[4]):
            # scale is expected to be a constant scalar; otherwise apply() will fail
            return self.none(node, inspect.currentframe().f_lineno)

        query, keys, values = node.input[:3]

        before_query = self._match_above(g, node, query)
        if before_query is None:
            return self.none(node, inspect.currentframe().f_lineno)
        before_keys = self._match_above(g, node, keys)
        if before_keys is None:
            return self.none(node, inspect.currentframe().f_lineno)
        before_values = self._match_above(g, node, values)
        if before_values is None:
            return self.none(node, inspect.currentframe().f_lineno)

        mm_q, re_q, tr_q = before_query
        mm_k, re_k, tr_k = before_keys
        mm_v, re_v, tr_v = before_values

        if mm_q.input[0] != mm_k.input[0] or mm_q.input[0] != mm_v.input[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        cst_q = g.get_computed_constant(re_q.input[1])
        cst_k = g.get_computed_constant(re_k.input[1])
        cst_v = g.get_computed_constant(re_v.input[1])
        if tuple(cst_q) != tuple(cst_k) or tuple(cst_q) != tuple(cst_v):
            return self.none(node, inspect.currentframe().f_lineno)

        transposes = g.next_nodes(node.output[0])
        if len(transposes) != 1 or transposes[0].op_type != "Transpose":
            return self.none(node, inspect.currentframe().f_lineno)
        transpose = transposes[0]
        if tuple(g.get_attribute(transpose, "perm").ints) != (0, 2, 1, 3):
            return self.none(node, inspect.currentframe().f_lineno)
        reshapes = g.next_nodes(transpose.output[0])
        if len(reshapes) != 1 or reshapes[0].op_type != "Reshape":
            return self.none(node, inspect.currentframe().f_lineno)
        reshape = reshapes[0]
        if not g.is_constant(reshape.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)
        cst = g.get_computed_constant(reshape.input[1])
        if cst is None:
            return self.none(node, inspect.currentframe().f_lineno)

        nodes = [mm_q, re_q, tr_q, mm_k, re_k, tr_k, mm_v, re_v, tr_v, node, transpose, reshape]
        for n in nodes[:-1]:
            if g.is_used_more_than_once(n.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)

        return MatchResult(self, nodes, self.apply)

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        mm_q: NodeProto,
        re_q: NodeProto,
        tr_q: NodeProto,
        mm_k: NodeProto,
        re_k: NodeProto,
        tr_k: NodeProto,
        mm_v: NodeProto,
        re_v: NodeProto,
        tr_v: NodeProto,
        attention: NodeProto,
        transpose: NodeProto,
        reshape: NodeProto,
    ) -> List[NodeProto]:

        packed = g.unique_name(
            f"{self.__class__.__name__}--{mm_q.input[1]}-{mm_k.input[1]}-{mm_v.input[1]}"
        )
        concat_node = g.make_node(
            "Concat",
            [mm_q.input[1], mm_k.input[1], mm_v.input[1]],
            [packed],
            axis=-1,
            name=f"{self.__class__.__name__}--{attention.name}",
        )
        scale = float(g.get_constant_scalar(attention.input[4])) ** 2
        num_heads = g.get_shape(attention.input[0])[1]
        sizes = [
            g.get_shape(mm_q.input[1])[-1],
            g.get_shape(mm_k.input[1])[-1],
            g.get_shape(mm_v.input[1])[-1],
        ]
        dtype = tensor_dtype_to_np_dtype(g.get_type(attention.input[0]))
        zero_bias = g.make_initializer(
            "", np.zeros(sum(sizes), dtype=dtype), source=f"{self.__class__.__name__}.bias"
        )

        where_node = None
        if len(attention.input) > 3 and attention.input[3]:
            if g.get_type(attention.input[3]) == TensorProto.BOOL:
                dtype = tensor_dtype_to_np_dtype(g.get_type(attention.input[0]))
                mask = g.unique_name(f"{self.__class__.__name__}--{attention.input[3]}")
                zero = g.make_initializer(
                    "", np.array([0], dtype=dtype), source=f"{self.__class__.__name__}.0"
                )
                inf = g.make_initializer(
                    "",
                    np.array([np.finfo(dtype).min], dtype=dtype),
                    source=f"{self.__class__.__name__}.inf",
                )
                where_node = g.make_node(
                    "Where",
                    [attention.input[3], zero, inf],
                    [mask],
                    name=f"{self.__class__.__name__}--{attention.name}",
                )
            else:
                mask = attention.input[3]
        else:
            mask = ""

        attention_node = g.make_node(
            "Attention",
            [mm_q.input[0], packed, zero_bias, "", "", mask],
            [reshape.output[0]],
            num_heads=num_heads,
            qkv_hidden_sizes=sizes,
            scale=scale,
            domain="com.microsoft",
            name=f"{self.__class__.__name__}--{attention.name}",
        )
        return [n for n in [concat_node, where_node, attention_node] if n]
