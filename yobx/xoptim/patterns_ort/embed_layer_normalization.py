import inspect
from typing import List, Optional, Tuple
from onnx import NodeProto, TensorProto, helper
from ..patterns_api import MatchResult, PatternOptimization


class EmbedLayerNormalizationPattern(PatternOptimization):
    """
    Fuses the sequence of Gather + Add + LayerNormalization nodes into
    ``com.microsoft.EmbedLayerNormalization``.

    This pattern handles transformer model embedding layers where word, position, and
    optionally segment embeddings are looked up via ``Gather`` nodes, summed via ``Add``
    nodes, and then normalized via ``LayerNormalization``.

    Model with nodes to be fused (3-embedding BERT variant):

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_input_ids(["input_ids INT64(B, S)"])
            I_segment_ids(["segment_ids INT64(B, S)"])
            I_position_ids(["position_ids INT64(B, S)"])
            I_word_table(["word_table FLOAT(V, D)"])
            I_seg_table(["seg_table FLOAT(NS, D)"])
            I_pos_table(["pos_table FLOAT(NP, D)"])
            I_gamma(["gamma FLOAT(D)"])
            I_beta(["beta FLOAT(D)"])

            Constant_0[["Constant() -#gt; word_table"]]
            Constant_1[["Constant() -#gt; pos_table"]]
            Constant_2[["Constant() -#gt; seg_table"]]
            Gather_0[["Gather(., .)"]]
            Gather_1[["Gather(., .)"]]
            Gather_2[["Gather(., .)"]]
            Add_0[["Add(., .)"]]
            Add_1[["Add(., .)"]]
            LayerNormalization_2[["LayerNormalization(., ., .)"]]

            I_input_ids -->|"INT64(B, S)"| Gather_0
            Constant_0 -->|"FLOAT(V, D)"| Gather_0
            I_position_ids -->|"INT64(B, S)"| Gather_1
            Constant_1 -->|"FLOAT(NP, D)"| Gather_1
            I_segment_ids -->|"INT64(B, S)"| Gather_2
            Constant_2 -->|"FLOAT(NS, D)"| Gather_2
            Gather_0 -->|"FLOAT(B, S, D)"| Add_0
            Gather_1 -->|"FLOAT(B, S, D)"| Add_0
            Add_0 -->|"FLOAT(B, S, D)"| Add_1
            Gather_2 -->|"FLOAT(B, S, D)"| Add_1
            Add_1 -->|"FLOAT(B, S, D)"| LayerNormalization_2
            I_gamma -->|"FLOAT(D)"| LayerNormalization_2
            I_beta -->|"FLOAT(D)"| LayerNormalization_2

            O_Y(["Y FLOAT(B, S, D)"])
            LayerNormalization_2 --> O_Y

            class I_input_ids,I_segment_ids,I_position_ids,I_gamma,I_beta,O_Y ioNode
            class Constant_0,Constant_1,Constant_2 constNode
            class Gather_0,Gather_1,Gather_2,Add_0,Add_1,LayerNormalization_2 opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_input_ids(["input_ids INT64(B, S)"])
            I_segment_ids(["segment_ids INT64(B, S)"])
            I_position_ids(["position_ids INT64(B, S)"])
            I_word_table(["word_table FLOAT(V, D)"])
            I_seg_table(["seg_table FLOAT(NS, D)"])
            I_pos_table(["pos_table FLOAT(NP, D)"])
            I_gamma(["gamma FLOAT(D)"])
            I_beta(["beta FLOAT(D)"])

            EmbedLayerNormalization[["com.microsoft.EmbedLayerNormalization(7 inputs)"]]
            I_input_ids -->|"INT64(B, S)"| EmbedLayerNormalization
            I_segment_ids -->|"INT64(B, S)"| EmbedLayerNormalization
            I_word_table -->|"FLOAT(V, D)"| EmbedLayerNormalization
            I_pos_table -->|"FLOAT(NP, D)"| EmbedLayerNormalization
            I_seg_table -->|"FLOAT(NS, D)"| EmbedLayerNormalization
            I_gamma -->|"FLOAT(D)"| EmbedLayerNormalization
            I_beta -->|"FLOAT(D)"| EmbedLayerNormalization

            O_Y(["Y FLOAT(B, S, D)"])
            EmbedLayerNormalization --> O_Y

            class I_input_ids,I_segment_ids,I_position_ids,I_gamma,I_beta,O_Y ioNode
            class EmbedLayerNormalization opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    def _try_get_gather_from_add(
        self, g: "GraphBuilderPatternOptimization", add_node: NodeProto  # noqa: F821
    ) -> Optional[Tuple[NodeProto, NodeProto]]:
        """
        Returns two Gather nodes if the given Add node's inputs are both Gather nodes,
        otherwise returns None.
        """
        g0 = g.node_before(add_node.input[0])
        g1 = g.node_before(add_node.input[1])
        if g0 is None or g1 is None:
            return None
        if g0.op_type != "Gather" or g0.domain != "":
            return None
        if g1.op_type != "Gather" or g1.domain != "":
            return None
        return g0, g1

    def _get_vocab_size(
        self, g: "GraphBuilderPatternOptimization", gather_node: NodeProto  # noqa: F821
    ) -> Optional[int]:
        """
        Returns the first dimension of the Gather node's weight table, or None if unknown.
        """
        if not g.has_shape(gather_node.input[0]):
            return None
        shape = g.get_shape(gather_node.input[0])
        if len(shape) < 1:
            return None
        return shape[0] if isinstance(shape[0], int) else None

    def _assign_gather_roles(
        self, g: "GraphBuilderPatternOptimization", gathers: List[NodeProto]  # noqa: F821
    ) -> Optional[Tuple[NodeProto, NodeProto, NodeProto]]:
        """
        Assigns the three Gather nodes to word/position/segment roles based on
        the vocabulary size (first dimension) of their constant tables.

        Returns (word_gather, position_gather, segment_gather) or None if unresolvable.
        The word embedding has the largest vocab, segment the smallest, position is in-between.
        """
        vocab_sizes = []
        for gather in gathers:
            sz = self._get_vocab_size(g, gather)
            vocab_sizes.append(sz)

        if any(v is None for v in vocab_sizes):
            # Cannot determine vocab sizes; use original order: word, pos, seg
            return gathers[0], gathers[1], gathers[2]

        # Sort by vocab size: largest=word, middle=position, smallest=segment
        sorted_gathers = sorted(zip(vocab_sizes, gathers), key=lambda x: x[0], reverse=True)
        word_gather = sorted_gathers[0][1]
        pos_gather = sorted_gathers[1][1]
        seg_gather = sorted_gathers[2][1]
        return word_gather, pos_gather, seg_gather

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        if node.op_type != "LayerNormalization" or node.domain != "":
            return self.none()

        # LayerNormalization must have scale (gamma) and bias (beta)
        if len(node.input) < 3 or not node.input[2]:
            return self.none(node, inspect.currentframe().f_lineno)

        # axis must be -1 (or last axis)
        axis = g.get_attribute(node, "axis", exc=False)
        axis_val = -1 if axis is None else axis.i
        if axis_val != -1:
            if g.has_rank(node.input[0]):
                rank = g.get_rank(node.input[0])
                if axis_val != rank - 1:
                    return self.none(node, inspect.currentframe().f_lineno)

        # The input to LayerNormalization must come from an Add node
        outer_add = g.node_before(node.input[0])
        if outer_add is None or outer_add.op_type != "Add" or outer_add.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # outer_add must be used only by this LayerNormalization
        if g.is_used_more_than_once(outer_add.output[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        # Try to find 3-embedding case: outer_add has one Gather and one Add input
        inner_add = None
        outer_seg_gather = None
        for i in range(2):
            candidate_add = g.node_before(outer_add.input[i])
            candidate_gather = g.node_before(outer_add.input[1 - i])
            if (
                candidate_add is not None
                and candidate_add.op_type == "Add"
                and candidate_add.domain == ""
                and candidate_gather is not None
                and candidate_gather.op_type == "Gather"
                and candidate_gather.domain == ""
            ):
                inner_add = candidate_add
                outer_seg_gather = candidate_gather
                break

        if inner_add is not None:
            # 3-embedding case
            # inner_add must not be used elsewhere
            if g.is_used_more_than_once(inner_add.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)

            # inner_add's inputs must both be Gather nodes
            inner_gathers = self._try_get_gather_from_add(g, inner_add)
            if inner_gathers is None:
                return self.none(node, inspect.currentframe().f_lineno)
            inner_gather_0, inner_gather_1 = inner_gathers

            # All 3 Gather nodes must have constant table inputs
            all_gathers = [inner_gather_0, inner_gather_1, outer_seg_gather]
            for gather in all_gathers:
                if not g.is_constant(gather.input[0]):
                    return self.none(node, inspect.currentframe().f_lineno)

            nodes = [inner_gather_0, inner_gather_1, outer_seg_gather, inner_add, outer_add, node]
            return MatchResult(self, nodes, self.apply, insert_at=node, comment="3-embedding")

        # Try 2-embedding case: outer_add's both inputs come from Gather nodes
        two_gathers = self._try_get_gather_from_add(g, outer_add)
        if two_gathers is None:
            return self.none(node, inspect.currentframe().f_lineno)
        gather_0, gather_1 = two_gathers

        # Both Gather nodes must have constant table inputs
        if not g.is_constant(gather_0.input[0]) or not g.is_constant(gather_1.input[0]):
            return self.none(node, inspect.currentframe().f_lineno)

        nodes = [gather_0, gather_1, None, outer_add, None, node]
        return MatchResult(self, nodes, self.apply, insert_at=node, comment="2-embedding")

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        gather_0: NodeProto,
        gather_1: NodeProto,
        gather_seg: Optional[NodeProto],
        inner_or_outer_add: NodeProto,
        outer_add: Optional[NodeProto],
        ln_node: NodeProto,
    ) -> List[NodeProto]:
        if gather_seg is None:
            # 2-embedding case: no segment embeddings
            # Determine word vs position by vocab size
            sz0 = self._get_vocab_size(g, gather_0)
            sz1 = self._get_vocab_size(g, gather_1)
            if sz0 is not None and sz1 is not None:
                if sz0 >= sz1:
                    word_gather, pos_gather = gather_0, gather_1
                else:
                    word_gather, pos_gather = gather_1, gather_0
            else:
                word_gather, pos_gather = gather_0, gather_1

            seg_gather = None
        else:
            # 3-embedding case
            all_gathers = [gather_0, gather_1, gather_seg]
            word_gather, pos_gather, seg_gather = self._assign_gather_roles(g, all_gathers)

        # Build inputs for EmbedLayerNormalization:
        # [input_ids, segment_ids, word_embedding, position_embedding,
        #  segment_embedding, gamma, beta]
        input_ids = word_gather.input[1]
        word_table = word_gather.input[0]
        pos_table = pos_gather.input[0]
        pos_ids = pos_gather.input[1]
        gamma = ln_node.input[1]
        beta = ln_node.input[2]

        nodes = []

        def _cast_to_int32_if_needed(name: str) -> str:
            """Inserts a Cast(INT32) node if the named input is INT64."""
            if not g.has_type(name):
                return name
            if g.get_type(name) != TensorProto.INT64:
                return name
            cast_out = g.unique_name(f"{name}_i32")
            cast_node = g.make_node(
                "Cast",
                [name],
                [cast_out],
                to=TensorProto.INT32,
                name=f"{self.__class__.__name__}--cast_i32--{ln_node.name}",
            )
            nodes.append(cast_node)
            return cast_out

        input_ids_i32 = _cast_to_int32_if_needed(input_ids)

        inputs = [input_ids_i32, "", word_table, pos_table, "", gamma, beta]
        if seg_gather is not None:
            seg_ids = seg_gather.input[1]
            seg_table = seg_gather.input[0]
            inputs[1] = _cast_to_int32_if_needed(seg_ids)
            inputs[4] = seg_table

        # If position_ids are not equal to input_ids, add them as optional input [8]
        if pos_ids != input_ids:
            pos_ids_i32 = _cast_to_int32_if_needed(pos_ids)
            # Pad optional inputs 7 (mask) as empty then position_ids at index 8
            inputs.extend(["", pos_ids_i32])

        mask_index_name = g.unique_name(f"{ln_node.name}_mask_index")
        new_node = g.make_node(
            "EmbedLayerNormalization",
            inputs,
            [ln_node.output[0], mask_index_name],
            name=f"{self.__class__.__name__}--{ln_node.name}",
            domain="com.microsoft",
        )

        # Copy epsilon attribute from LayerNormalization, defaulting to the ONNX
        # LayerNormalization default of 1e-5 if not explicitly set.
        epsilon = g.get_attribute(ln_node, "epsilon", exc=False)
        if epsilon is not None:
            new_node.attribute.append(epsilon)
        else:
            new_node.attribute.append(helper.make_attribute("epsilon", 1e-5))

        nodes.append(new_node)
        return nodes
