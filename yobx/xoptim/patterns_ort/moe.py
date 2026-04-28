import inspect
from typing import List, Optional, Tuple
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class MoEPattern(PatternOptimization):
    """
    Fuses the Mixture-of-Experts (MoE) computation pattern into a single
    ``com.microsoft.MoE`` node.

    The pattern matches a standard top-k expert dispatch with two FC layers
    and an element-wise activation between them.  The routing probabilities
    must already be computed (e.g. via ``Softmax``) before the pattern.

    Model with nodes to be fused (k=1, relu, both biases present):

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_input(["input FLOAT(T, H)"])
            I_rp(["router_probs FLOAT(T, E)"])
            I_fc1w(["fc1_w FLOAT(E, I, H)"])
            I_fc1b(["fc1_b FLOAT(E, I)"])
            I_fc2w(["fc2_w FLOAT(E, H, I)"])
            I_fc2b(["fc2_b FLOAT(E, H)"])

            TopK_0[["TopK(., k)"]]
            Reshape_ids[["Reshape(top_indices, (T,))"]]
            Reshape_w[["Reshape(top_weights, (T, 1))"]]
            Gather_fc1w[["Gather(fc1_w, flat_ids, 0)"]]
            Gather_fc1b[["Gather(fc1_b, flat_ids, 0)"]]
            Transpose_fc1w[["Transpose(., [0,2,1])"]]
            Unsqueeze_in[["Unsqueeze(input, [1])"]]
            MatMul_fc1[["MatMul(., .)"]]
            Squeeze_fc1[["Squeeze(., [1])"]]
            Add_fc1[["Add(., .)"]]
            Activation_0[["Relu/Gelu/Silu(.)"]]
            Gather_fc2w[["Gather(fc2_w, flat_ids, 0)"]]
            Gather_fc2b[["Gather(fc2_b, flat_ids, 0)"]]
            Transpose_fc2w[["Transpose(., [0,2,1])"]]
            Unsqueeze_h1[["Unsqueeze(., [1])"]]
            MatMul_fc2[["MatMul(., .)"]]
            Squeeze_fc2[["Squeeze(., [1])"]]
            Add_fc2[["Add(., .)"]]
            Mul_out[["Mul(., .)"]]

            I_rp -->|"FLOAT(T, E)"| TopK_0
            TopK_0 -->|"weights FLOAT(T, 1)"| Reshape_w
            TopK_0 -->|"indices INT64(T, 1)"| Reshape_ids
            Reshape_ids -->|"INT64(T,)"| Gather_fc1w
            Reshape_ids -->|"INT64(T,)"| Gather_fc1b
            Reshape_ids -->|"INT64(T,)"| Gather_fc2w
            Reshape_ids -->|"INT64(T,)"| Gather_fc2b
            I_fc1w --> Gather_fc1w
            I_fc1b --> Gather_fc1b
            I_fc2w --> Gather_fc2w
            I_fc2b --> Gather_fc2b
            Gather_fc1w --> Transpose_fc1w
            I_input --> Unsqueeze_in
            Unsqueeze_in --> MatMul_fc1
            Transpose_fc1w --> MatMul_fc1
            MatMul_fc1 --> Squeeze_fc1
            Squeeze_fc1 --> Add_fc1
            Gather_fc1b --> Add_fc1
            Add_fc1 --> Activation_0
            Gather_fc2w --> Transpose_fc2w
            Activation_0 --> Unsqueeze_h1
            Unsqueeze_h1 --> MatMul_fc2
            Transpose_fc2w --> MatMul_fc2
            MatMul_fc2 --> Squeeze_fc2
            Squeeze_fc2 --> Add_fc2
            Gather_fc2b --> Add_fc2
            Add_fc2 --> Mul_out
            Reshape_w --> Mul_out

            O_out(["output FLOAT(T, H)"])
            Mul_out --> O_out

            class I_input,I_rp,I_fc1w,I_fc1b,I_fc2w,I_fc2b,O_out ioNode
            class TopK_0,Reshape_ids,Reshape_w,Gather_fc1w,Gather_fc1b opNode
            class Transpose_fc1w,Unsqueeze_in,MatMul_fc1,Squeeze_fc1,Add_fc1 opNode
            class Activation_0,Gather_fc2w,Gather_fc2b,Transpose_fc2w opNode
            class Unsqueeze_h1,MatMul_fc2,Squeeze_fc2,Add_fc2,Mul_out opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_input(["input FLOAT(T, H)"])
            I_rp(["router_probs FLOAT(T, E)"])
            I_fc1w(["fc1_w FLOAT(E, I, H)"])
            I_fc1b(["fc1_b FLOAT(E, I)"])
            I_fc2w(["fc2_w FLOAT(E, H, I)"])
            I_fc2b(["fc2_b FLOAT(E, H)"])

            MoE_0[["com.microsoft.MoE(., ., ., ., ., .)"]]

            I_input --> MoE_0
            I_rp --> MoE_0
            I_fc1w --> MoE_0
            I_fc1b --> MoE_0
            I_fc2w --> MoE_0
            I_fc2b --> MoE_0

            O_out(["output FLOAT(T, H)"])
            MoE_0 --> O_out

            class I_input,I_rp,I_fc1w,I_fc1b,I_fc2w,I_fc2b,O_out ioNode
            class MoE_0 opNode
    """

    _ACTIVATION_MAP = {"Relu": "relu", "Gelu": "gelu", "Silu": "silu"}

    def __init__(self, verbose: int = 0, priority: int = 2):
        super().__init__(verbose, priority)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _unsqueeze_axes(
        self, g: "GraphBuilderPatternOptimization", node: NodeProto  # noqa: F821
    ) -> Optional[Tuple[int, ...]]:
        """Returns the axes of an Unsqueeze node as a tuple, or None on failure."""
        axes = g.get_constant_or_attribute(node, "axes", input_index=1, cvt=tuple)
        if axes is None:
            return None
        return tuple(int(a) for a in axes)

    def _squeeze_axes(
        self, g: "GraphBuilderPatternOptimization", node: NodeProto  # noqa: F821
    ) -> Optional[Tuple[int, ...]]:
        """Returns the axes of a Squeeze node as a tuple, or None on failure."""
        if len(node.input) < 2:
            return None
        axes = g.get_constant_or_attribute(node, "axes", input_index=1, cvt=tuple)
        if axes is None:
            return None
        return tuple(int(a) for a in axes)

    def _is_gather_from(
        self, g: "GraphBuilderPatternOptimization", gather: NodeProto, flat_ids: str  # noqa: F821
    ) -> bool:
        """Checks whether *gather* is Gather(data, flat_ids, axis=0)."""
        if gather.op_type != "Gather" or gather.domain != "":
            return False
        if gather.input[1] != flat_ids:
            return False
        axis = g.get_attribute_with_default(gather, "axis", 0)
        return axis == 0

    # ------------------------------------------------------------------
    # match
    # ------------------------------------------------------------------

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        # Anchor: the final Mul that blends the expert output with the routing weight.
        if node.op_type != "Mul" or node.domain != "":
            return self.none()

        # ---- identify routing-weight input vs. expert-output input --------
        # One input to Mul is the routing weight (shape (T, 1)) coming from a
        # Reshape node; the other is the expert output (shape (T, H)).
        r0 = g.node_before(node.input[0])
        r1 = g.node_before(node.input[1])

        if r0 is not None and r0.op_type == "Reshape":
            routing_reshape = r0
            expert_out_name = node.input[1]
        elif r1 is not None and r1.op_type == "Reshape":
            routing_reshape = r1
            expert_out_name = node.input[0]
        else:
            return self.none(node, inspect.currentframe().f_lineno)

        # ---- routing path: Reshape <- TopK <- router_probs ----------------
        topk_node = g.node_before(routing_reshape.input[0])
        if topk_node is None or topk_node.op_type != "TopK" or topk_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # TopK output[0] = weights, output[1] = indices.
        # routing_reshape must consume output[0] (weights).
        if routing_reshape.input[0] != topk_node.output[0]:
            return self.none(node, inspect.currentframe().f_lineno)

        # k must be a compile-time constant.
        topk_k_name = topk_node.input[1] if len(topk_node.input) > 1 else None
        if topk_k_name is None or not g.is_constant(topk_k_name):
            return self.none(node, inspect.currentframe().f_lineno)
        topk_k_val = g.get_computed_constant(topk_k_name)
        if topk_k_val is None:
            return self.none(node, inspect.currentframe().f_lineno)
        k_int = int(topk_k_val.flatten()[0])
        if k_int < 1:
            return self.none(node, inspect.currentframe().f_lineno)

        # ---- flat_ids: Reshape of TopK indices ---------------------------
        topk_indices_name = topk_node.output[1]
        ids_consumers = g.next_nodes(topk_indices_name)
        if len(ids_consumers) != 1:
            return self.none(node, inspect.currentframe().f_lineno)
        ids_reshape = ids_consumers[0]
        if ids_reshape.op_type != "Reshape" or ids_reshape.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        flat_ids = ids_reshape.output[0]

        # ---- expert output side: optional Add(fc2_out, fc2_bias) ----------
        expert_out_node = g.node_before(expert_out_name)
        fc2_bias_gather_node: Optional[NodeProto] = None
        fc2_add_node: Optional[NodeProto] = None

        if expert_out_node is not None and expert_out_node.op_type == "Add":
            fc2_sq_cand_a = expert_out_node.input[0]
            fc2_sq_cand_b = expert_out_node.input[1]
            node_a = g.node_before(fc2_sq_cand_a)
            node_b = g.node_before(fc2_sq_cand_b)

            if node_a is not None and self._is_gather_from(g, node_a, flat_ids):
                fc2_squeeze_name = fc2_sq_cand_b
                fc2_bias_gather_node = node_a
            elif node_b is not None and self._is_gather_from(g, node_b, flat_ids):
                fc2_squeeze_name = fc2_sq_cand_a
                fc2_bias_gather_node = node_b
            else:
                return self.none(node, inspect.currentframe().f_lineno)
            fc2_add_node = expert_out_node
        else:
            fc2_squeeze_name = expert_out_name

        # ---- fc2 squeeze -------------------------------------------------
        fc2_squeeze = g.node_before(fc2_squeeze_name)
        if fc2_squeeze is None or fc2_squeeze.op_type != "Squeeze" or fc2_squeeze.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        sq_axes = self._squeeze_axes(g, fc2_squeeze)
        if sq_axes not in ((1,), (-2,)):
            return self.none(node, inspect.currentframe().f_lineno)

        # ---- fc2 matmul --------------------------------------------------
        fc2_matmul = g.node_before(fc2_squeeze.input[0])
        if fc2_matmul is None or fc2_matmul.op_type != "MatMul" or fc2_matmul.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        fc2_matmul_inp_a = fc2_matmul.input[0]
        fc2_matmul_inp_b = fc2_matmul.input[1]

        # ---- fc2 weight transpose ----------------------------------------
        fc2_w_transpose = g.node_before(fc2_matmul_inp_b)
        if (
            fc2_w_transpose is None
            or fc2_w_transpose.op_type != "Transpose"
            or fc2_w_transpose.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # ---- fc2 weight gather -------------------------------------------
        fc2_w_gather = g.node_before(fc2_w_transpose.input[0])
        if fc2_w_gather is None or not self._is_gather_from(g, fc2_w_gather, flat_ids):
            return self.none(node, inspect.currentframe().f_lineno)

        # ---- fc1 activation unsqueeze ------------------------------------
        fc1_act_unsqueeze = g.node_before(fc2_matmul_inp_a)
        if (
            fc1_act_unsqueeze is None
            or fc1_act_unsqueeze.op_type != "Unsqueeze"
            or fc1_act_unsqueeze.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        usq_axes = self._unsqueeze_axes(g, fc1_act_unsqueeze)
        if usq_axes not in ((1,), (-2,)):
            return self.none(node, inspect.currentframe().f_lineno)

        fc1_activated_name = fc1_act_unsqueeze.input[0]

        # ---- activation --------------------------------------------------
        act_node = g.node_before(fc1_activated_name)
        if act_node is None or act_node.op_type not in self._ACTIVATION_MAP:
            return self.none(node, inspect.currentframe().f_lineno)
        fc1_biased_name = act_node.input[0]

        # ---- fc1 bias (optional) -----------------------------------------
        fc1_add_node: Optional[NodeProto] = None
        fc1_bias_gather_node: Optional[NodeProto] = None
        fc1_add_pre = g.node_before(fc1_biased_name)
        if fc1_add_pre is not None and fc1_add_pre.op_type == "Add":
            fc1_sq_cand_a = fc1_add_pre.input[0]
            fc1_sq_cand_b = fc1_add_pre.input[1]
            node_fa = g.node_before(fc1_sq_cand_a)
            node_fb = g.node_before(fc1_sq_cand_b)
            if node_fa is not None and self._is_gather_from(g, node_fa, flat_ids):
                fc1_squeeze_name = fc1_sq_cand_b
                fc1_bias_gather_node = node_fa
            elif node_fb is not None and self._is_gather_from(g, node_fb, flat_ids):
                fc1_squeeze_name = fc1_sq_cand_a
                fc1_bias_gather_node = node_fb
            else:
                return self.none(node, inspect.currentframe().f_lineno)
            fc1_add_node = fc1_add_pre
        else:
            fc1_squeeze_name = fc1_biased_name

        # ---- fc1 squeeze -------------------------------------------------
        fc1_squeeze = g.node_before(fc1_squeeze_name)
        if fc1_squeeze is None or fc1_squeeze.op_type != "Squeeze" or fc1_squeeze.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        sq_axes_1 = self._squeeze_axes(g, fc1_squeeze)
        if sq_axes_1 not in ((1,), (-2,)):
            return self.none(node, inspect.currentframe().f_lineno)

        # ---- fc1 matmul --------------------------------------------------
        fc1_matmul = g.node_before(fc1_squeeze.input[0])
        if fc1_matmul is None or fc1_matmul.op_type != "MatMul" or fc1_matmul.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        fc1_matmul_inp_a = fc1_matmul.input[0]
        fc1_matmul_inp_b = fc1_matmul.input[1]

        # ---- fc1 weight transpose ----------------------------------------
        fc1_w_transpose = g.node_before(fc1_matmul_inp_b)
        if (
            fc1_w_transpose is None
            or fc1_w_transpose.op_type != "Transpose"
            or fc1_w_transpose.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)

        # ---- fc1 weight gather -------------------------------------------
        fc1_w_gather = g.node_before(fc1_w_transpose.input[0])
        if fc1_w_gather is None or not self._is_gather_from(g, fc1_w_gather, flat_ids):
            return self.none(node, inspect.currentframe().f_lineno)

        # ---- input unsqueeze ---------------------------------------------
        input_unsqueeze = g.node_before(fc1_matmul_inp_a)
        if (
            input_unsqueeze is None
            or input_unsqueeze.op_type != "Unsqueeze"
            or input_unsqueeze.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        usq_axes_in = self._unsqueeze_axes(g, input_unsqueeze)
        if usq_axes_in not in ((1,), (-2,)):
            return self.none(node, inspect.currentframe().f_lineno)

        # ---- collect all nodes in fixed positional order ------------------
        # None is used as a placeholder for optional nodes that are absent.
        # The order must match the apply() signature exactly.
        all_nodes: List[Optional[NodeProto]] = [
            topk_node,  # 0  _IDX_TOPK
            ids_reshape,  # 1  _IDX_IDS_RESHAPE
            routing_reshape,  # 2  _IDX_ROUTING_RESHAPE
            input_unsqueeze,  # 3  _IDX_INPUT_UNSQUEEZE
            fc1_w_gather,  # 4  _IDX_FC1_W_GATHER
            fc1_w_transpose,  # 5  _IDX_FC1_W_TRANSPOSE
            fc1_matmul,  # 6  _IDX_FC1_MATMUL
            fc1_squeeze,  # 7  _IDX_FC1_SQUEEZE
            fc1_bias_gather_node,  # 8  _IDX_FC1_BIAS_GATHER (may be None)
            fc1_add_node,  # 9  _IDX_FC1_ADD         (may be None)
            act_node,  # 10 _IDX_ACT
            fc1_act_unsqueeze,  # 11 _IDX_FC1_ACT_UNSQUEEZE
            fc2_w_gather,  # 12 _IDX_FC2_W_GATHER
            fc2_w_transpose,  # 13 _IDX_FC2_W_TRANSPOSE
            fc2_matmul,  # 14 _IDX_FC2_MATMUL
            fc2_squeeze,  # 15 _IDX_FC2_SQUEEZE
            fc2_bias_gather_node,  # 16 _IDX_FC2_BIAS_GATHER (may be None)
            fc2_add_node,  # 17 _IDX_FC2_ADD         (may be None)
            node,  # 18 _IDX_MUL (anchor)
        ]

        return MatchResult(self, all_nodes, self.apply, insert_at=node)

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

    def apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        topk_node: NodeProto,
        ids_reshape: NodeProto,
        routing_reshape: NodeProto,
        input_unsqueeze: NodeProto,
        fc1_w_gather: NodeProto,
        fc1_w_transpose: NodeProto,
        fc1_matmul: NodeProto,
        fc1_squeeze: NodeProto,
        fc1_bias_gather: Optional[NodeProto],
        fc1_add: Optional[NodeProto],
        act_node: NodeProto,
        fc1_act_unsqueeze: NodeProto,
        fc2_w_gather: NodeProto,
        fc2_w_transpose: NodeProto,
        fc2_matmul: NodeProto,
        fc2_squeeze: NodeProto,
        fc2_bias_gather: Optional[NodeProto],
        fc2_add: Optional[NodeProto],
        mul_node: NodeProto,
    ) -> List[NodeProto]:
        """Replaces the matched expert-computation sub-graph with one MoE node."""
        # Recover the original source tensor names from the matched nodes.
        input_name = input_unsqueeze.input[0]
        router_probs = topk_node.input[0]
        fc1_weights = fc1_w_gather.input[0]
        fc1_bias_source = fc1_bias_gather.input[0] if fc1_bias_gather is not None else None
        fc2_weights = fc2_w_gather.input[0]
        fc2_bias_source = fc2_bias_gather.input[0] if fc2_bias_gather is not None else None

        # Recover the k constant from the TopK node.
        k_val = g.get_computed_constant(topk_node.input[1])
        k_int = int(k_val.flatten()[0])

        # Map the activation op_type to the string expected by MoE.
        activation_type = self._ACTIVATION_MAP[act_node.op_type]

        # The fused node writes to the same output as the anchor Mul.
        output_name = mul_node.output[0]

        # Build the input list:
        #   [0] input
        #   [1] router_probs
        #   [2] fc1_experts_weights
        #   [3] fc1_experts_bias  (empty string = not present)
        #   [4] fc2_experts_weights
        #   [5] fc2_experts_bias  (optional, omit if absent)
        inputs = [input_name, router_probs, fc1_weights]
        inputs.append(fc1_bias_source if fc1_bias_source is not None else "")
        inputs.append(fc2_weights)
        if fc2_bias_source is not None:
            inputs.append(fc2_bias_source)

        moe_node = g.make_node(
            "MoE",
            inputs,
            [output_name],
            domain="com.microsoft",
            k=k_int,
            activation_type=activation_type,
            normalize_routing_weights=1,
            name=f"{self.__class__.__name__}--{mul_node.name}",
        )
        return [moe_node]
