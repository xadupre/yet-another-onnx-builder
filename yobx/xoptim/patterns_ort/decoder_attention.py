import inspect
from typing import List, Optional, Tuple
import numpy as np
from onnx import NodeProto, TensorProto
from ..patterns_api import MatchResult, PatternOptimization


class DecoderAttentionPattern(PatternOptimization):
    """
    Fuses a sequence-first decoder cross-attention or self-attention computation
    into ``com.microsoft.DecoderAttention``.

    The operator expects inputs in **sequence-first** format ``(S, B, H)``
    (sequence length, batch size, hidden size) and separate weight matrices:

    * ``query``      – ``(S, B, H)``
    * ``key``        – ``(T, B, H)``  (same as query for self-attention)
    * ``q_weight``   – ``(H, H)``
    * ``kv_weight``  – ``(H, 2*H)``  (K and V weights concatenated)
    * ``bias``       – ``(3*H,)``     (Q, K, V biases concatenated)
    * ``static_kv``  – bool scalar: ``True`` for cross-attention, ``False`` for self-attention
    * ``use_past``   – bool scalar: ``False`` (no KV-cache in this pattern)
    * ``has_layer_state`` – bool scalar: ``False``
    * ``has_key_padding_mask`` – bool scalar: ``False``

    **Cross-attention** is detected when the source of the Q projection differs
    from the source of the K/V projections.

    Model with nodes to be fused (seq-first cross-attention, no cache, no mask):

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_query(["query FLOAT(S, B, H)"])
            I_key(["key FLOAT(T, B, H)"])
            W_q(["q_weight FLOAT(H, H)"])
            W_k(["k_weight FLOAT(H, H)"])
            W_v(["v_weight FLOAT(H, H)"])
            B_q(["q_bias FLOAT(H)"])
            B_k(["k_bias FLOAT(H)"])
            B_v(["v_bias FLOAT(H)"])

            MM_q[["MatMul(query, q_weight)"]]
            Add_q[["Add(mm_q, q_bias)"]]
            Re_q[["Reshape(., [0,0,N,d])"]]
            Tr_q[["Transpose(., perm=[1,2,0,3])"]]

            MM_k[["MatMul(key, k_weight)"]]
            Add_k[["Add(mm_k, k_bias)"]]
            Re_k[["Reshape(., [0,0,N,d])"]]
            Tr_k[["Transpose(., perm=[1,2,0,3])"]]
            Tr_kt[["Transpose(., perm=[0,1,3,2])"]]

            MM_v[["MatMul(key, v_weight)"]]
            Add_v[["Add(mm_v, v_bias)"]]
            Re_v[["Reshape(., [0,0,N,d])"]]
            Tr_v[["Transpose(., perm=[1,2,0,3])"]]

            Mul_scale[["Mul(Q, scale)"]]
            MM_qk[["MatMul(Q_scaled, K_T)"]]
            Softmax[["Softmax(., axis=-1)"]]
            MM_qkv[["MatMul(attn_probs, V)"]]

            Tr_out[["Transpose(., perm=[2,0,1,3])"]]
            Re_out[["Reshape(., [0,0,-1])"]]

            I_query --> MM_q --> Add_q --> Re_q --> Tr_q
            W_q --> MM_q
            B_q --> Add_q

            I_key --> MM_k --> Add_k --> Re_k --> Tr_k --> Tr_kt
            W_k --> MM_k
            B_k --> Add_k

            I_key --> MM_v --> Add_v --> Re_v --> Tr_v
            W_v --> MM_v
            B_v --> Add_v

            Tr_q --> Mul_scale
            Mul_scale --> MM_qk
            Tr_kt --> MM_qk
            MM_qk --> Softmax --> MM_qkv
            Tr_v --> MM_qkv

            MM_qkv --> Tr_out --> Re_out

            O_output(["output FLOAT(S, B, H)"])
            Re_out --> O_output

            class I_query,I_key,O_output ioNode
            class W_q,W_k,W_v,B_q,B_k,B_v initNode
            class MM_q,Add_q,Re_q,Tr_q,MM_k,Add_k,Re_k,Tr_k,Tr_kt opNode
            class MM_v,Add_v,Re_v,Tr_v,Mul_scale,MM_qk,Softmax,MM_qkv opNode
            class Tr_out,Re_out opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_query(["query FLOAT(S, B, H)"])
            I_key(["key FLOAT(T, B, H)"])
            W_q(["q_weight FLOAT(H, H)"])
            W_kv(["kv_weight FLOAT(H, 2*H)"])
            B_qkv(["bias FLOAT(3*H)"])
            C_static_kv(["static_kv BOOL()"])
            C_use_past(["use_past BOOL()"])
            C_has_state(["has_layer_state BOOL()"])
            C_has_mask(["has_key_padding_mask BOOL()"])

            DecoderAttention_0[["com.microsoft.DecoderAttention(., ., ., ., ., , , , ., ., .,
            .)"]]

            I_query --> DecoderAttention_0
            I_key --> DecoderAttention_0
            W_q --> DecoderAttention_0
            W_kv --> DecoderAttention_0
            B_qkv --> DecoderAttention_0
            C_static_kv --> DecoderAttention_0
            C_use_past --> DecoderAttention_0
            C_has_state --> DecoderAttention_0
            C_has_mask --> DecoderAttention_0

            O_output(["output FLOAT(S, B, H)"])
            DecoderAttention_0 --> O_output

            class I_query,I_key,O_output ioNode
            class W_q,W_kv,B_qkv,C_static_kv,C_use_past,C_has_state,C_has_mask initNode
            class DecoderAttention_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 2):
        """Initializes the pattern with the given verbosity and matching priority."""
        super().__init__(verbose, priority)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_transpose_perm(
        self, g: "GraphBuilderPatternOptimization", node: NodeProto  # noqa: F821
    ) -> Optional[Tuple[int, ...]]:
        """Returns the perm attribute of a Transpose node as a tuple, or ``None``."""
        if node.op_type != "Transpose" or node.domain != "":
            return None
        perm = g.get_attribute(node, "perm")
        if perm is None:
            return None
        return tuple(int(p) for p in perm.ints)

    def _match_seq_first_proj(
        self, g: "GraphBuilderPatternOptimization", tensor_name: str  # noqa: F821
    ) -> Optional[
        Tuple[NodeProto, Optional[NodeProto], Optional[str], NodeProto, NodeProto, str]
    ]:
        """Matches a seq-first linear projection ending at *tensor_name*.

        Expected structure::

            Transpose([1,2,0,3]) ← Reshape ← [Add(bias)] ← MatMul(input, weight)

        Here the weight must be a constant initializer.

        Returns ``(mm, add_node, bias_name, reshape, transpose, src_name)``
        where ``add_node`` and ``bias_name`` are ``None`` when there is no bias.
        Returns ``None`` if the pattern does not match.
        """
        transpose = g.node_before(tensor_name)
        if transpose is None or transpose.op_type != "Transpose" or transpose.domain != "":
            return None
        if self._get_transpose_perm(g, transpose) != (1, 2, 0, 3):
            return None

        reshape = g.node_before(transpose.input[0])
        if reshape is None or reshape.op_type != "Reshape" or reshape.domain != "":
            return None
        if not g.is_constant(reshape.input[1]):
            return None

        before_reshape = g.node_before(reshape.input[0])
        if before_reshape is None:
            return None

        if before_reshape.op_type == "Add" and before_reshape.domain == "":
            add_node = before_reshape
            for matmul_idx, bias_idx in ((0, 1), (1, 0)):
                if not g.is_constant(add_node.input[bias_idx]):
                    continue
                mm = g.node_before(add_node.input[matmul_idx])
                if mm is None or mm.op_type != "MatMul" or mm.domain != "":
                    continue
                if not g.is_constant(mm.input[1]):
                    continue
                return mm, add_node, add_node.input[bias_idx], reshape, transpose, mm.input[0]
            return None
        elif before_reshape.op_type == "MatMul" and before_reshape.domain == "":
            mm = before_reshape
            if not g.is_constant(mm.input[1]):
                return None
            return mm, None, None, reshape, transpose, mm.input[0]
        else:
            return None

    # ------------------------------------------------------------------
    # match
    # ------------------------------------------------------------------

    def match(
        self,
        g: "GraphBuilderPatternOptimization",  # noqa: F821
        node: NodeProto,
        matched: List[MatchResult],
    ) -> Optional[MatchResult]:
        """Attempts to match starting from the output Reshape node."""
        # Anchor: Reshape with constant shape that produces (S, B, H) output.
        if node.op_type != "Reshape" or node.domain != "":
            return self.none()
        if not g.is_constant(node.input[1]):
            return self.none(node, inspect.currentframe().f_lineno)

        # Check for Transpose([2,0,1,3]) immediately before this Reshape.
        transpose_out = g.node_before(node.input[0])
        if (
            transpose_out is None
            or transpose_out.op_type != "Transpose"
            or transpose_out.domain != ""
        ):
            return self.none(node, inspect.currentframe().f_lineno)
        if self._get_transpose_perm(g, transpose_out) != (2, 0, 1, 3):
            return self.none(node, inspect.currentframe().f_lineno)

        # Before the Transpose: MatMul(attn_probs, V) possibly with an
        # optional Where(IsNaN(...), 0, attn_probs) in between.
        attn_qv_cand = g.node_before(transpose_out.input[0])
        if attn_qv_cand is None:
            return self.none(node, inspect.currentframe().f_lineno)

        nan_where_node: Optional[NodeProto] = None
        is_nan_node: Optional[NodeProto] = None
        if attn_qv_cand.op_type == "Where" and attn_qv_cand.domain == "":
            # Where(IsNaN(softmax_out), 0, softmax_out)
            nan_where_node = attn_qv_cand
            is_nan_cand = g.node_before(nan_where_node.input[0])
            if is_nan_cand is None or is_nan_cand.op_type != "IsNaN":
                return self.none(node, inspect.currentframe().f_lineno)
            is_nan_node = is_nan_cand
            matmul_qv = g.node_before(nan_where_node.input[2])
            if matmul_qv is None or matmul_qv.op_type != "MatMul" or matmul_qv.domain != "":
                return self.none(node, inspect.currentframe().f_lineno)
        elif attn_qv_cand.op_type == "MatMul" and attn_qv_cand.domain == "":
            matmul_qv = attn_qv_cand
        else:
            return self.none(node, inspect.currentframe().f_lineno)

        # V projection: Transpose([1,2,0,3]) → Reshape → [Add(bias)] → MatMul
        v_result = self._match_seq_first_proj(g, matmul_qv.input[1])
        if v_result is None:
            return self.none(node, inspect.currentframe().f_lineno)
        mm_v, add_v, bias_v_name, reshape_v, transpose_v, key_v_src = v_result

        # attn_probs from Softmax (possibly after Where for NaN handling).
        softmax_input = matmul_qv.input[0]
        softmax_cand = g.node_before(softmax_input)
        if softmax_cand is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # Handle: attn_probs = Where(IsNaN(s), 0, s) where s is from Softmax.
        pre_softmax_where: Optional[NodeProto] = None
        pre_softmax_is_nan: Optional[NodeProto] = None
        if softmax_cand.op_type == "Where" and softmax_cand.domain == "":
            pre_softmax_where = softmax_cand
            pre_softmax_is_nan_cand = g.node_before(pre_softmax_where.input[0])
            if pre_softmax_is_nan_cand is None or pre_softmax_is_nan_cand.op_type != "IsNaN":
                return self.none(node, inspect.currentframe().f_lineno)
            pre_softmax_is_nan = pre_softmax_is_nan_cand
            softmax_cand = g.node_before(pre_softmax_where.input[2])
            if softmax_cand is None:
                return self.none(node, inspect.currentframe().f_lineno)

        softmax = softmax_cand
        if softmax.op_type != "Softmax" or softmax.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        axis_attr = g.get_attribute(softmax, "axis")
        if axis_attr is None or int(axis_attr.i) != -1:
            return self.none(node, inspect.currentframe().f_lineno)

        # Before Softmax: QK^T logits (possibly with a mask Add).
        before_softmax = g.node_before(softmax.input[0])
        if before_softmax is None:
            return self.none(node, inspect.currentframe().f_lineno)

        mask_add_node: Optional[NodeProto] = None
        if before_softmax.op_type == "Add" and before_softmax.domain == "":
            mask_add_node = before_softmax
            # One input is QK^T MatMul, other is a float mask.
            matmul_qk: Optional[NodeProto] = None
            for idx in (0, 1):
                cand = g.node_before(mask_add_node.input[idx])
                if cand is not None and cand.op_type == "MatMul" and cand.domain == "":
                    matmul_qk = cand
                    break
            if matmul_qk is None:
                return self.none(node, inspect.currentframe().f_lineno)
        elif before_softmax.op_type == "MatMul" and before_softmax.domain == "":
            matmul_qk = before_softmax
        else:
            return self.none(node, inspect.currentframe().f_lineno)

        # matmul_qk: MatMul(Q_scaled, K_T)
        # input[1] should be K_T = Transpose(K, perm=[0,1,3,2])
        k_T_node = g.node_before(matmul_qk.input[1])
        if k_T_node is None or k_T_node.op_type != "Transpose" or k_T_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        if self._get_transpose_perm(g, k_T_node) != (0, 1, 3, 2):
            return self.none(node, inspect.currentframe().f_lineno)

        # K projection: Transpose([1,2,0,3]) → Reshape → [Add] → MatMul
        k_result = self._match_seq_first_proj(g, k_T_node.input[0])
        if k_result is None:
            return self.none(node, inspect.currentframe().f_lineno)
        mm_k, add_k, bias_k_name, reshape_k, transpose_k, key_k_src = k_result

        # K and V must come from the same source.
        if key_k_src != key_v_src:
            return self.none(node, inspect.currentframe().f_lineno)

        # Q_scaled: Mul(Q, scale)
        mul_q_scale = g.node_before(matmul_qk.input[0])
        if mul_q_scale is None or mul_q_scale.op_type != "Mul" or mul_q_scale.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        scale_name: Optional[str] = None
        q_4d_name: Optional[str] = None
        if g.is_constant_scalar(mul_q_scale.input[0]):
            scale_name = mul_q_scale.input[0]
            q_4d_name = mul_q_scale.input[1]
        elif g.is_constant_scalar(mul_q_scale.input[1]):
            scale_name = mul_q_scale.input[1]
            q_4d_name = mul_q_scale.input[0]
        else:
            return self.none(node, inspect.currentframe().f_lineno)

        # Q projection: Transpose([1,2,0,3]) → Reshape → [Add] → MatMul
        q_result = self._match_seq_first_proj(g, q_4d_name)
        if q_result is None:
            return self.none(node, inspect.currentframe().f_lineno)
        mm_q, add_q, bias_q_name, reshape_q, transpose_q, query_src = q_result

        # Weights must have shape (H, H).
        for mm_node in (mm_q, mm_k, mm_v):
            if not g.has_shape(mm_node.input[1]):
                return self.none(node, inspect.currentframe().f_lineno)
            w_shape = g.get_shape(mm_node.input[1])
            if len(w_shape) != 2:
                return self.none(node, inspect.currentframe().f_lineno)

        # Infer num_heads from the Q reshape shape [0, 0, N, d].
        num_heads: Optional[int] = None
        if g.is_constant(reshape_q.input[1]):
            shape_cst = g.get_computed_constant(reshape_q.input[1])
            if shape_cst is not None and shape_cst.ndim == 1 and len(shape_cst) == 4:
                num_heads = int(shape_cst[2])
        if num_heads is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # Determine static_kv: True for cross-attention, False for self-attention.
        static_kv = query_src != key_k_src

        # Uniqueness: the intermediate tensors must not be shared.
        single_use_nodes = [
            transpose_q,
            reshape_q,
            mul_q_scale,
            transpose_k,
            reshape_k,
            k_T_node,
            transpose_v,
            reshape_v,
            matmul_qk,
            softmax,
            matmul_qv,
            transpose_out,
        ]
        if add_q is not None:
            single_use_nodes.append(add_q)
        if add_k is not None:
            single_use_nodes.append(add_k)
        if add_v is not None:
            single_use_nodes.append(add_v)
        if pre_softmax_where is not None:
            single_use_nodes.append(pre_softmax_where)
        if nan_where_node is not None:
            single_use_nodes.append(nan_where_node)
        for n in single_use_nodes:
            if g.is_used_more_than_once(n.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)

        # Capture info needed by apply() in a closure.
        _query_src = query_src
        _key_src = key_k_src
        _q_weight = mm_q.input[1]
        _k_weight = mm_k.input[1]
        _v_weight = mm_v.input[1]
        _q_bias = bias_q_name
        _k_bias = bias_k_name
        _v_bias = bias_v_name
        _scale_name = scale_name
        _num_heads = num_heads
        _static_kv = static_kv
        _output_name = node.output[0]
        _anchor_name = node.name

        def _apply(g_builder, *_nodes):
            return self._do_apply(
                g_builder,
                query_src=_query_src,
                key_src=_key_src,
                q_weight=_q_weight,
                k_weight=_k_weight,
                v_weight=_v_weight,
                q_bias=_q_bias,
                k_bias=_k_bias,
                v_bias=_v_bias,
                scale_name=_scale_name,
                num_heads=_num_heads,
                static_kv=_static_kv,
                output_name=_output_name,
                anchor_name=_anchor_name,
            )

        all_nodes: List[Optional[NodeProto]] = [
            mm_q,
            add_q,
            reshape_q,
            transpose_q,
            mm_k,
            add_k,
            reshape_k,
            transpose_k,
            k_T_node,
            mm_v,
            add_v,
            reshape_v,
            transpose_v,
            mul_q_scale,
            matmul_qk,
            mask_add_node,
            softmax,
            pre_softmax_where,
            pre_softmax_is_nan,
            nan_where_node,
            is_nan_node,
            matmul_qv,
            transpose_out,
            node,
        ]

        return MatchResult(self, [n for n in all_nodes if n is not None], _apply, insert_at=node)

    # ------------------------------------------------------------------
    # _do_apply
    # ------------------------------------------------------------------

    def _do_apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        query_src: str,
        key_src: str,
        q_weight: str,
        k_weight: str,
        v_weight: str,
        q_bias: Optional[str],
        k_bias: Optional[str],
        v_bias: Optional[str],
        scale_name: str,
        num_heads: int,
        static_kv: bool,
        output_name: str,
        anchor_name: str,
    ) -> List[NodeProto]:
        """Creates and returns the ``com.microsoft.DecoderAttention`` node."""
        nodes = []
        cls_name = self.__class__.__name__

        # kv_weight = Concat(k_weight, v_weight, axis=1) → (H, 2H)
        kv_weight_name = g.unique_name(f"{cls_name}--kv_weight")
        nodes.append(
            g.make_node(
                "Concat",
                [k_weight, v_weight],
                [kv_weight_name],
                axis=1,
                name=f"{cls_name}--{anchor_name}--kv_concat",
            )
        )

        # bias = Concat(q_bias, k_bias, v_bias, axis=0) → (3H,)
        # If any bias is missing, create a zero bias vector from the weight shapes.
        dtype = g.get_type(query_src) if g.has_type(query_src) else TensorProto.FLOAT
        if q_bias is not None and k_bias is not None and v_bias is not None:
            bias_name = g.unique_name(f"{cls_name}--bias")
            nodes.append(
                g.make_node(
                    "Concat",
                    [q_bias, k_bias, v_bias],
                    [bias_name],
                    axis=0,
                    name=f"{cls_name}--{anchor_name}--bias_concat",
                )
            )
        else:
            # Derive bias size from weight shapes.
            q_w_cst = g.get_computed_constant(q_weight)
            k_w_cst = g.get_computed_constant(k_weight)
            v_w_cst = g.get_computed_constant(v_weight)
            if q_w_cst is not None and k_w_cst is not None and v_w_cst is not None:
                hidden_size = int(q_w_cst.shape[1])
                np_dtype = np.float16 if dtype == TensorProto.FLOAT16 else np.float32
                zero_bias = np.zeros(3 * hidden_size, dtype=np_dtype)
                bias_name = g.make_initializer("", zero_bias, source=f"{cls_name}.zero_bias")
            else:
                # Fallback: create the bias size from weight shape dynamically.
                # We rely on shape (H,) from q_weight column count × 3.
                w_shape_q = g.get_shape(q_weight) if g.has_shape(q_weight) else None
                if w_shape_q is not None and isinstance(w_shape_q[1], int):
                    hidden_size = int(w_shape_q[1])
                    np_dtype = np.float16 if dtype == TensorProto.FLOAT16 else np.float32
                    zero_bias = np.zeros(3 * hidden_size, dtype=np_dtype)
                    bias_name = g.make_initializer("", zero_bias, source=f"{cls_name}.zero_bias")
                else:
                    # Cannot determine bias size statically; skip fusion.
                    return []

        # Boolean scalar inputs.
        static_kv_init = g.make_initializer(
            "", np.array([static_kv], dtype=bool), source=f"{cls_name}.static_kv"
        )
        use_past_init = g.make_initializer(
            "", np.array([False], dtype=bool), source=f"{cls_name}.use_past"
        )
        has_layer_state_init = g.make_initializer(
            "", np.array([False], dtype=bool), source=f"{cls_name}.has_layer_state"
        )
        has_key_padding_mask_init = g.make_initializer(
            "", np.array([False], dtype=bool), source=f"{cls_name}.has_key_padding_mask"
        )

        nodes.append(
            g.make_node(
                "DecoderAttention",
                [
                    query_src,
                    key_src,
                    q_weight,
                    kv_weight_name,
                    bias_name,
                    "",  # key_padding_mask (absent)
                    "",  # key_cache (absent)
                    "",  # value_cache (absent)
                    static_kv_init,
                    use_past_init,
                    has_layer_state_init,
                    has_key_padding_mask_init,
                ],
                [output_name],
                num_heads=num_heads,
                domain="com.microsoft",
                name=f"{cls_name}--{anchor_name}",
            )
        )
        return nodes

    # apply is not called directly; the closure _apply defined in match() dispatches.
    def apply(self, g: "GraphBuilder", *nodes: NodeProto) -> List[NodeProto]:  # noqa: F821
        """Not called directly; the match closure handles dispatch."""
        raise NotImplementedError("DecoderAttentionPattern.apply should not be called directly")
