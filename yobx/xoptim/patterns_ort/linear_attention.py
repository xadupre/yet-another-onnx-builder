import inspect
from typing import List, Optional, Tuple
from onnx import NodeProto
from ..patterns_api import MatchResult, PatternOptimization


class LinearAttentionPattern(PatternOptimization):
    """
    Fuses a linear-attention recurrent state update into
    ``com.microsoft.LinearAttention``.

    The pattern supports two *update_rule* variants
    (``'linear'`` and ``'gated'``), which correspond exactly to the
    ``update_rule`` attribute of the ORT contrib op.

    **Inputs expected by the pattern** (all 3-D packed, i.e. batch-first with
    heads folded into the last dimension):

    * ``query``      – ``FLOAT(B, T, H_q * d_k)``
    * ``key``        – ``FLOAT(B, T, H_kv * d_k)``
    * ``value``      – ``FLOAT(B, T, H_kv * d_v)``
    * ``past_state`` (optional) – ``FLOAT(B, H_kv, d_k, d_v)``
    * ``decay``      (optional, gated only) – ``FLOAT(B, T, H_kv * d_k)`` or
      ``FLOAT(B, T, H_kv)``

    Update rules (where ⊗ denotes outer product):

    * ``'linear'``:
      ``S_t = S_{t-1} + k_t ⊗ v_t``
    * ``'gated'``:
      ``S_t = exp(g_t) * S_{t-1} + k_t ⊗ v_t``

    followed in all cases by:
    ``o_t = scale * q_t^T S_t``

    The pattern operates on the 4-D internal representation obtained after
    unpacking and transposing the 3-D packed inputs:
    ``[B, T, H * d]  →  Reshape → Transpose  →  [B, H, T, d]``

    For the decoding case (``T = 1``) the sequence/time dimension is squeezed
    before the core computation and unsqueezed afterwards.

    Model with nodes to be fused (``'linear'`` rule, ``T = 1``):

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_query_3d(["query FLOAT(B, 1, H_q*d_k)"])
            I_key_3d(["key FLOAT(B, 1, H_kv*d_k)"])
            I_value_3d(["value FLOAT(B, 1, H_kv*d_v)"])
            I_past_state(["past_state FLOAT(B, H_kv, d_k, d_v)"])

            Reshape_q[["Reshape(., [0, 1, H_q, d_k])"]]
            Reshape_k[["Reshape(., [0, 1, H_kv, d_k])"]]
            Reshape_v[["Reshape(., [0, 1, H_kv, d_v])"]]
            Transpose_q[["Transpose(., perm=[0,2,1,3])"]]
            Transpose_k[["Transpose(., perm=[0,2,1,3])"]]
            Transpose_v[["Transpose(., perm=[0,2,1,3])"]]
            Squeeze_q[["Squeeze(., [2])"]]
            Squeeze_k[["Squeeze(., [2])"]]
            Squeeze_v[["Squeeze(., [2])"]]
            Unsqueeze_k[["Unsqueeze(., [-1])"]]
            Unsqueeze_v[["Unsqueeze(., [-2])"]]
            Mul_kv[["Mul(., .)"]]
            Add_state[["Add(past_state, kv)"]]
            Unsqueeze_q[["Unsqueeze(., [-2])"]]
            MatMul_out[["MatMul(., .)"]]
            Squeeze_out[["Squeeze(., [-2])"]]
            Mul_scale[["Mul(., scale)"]]
            Unsqueeze_out[["Unsqueeze(., [2])"]]
            Transpose_out[["Transpose(., perm=[0,2,1,3])"]]
            Reshape_out[["Reshape(., [0, -1, H_q*d_v])"]]

            I_query_3d --> Reshape_q --> Transpose_q --> Squeeze_q
            I_key_3d --> Reshape_k --> Transpose_k --> Squeeze_k
            I_value_3d --> Reshape_v --> Transpose_v --> Squeeze_v
            Squeeze_k --> Unsqueeze_k
            Squeeze_v --> Unsqueeze_v
            Unsqueeze_k --> Mul_kv
            Unsqueeze_v --> Mul_kv
            I_past_state --> Add_state
            Mul_kv --> Add_state
            Squeeze_q --> Unsqueeze_q
            Unsqueeze_q --> MatMul_out
            Add_state --> MatMul_out
            MatMul_out --> Squeeze_out --> Mul_scale
            Mul_scale --> Unsqueeze_out --> Transpose_out --> Reshape_out

            O_output(["output FLOAT(B, 1, H_q*d_v)"])
            Reshape_out --> O_output
            O_state(["present_state FLOAT(B, H_kv, d_k, d_v)"])
            Add_state --> O_state

            class I_query_3d,I_key_3d,I_value_3d,I_past_state,O_output,O_state ioNode
            class Reshape_q,Reshape_k,Reshape_v,Transpose_q,Transpose_k,Transpose_v opNode
            class Squeeze_q,Squeeze_k,Squeeze_v,Unsqueeze_k,Unsqueeze_v opNode
            class Mul_kv,Add_state,Unsqueeze_q,MatMul_out,Squeeze_out,Mul_scale opNode
            class Unsqueeze_out,Transpose_out,Reshape_out opNode

    Outcome of the fusion:

    .. mermaid::

        graph TD

            classDef ioNode fill:#dfd,stroke:#333,color:#333
            classDef initNode fill:#cccc00,stroke:#333,color:#333
            classDef constNode fill:#f9f,stroke:#333,stroke-width:2px,color:#333
            classDef opNode fill:#bbf,stroke:#333,stroke-width:2px,color:#333

            I_query_3d(["query FLOAT(B, 1, H_q*d_k)"])
            I_key_3d(["key FLOAT(B, 1, H_kv*d_k)"])
            I_value_3d(["value FLOAT(B, 1, H_kv*d_v)"])
            I_past_state(["past_state FLOAT(B, H_kv, d_k, d_v)"])

            LinearAttention_0[["com.microsoft.LinearAttention(., ., ., .)"]]

            I_query_3d --> LinearAttention_0
            I_key_3d --> LinearAttention_0
            I_value_3d --> LinearAttention_0
            I_past_state --> LinearAttention_0

            O_output(["output FLOAT(B, 1, H_q*d_v)"])
            LinearAttention_0 --> O_output
            O_state(["present_state FLOAT(B, H_kv, d_k, d_v)"])
            LinearAttention_0 --> O_state

            class I_query_3d,I_key_3d,I_value_3d,I_past_state,O_output,O_state ioNode
            class LinearAttention_0 opNode
    """

    def __init__(self, verbose: int = 0, priority: int = 2):
        """Initializes the pattern with the given verbosity and matching priority."""
        super().__init__(verbose, priority)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_unsqueeze_axes(
        self, g: "GraphBuilderPatternOptimization", node: NodeProto  # noqa: F821
    ) -> Optional[Tuple[int, ...]]:
        """Returns the axes of an Unsqueeze node as a tuple, or ``None``."""
        axes = g.get_constant_or_attribute(node, "axes", input_index=1, cvt=tuple)
        if axes is None:
            return None
        return tuple(int(a) for a in axes)

    def _get_squeeze_axes(
        self, g: "GraphBuilderPatternOptimization", node: NodeProto  # noqa: F821
    ) -> Optional[Tuple[int, ...]]:
        """Returns the axes of a Squeeze node as a tuple, or ``None``."""
        if len(node.input) < 2:
            return None
        axes = g.get_constant_or_attribute(node, "axes", input_index=1, cvt=tuple)
        if axes is None:
            return None
        return tuple(int(a) for a in axes)

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

    def _match_unpack_3d(
        self, g: "GraphBuilderPatternOptimization", sq_out_name: str  # noqa: F821
    ) -> Optional[Tuple[NodeProto, NodeProto, NodeProto, str]]:
        """Attempts to match a 3-D-unpacking path ending at *sq_out_name*.

        The expected path is:
        ``Squeeze([2]) ← Transpose([0,2,1,3]) ← Reshape ← packed_3d``

        Returns ``(reshape, transpose, squeeze, src_3d_name)`` or ``None``.
        """
        squeeze = g.node_before(sq_out_name)
        if squeeze is None or squeeze.op_type != "Squeeze" or squeeze.domain != "":
            return None
        sq_axes = self._get_squeeze_axes(g, squeeze)
        if sq_axes not in ((2,), (-2,)):
            return None

        transpose = g.node_before(squeeze.input[0])
        if transpose is None or transpose.op_type != "Transpose" or transpose.domain != "":
            return None
        perm = self._get_transpose_perm(g, transpose)
        if perm != (0, 2, 1, 3):
            return None

        reshape = g.node_before(transpose.input[0])
        if reshape is None or reshape.op_type != "Reshape" or reshape.domain != "":
            return None
        if not g.is_constant(reshape.input[1]):
            return None
        return reshape, transpose, squeeze, reshape.input[0]

    def _match_repack_3d(
        self, g: "GraphBuilderPatternOptimization", scaled_out_name: str  # noqa: F821
    ) -> Optional[Tuple[NodeProto, NodeProto, NodeProto]]:
        """Attempts to match the 3-D-repacking path starting from *scaled_out_name*.

        The expected path is:
        ``Unsqueeze([2]) → Transpose([0,2,1,3]) → Reshape``

        Returns ``(unsqueeze, transpose, reshape)`` or ``None``.
        """
        consumers = g.next_nodes(scaled_out_name)
        unsqueeze = next(
            (n for n in consumers if n.op_type == "Unsqueeze" and n.domain == ""), None
        )
        if unsqueeze is None:
            return None
        us_axes = self._get_unsqueeze_axes(g, unsqueeze)
        if us_axes not in ((2,), (-2,)):
            return None

        tr_consumers = g.next_nodes(unsqueeze.output[0])
        transpose = next(
            (n for n in tr_consumers if n.op_type == "Transpose" and n.domain == ""), None
        )
        if transpose is None:
            return None
        perm = self._get_transpose_perm(g, transpose)
        if perm != (0, 2, 1, 3):
            return None

        re_consumers = g.next_nodes(transpose.output[0])
        reshape = next(
            (n for n in re_consumers if n.op_type == "Reshape" and n.domain == ""), None
        )
        if reshape is None:
            return None
        if not g.is_constant(reshape.input[1]):
            return None
        return unsqueeze, transpose, reshape

    def _match_outer_product(
        self, g: "GraphBuilderPatternOptimization", kv_node: NodeProto  # noqa: F821
    ) -> Optional[Tuple[NodeProto, NodeProto]]:
        """Attempts to match a k ⊗ v outer-product node.

        Accepted forms: ``Mul(Unsqueeze(k, [-1]), Unsqueeze(v, [-2]))``
        or the equivalent ``MatMul``.

        Returns ``(k_unsqueeze, v_unsqueeze)`` or ``None``.
        """
        if kv_node.op_type not in ("Mul", "MatMul") or kv_node.domain != "":
            return None
        for k_idx, v_idx in ((0, 1), (1, 0)):
            k_uns = g.node_before(kv_node.input[k_idx])
            v_uns = g.node_before(kv_node.input[v_idx])
            if k_uns is None or v_uns is None:
                continue
            if k_uns.op_type != "Unsqueeze" or k_uns.domain != "":
                continue
            if v_uns.op_type != "Unsqueeze" or v_uns.domain != "":
                continue
            k_axes = self._get_unsqueeze_axes(g, k_uns)
            v_axes = self._get_unsqueeze_axes(g, v_uns)
            # k gets column dim: [..., d_k] → [..., d_k, 1]
            # v gets row dim:    [..., d_v] → [..., 1, d_v]
            if k_axes in ((-1,), (3,)) and v_axes in ((-2,), (2,)):
                return k_uns, v_uns
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
        """Attempts to match starting from a scalar-scale Mul anchor node."""
        # Anchor: Mul with exactly one constant scalar input.
        if node.op_type != "Mul" or node.domain != "":
            return self.none()

        scale_input: Optional[str] = None
        mm_out_squeezed: Optional[str] = None
        for inp in node.input:
            if g.is_constant_scalar(inp):
                scale_input = inp
            else:
                mm_out_squeezed = inp
        if scale_input is None or mm_out_squeezed is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # ---- Squeeze([-2]) of the MatMul output ---------------------------
        sq_out_node = g.node_before(mm_out_squeezed)
        if sq_out_node is None or sq_out_node.op_type != "Squeeze" or sq_out_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        sq_out_axes = self._get_squeeze_axes(g, sq_out_node)
        if sq_out_axes not in ((-2,), (2,)):
            return self.none(node, inspect.currentframe().f_lineno)

        # ---- MatMul(q_us, state) ------------------------------------------
        mm_node = g.node_before(sq_out_node.input[0])
        if mm_node is None or mm_node.op_type != "MatMul" or mm_node.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)

        # ---- q_unsqueeze: Unsqueeze(q_sq, [-2]) ---------------------------
        q_uns = g.node_before(mm_node.input[0])
        if q_uns is None or q_uns.op_type != "Unsqueeze" or q_uns.domain != "":
            return self.none(node, inspect.currentframe().f_lineno)
        q_us_axes = self._get_unsqueeze_axes(g, q_uns)
        if q_us_axes not in ((-2,), (2,)):
            return self.none(node, inspect.currentframe().f_lineno)

        # ---- state node ---------------------------------------------------
        state_node = g.node_before(mm_node.input[1])
        if state_node is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # Try "linear" rule: state = Add(past_state, kv_update)
        update_rule = None
        add_state = None
        kv_mul = None
        k_uns = None
        v_uns = None
        decay_exp_node = None
        state_decay_mul = None
        past_state_name = None
        decay_name: Optional[str] = None
        extra_decay_unsqueeze: Optional[NodeProto] = None

        # Try "gated" rule first (more specific):
        # state = Add(Mul(Exp(decay), past_state), kv_update) — direct
        # or Add(Mul(Unsqueeze(Exp(decay), -1), past_state), kv_update) — with broadcast
        if state_node.op_type == "Add" and state_node.domain == "":
            for gated_idx in (0, 1):
                gated_mul_cand = g.node_before(state_node.input[gated_idx])
                if gated_mul_cand is None or gated_mul_cand.op_type != "Mul":
                    continue
                # One input must be Exp(something) or Unsqueeze(Exp(something), [-1])
                for exp_idx in (0, 1):
                    decay_exp_cand_node = g.node_before(gated_mul_cand.input[exp_idx])
                    if decay_exp_cand_node is None:
                        continue
                    # Direct: Exp(decay)
                    if decay_exp_cand_node.op_type == "Exp" and decay_exp_cand_node.domain == "":
                        decay_src_name = decay_exp_cand_node.input[0]
                        exp_node_for_decay = decay_exp_cand_node
                        us_exp_node = None
                    # Broadcast: Unsqueeze(Exp(decay), [-1])
                    elif (
                        decay_exp_cand_node.op_type == "Unsqueeze"
                        and decay_exp_cand_node.domain == ""
                    ):
                        us_axes = self._get_unsqueeze_axes(g, decay_exp_cand_node)
                        if us_axes not in ((-1,), (3,)):
                            continue
                        inner_exp = g.node_before(decay_exp_cand_node.input[0])
                        if (
                            inner_exp is None
                            or inner_exp.op_type != "Exp"
                            or inner_exp.domain != ""
                        ):
                            continue
                        decay_src_name = inner_exp.input[0]
                        exp_node_for_decay = inner_exp
                        us_exp_node = decay_exp_cand_node
                    else:
                        continue
                    # The other input is past_state
                    ps_gated_idx = 1 - exp_idx
                    ps_name = gated_mul_cand.input[ps_gated_idx]
                    # The other Add input should be kv_update
                    kv_add_idx = 1 - gated_idx
                    kv_node = g.node_before(state_node.input[kv_add_idx])
                    if kv_node is None:
                        continue
                    result = self._match_outer_product(g, kv_node)
                    if result is None:
                        continue
                    add_state = state_node
                    kv_mul = kv_node
                    k_uns, v_uns = result
                    past_state_name = ps_name
                    decay_exp_node = exp_node_for_decay
                    state_decay_mul = gated_mul_cand
                    decay_name = decay_src_name
                    update_rule = "gated"
                    extra_decay_unsqueeze = us_exp_node
                    break
                if update_rule is not None:
                    break

        # Try "linear" rule: state = Add(past_state, kv_update)
        if update_rule is None and state_node.op_type == "Add" and state_node.domain == "":
            for kv_idx in (0, 1):
                kv_node = g.node_before(state_node.input[kv_idx])
                if kv_node is None:
                    continue
                result = self._match_outer_product(g, kv_node)
                if result is not None:
                    add_state = state_node
                    kv_mul = kv_node
                    k_uns, v_uns = result
                    # The other Add input is past_state
                    ps_idx = 1 - kv_idx
                    past_state_name = state_node.input[ps_idx]
                    update_rule = "linear"
                    break

        if update_rule is None:
            return self.none(node, inspect.currentframe().f_lineno)

        # ---- unpack Q, K, V from 3D packed --------------------------------
        q_sq_name = q_uns.input[0]
        k_sq_name = k_uns.input[0]
        v_sq_name = v_uns.input[0]

        q_unpack = self._match_unpack_3d(g, q_sq_name)
        k_unpack = self._match_unpack_3d(g, k_sq_name)
        v_unpack = self._match_unpack_3d(g, v_sq_name)
        if q_unpack is None or k_unpack is None or v_unpack is None:
            return self.none(node, inspect.currentframe().f_lineno)

        rq, trq, sqq, query_3d = q_unpack
        rk, trk, sqk, key_3d = k_unpack
        rv, trv, sqv, value_3d = v_unpack

        # ---- unpack decay from 3D packed (gated rule only) ----------------
        rd: Optional[NodeProto] = None
        trd: Optional[NodeProto] = None
        sqd: Optional[NodeProto] = None
        if update_rule == "gated" and decay_name is not None:
            d_unpack = self._match_unpack_3d(g, decay_name)
            if d_unpack is not None:
                rd, trd, sqd, decay_name = d_unpack

        # ---- infer num_heads from reshape shapes --------------------------
        q_num_heads = 1
        kv_num_heads = 1
        if g.is_constant(rq.input[1]):
            cst = g.get_computed_constant(rq.input[1])
            if cst is not None and cst.ndim == 1 and len(cst) == 4:
                q_num_heads = int(cst[2])
        if g.is_constant(rk.input[1]):
            cst = g.get_computed_constant(rk.input[1])
            if cst is not None and cst.ndim == 1 and len(cst) == 4:
                kv_num_heads = int(cst[2])

        # ---- repack output path: Mul → Unsqueeze → Transpose → Reshape ---
        repack = self._match_repack_3d(g, node.output[0])
        if repack is None:
            return self.none(node, inspect.currentframe().f_lineno)
        us_out, tr_out, re_out = repack

        # ---- uniqueness checks -------------------------------------------
        check_single = [sqq, sqk, sqv, k_uns, v_uns, kv_mul, q_uns, mm_node, sq_out_node]
        for cn in check_single:
            if g.is_used_more_than_once(cn.output[0]):
                return self.none(node, inspect.currentframe().f_lineno)
        # add_state may be consumed by matmul AND as state output.

        # ---- capture derived info in a closure ---------------------------
        _query_3d = query_3d
        _key_3d = key_3d
        _value_3d = value_3d
        _past_state_name = past_state_name
        _output_3d = re_out.output[0]
        _state_output = add_state.output[0]
        _update_rule = update_rule
        _scale_input = scale_input
        _q_num_heads = q_num_heads
        _kv_num_heads = kv_num_heads
        _decay_name = decay_name
        _add_state_name = add_state.name

        def _apply(g_builder, *_nodes):
            return self._do_apply(
                g_builder,
                query_3d=_query_3d,
                key_3d=_key_3d,
                value_3d=_value_3d,
                past_state_name=_past_state_name,
                output_3d=_output_3d,
                state_output=_state_output,
                update_rule=_update_rule,
                scale_input=_scale_input,
                q_num_heads=_q_num_heads,
                kv_num_heads=_kv_num_heads,
                decay_name=_decay_name,
                add_state_name=_add_state_name,
            )

        # ---- build node list (order matches apply signature) -------------
        all_nodes: List[Optional[NodeProto]] = [
            rq,
            trq,
            sqq,  # unpack q
            rk,
            trk,
            sqk,  # unpack k
            rv,
            trv,
            sqv,  # unpack v
            rd,
            trd,
            sqd,  # unpack decay (may be None)
            k_uns,
            v_uns,
            kv_mul,  # outer product
            decay_exp_node,
            extra_decay_unsqueeze,  # optional Exp + Unsqueeze for decay
            state_decay_mul,  # optional state Mul (gated)
            add_state,  # state update
            q_uns,
            mm_node,
            sq_out_node,
            node,  # output computation
            us_out,
            tr_out,
            re_out,  # repack
        ]

        return MatchResult(
            self, [n for n in all_nodes if n is not None], _apply, insert_at=re_out
        )

    # ------------------------------------------------------------------
    # _do_apply
    # ------------------------------------------------------------------

    def _do_apply(
        self,
        g: "GraphBuilder",  # noqa: F821
        query_3d: str,
        key_3d: str,
        value_3d: str,
        past_state_name: str,
        output_3d: str,
        state_output: str,
        update_rule: str,
        scale_input: str,
        q_num_heads: int,
        kv_num_heads: int,
        decay_name: Optional[str],
        add_state_name: str,
    ) -> List[NodeProto]:
        """Creates and returns the ``com.microsoft.LinearAttention`` node."""
        scale_val = float(g.get_constant_scalar(scale_input)) if scale_input else None

        # Inputs: [query, key, value, past_state (opt), decay (opt)]
        inputs = [query_3d, key_3d, value_3d]
        if past_state_name:
            inputs.append(past_state_name)
        else:
            inputs.append("")
        if decay_name is not None:
            inputs.append(decay_name)

        # Outputs: [output, present_state]
        outputs = [output_3d]
        if state_output:
            outputs.append(state_output)

        kwargs = {
            "domain": "com.microsoft",
            "q_num_heads": q_num_heads,
            "kv_num_heads": kv_num_heads,
            "update_rule": update_rule,
            "name": f"{self.__class__.__name__}--{add_state_name}",
        }
        if scale_val is not None:
            kwargs["scale"] = scale_val

        la_node = g.make_node("LinearAttention", inputs, outputs, **kwargs)
        return [la_node]

    # apply is not used directly; the closure _apply defined in match() is used.
    def apply(self, g: "GraphBuilder", *nodes: NodeProto) -> List[NodeProto]:  # noqa: F821
        """Not called directly; the match closure handles dispatch."""
        raise NotImplementedError("LinearAttentionPattern.apply should not be called directly")
