"""
Direct ONNX converter for :class:`transformers.models.llama.modeling_llama.LlamaAttention`.

This converter builds a complete ONNX graph from a fitted
:class:`transformers.models.llama.modeling_llama.LlamaAttention` module without
going through :func:`torch.export.export`.

Three computation backends are supported, selected automatically based on
the requested target opset:

* **com.microsoft** (``"com.microsoft" in target_opset``):
  Uses the ``com.microsoft.MultiHeadAttention`` contrib op from *onnxruntime*.
  GQA key/value heads are expanded (via repeat-interleave) to match the query
  head count before being passed to the op.
  The model runs efficiently on CPU and CUDA with OnnxRuntime.
* **opset ≥ 24** (``main_opset >= 24``):
  Uses the standard ONNX ``Attention`` operator introduced in opset 23
  (revision 24 fixes a correctness bug; this converter therefore requires 24).
* **opset ≤ 22** (default fallback):
  Uses basic ONNX ops (``MatMul``, ``Softmax``, ``Transpose``, …).

Inputs to the produced ONNX model:

* ``hidden_states``: ``(batch, seq, hidden_size)``
* ``cos``: ``(batch, seq, head_dim)``
* ``sin``: ``(batch, seq, head_dim)``
* ``attention_mask`` *(optional)*: ``(batch, 1, seq_q, total_seq)``

Output:

* ``attn_output``: ``(batch, seq, hidden_size)``
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import onnx
from onnx import ModelProto, TensorProto

from ....helpers.onnx_helper import np_dtype_to_tensor_dtype, tensor_dtype_to_np_dtype
from ....xbuilder import GraphBuilder

T = str

# Upper bound for open-ended ONNX Slice (INT64_MAX would work but 2^30 is safe and smaller)
_MAX_SLICE_END = 2**30


def _np_dtype_for_onnx(onnx_dtype: int) -> np.dtype:
    return tensor_dtype_to_np_dtype(onnx_dtype)


def _rotate_half(g: GraphBuilder, x: T, head_dim: int, name: str) -> T:
    """
    Implements ``rotate_half(x)``:
    split the last dimension into two halves, negate the second, then concatenate::

        x1 = x[..., :head_dim // 2]
        x2 = x[..., head_dim // 2 :]
        return concat(-x2, x1, dim=-1)
    """
    half = head_dim // 2
    x1 = g.op.Slice(
        x,
        np.array([0], dtype=np.int64),
        np.array([half], dtype=np.int64),
        np.array([-1], dtype=np.int64),
        name=name,
    )
    x2 = g.op.Slice(
        x,
        np.array([half], dtype=np.int64),
        np.array([_MAX_SLICE_END], dtype=np.int64),
        np.array([-1], dtype=np.int64),
        name=name,
    )
    neg_x2 = g.op.Neg(x2, name=name)
    return g.op.Concat(neg_x2, x1, axis=-1, name=name)


def _apply_rope(
    g: GraphBuilder,
    states_4d: T,
    cos_4d: T,
    sin_4d: T,
    head_dim: int,
    name: str,
) -> T:
    """
    Applies Rotary Position Embedding to a 4-D tensor
    ``(batch, n_heads, seq, head_dim)`` given pre-computed ``cos`` and
    ``sin`` tensors broadcast to the same shape.

    ``cos_4d`` and ``sin_4d`` are expected to be
    ``(batch, 1, seq, head_dim)`` (already unsqueezed).
    """
    rotated = _rotate_half(g, states_4d, head_dim, name=name)
    cos_mul = g.op.Mul(states_4d, cos_4d, name=name)
    sin_mul = g.op.Mul(rotated, sin_4d, name=name)
    return g.op.Add(cos_mul, sin_mul, name=name)


def _repeat_kv(
    g: GraphBuilder,
    kv: T,
    n_rep: int,
    num_kv_heads: int,
    head_dim: int,
    name: str,
) -> T:
    """
    Repeats key / value heads ``n_rep`` times along the heads dimension
    (dimension 1) to expand a GQA tensor from
    ``(batch, num_kv_heads, seq, head_dim)`` to
    ``(batch, num_kv_heads * n_rep, seq, head_dim)``.

    Mirrors ``transformers.models.llama.modeling_llama.repeat_kv``.
    """
    if n_rep == 1:
        return kv
    # (batch, num_kv_heads, seq, head_dim)
    # -> (batch, num_kv_heads, 1, seq, head_dim)
    unsqueezed = g.op.Unsqueeze(kv, np.array([2], dtype=np.int64), name=name)

    # Tile along dim 2 -> (batch, num_kv_heads, n_rep, seq, head_dim)
    tiled = g.op.Tile(unsqueezed, np.array([1, 1, n_rep, 1, 1], dtype=np.int64), name=name)

    # Reshape to (batch, num_kv_heads * n_rep, seq, head_dim).
    # Use Shape-based dynamic reshape because batch and seq are dynamic.
    kv_shape = g.op.Shape(kv, name=name)
    batch_val = g.op.Slice(
        kv_shape,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        name=name,
    )
    seq_val = g.op.Slice(
        kv_shape,
        np.array([2], dtype=np.int64),
        np.array([3], dtype=np.int64),
        name=name,
    )
    new_shape = g.op.Concat(
        batch_val,
        np.array([num_kv_heads * n_rep], dtype=np.int64),
        seq_val,
        np.array([head_dim], dtype=np.int64),
        axis=0,
        name=name,
    )
    return g.op.Reshape(tiled, new_shape, name=name)


def _project_and_split(
    g: GraphBuilder,
    hidden_states: T,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    num_heads: int,
    head_dim: int,
    name: str,
    weight_name: str,
) -> T:
    """
    Applies a linear projection followed by a reshape and transpose
    to produce a 4-D ``(batch, n_heads, seq, head_dim)`` tensor.
    """
    w = g.make_initializer(weight_name, weight.T)
    projected = g.op.MatMul(hidden_states, w, name=name)
    if bias is not None:
        b = g.make_initializer(f"{weight_name}_bias", bias)
        projected = g.op.Add(projected, b, name=name)

    # (batch, seq, n_heads * head_dim) -> (batch, seq, n_heads, head_dim)
    split_shape = np.array([0, 0, num_heads, head_dim], dtype=np.int64)
    reshaped = g.op.Reshape(projected, split_shape, name=name)

    # (batch, seq, n_heads, head_dim) -> (batch, n_heads, seq, head_dim)
    transposed = g.op.Transpose(reshaped, perm=[0, 2, 1, 3], name=name)
    return transposed


def _standard_attention(
    g: GraphBuilder,
    query_4d: T,
    key_4d: T,
    value_4d: T,
    attention_mask: Optional[T],
    scaling: float,
    dtype: int,
    name: str,
) -> T:
    """
    Computes scaled dot-product attention using basic ONNX ops::

        attn_weights = Q @ K^T * scaling
        if mask: attn_weights += mask
        attn_weights = softmax(attn_weights.float(), dim=-1).to(dtype)
        output = attn_weights @ V

    Returns ``(batch, n_heads, seq, head_dim)``.
    """
    np_dtype = _np_dtype_for_onnx(dtype)
    # K^T: (batch, n_heads, head_dim, seq)
    key_t = g.op.Transpose(key_4d, perm=[0, 1, 3, 2], name=name)
    # attn_weights: (batch, n_heads, seq_q, seq_kv)
    raw = g.op.MatMul(query_4d, key_t, name=name)
    scaled = g.op.Mul(raw, np.array([scaling], dtype=np_dtype), name=name)

    if attention_mask is not None:
        scaled = g.op.Add(scaled, attention_mask, name=name)

    # softmax in float32 for numerical stability, then cast back
    if dtype != TensorProto.FLOAT:
        scaled_f32 = g.op.Cast(scaled, to=TensorProto.FLOAT, name=name)
        soft_f32 = g.op.Softmax(scaled_f32, axis=-1, name=name)
        attn_weights = g.op.Cast(soft_f32, to=dtype, name=name)
    else:
        attn_weights = g.op.Softmax(scaled, axis=-1, name=name)

    return g.op.MatMul(attn_weights, value_4d, name=name)


def _attention_opset24(
    g: GraphBuilder,
    query_4d: T,
    key_4d: T,
    value_4d: T,
    attention_mask: Optional[T],
    scaling: float,
    name: str,
) -> T:
    """
    Uses the ONNX ``Attention`` operator (opset ≥ 24).
    Returns ``(batch, n_heads, seq, head_dim)``.
    """
    return g.op.Attention(
        query_4d,
        key_4d,
        value_4d,
        attention_mask,
        scale=scaling,
        name=name,
    )


def _mha_com_microsoft(
    g: GraphBuilder,
    query_3d: T,
    key_3d: T,
    value_3d: T,
    attention_mask: Optional[T],
    num_heads: int,
    scaling: float,
    name: str,
) -> T:
    """
    Uses ``com.microsoft.MultiHeadAttention``.

    Inputs are 3-D tensors ``(batch, seq, n_heads * head_dim)``.
    Query, key and value must all have the same last dimension, so GQA
    key/value heads must be repeated before calling this function.

    Returns the 3-D output ``(batch, seq, n_heads * head_dim)``.
    """
    inputs: list = [query_3d, key_3d, value_3d, "", ""]
    if attention_mask is not None:
        inputs.append(attention_mask)

    out = g.op.MultiHeadAttention(
        *inputs,
        domain="com.microsoft",
        num_heads=num_heads,
        scale=scaling,
        name=name,
    )
    # MultiHeadAttention returns (output [, present_key, present_value]); keep only output
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out


def llama_attention_to_onnx(
    attn: "LlamaAttention",  # noqa: F821
    args: Tuple[Any, ...],
    target_opset: Union[int, Dict[str, int]] = 22,
    with_mask: bool = False,
    input_names: Optional[Tuple[str, ...]] = None,
) -> ModelProto:
    """
    Converts a :class:`transformers.models.llama.modeling_llama.LlamaAttention`
    module into an ONNX model.

    :param attn: a fitted ``LlamaAttention`` module (weights must be initialised)
    :param args: example inputs used only to determine shapes and dtype; a tuple of
        ``(hidden_states, cos, sin)`` or ``(hidden_states, cos, sin, attention_mask)``
    :param target_opset: target opset — either an integer for the default domain
        (``""``), or a dictionary mapping domain names to opset versions,
        e.g. ``{"": 22}`` (standard), ``{"": 22, "com.microsoft": 1}`` (ORT
        contrib ops), or ``{"": 24}`` (ONNX Attention op)
    :param with_mask: if ``True`` add an ``attention_mask`` input even when
        not provided in *args*
    :param input_names: optional names for the ONNX inputs; defaults to
        ``("hidden_states", "cos", "sin")`` / ``("hidden_states", "cos", "sin", "attention_mask")``
    :return: :class:`onnx.ModelProto`
    """
    import torch

    if isinstance(target_opset, int):
        dict_opset: Dict[str, int] = {"": target_opset}
    else:
        dict_opset = dict(target_opset)

    main_opset = dict_opset.get("", 22)
    has_ms = "com.microsoft" in dict_opset

    # ------------------------------------------------------------------ #
    # Inspect module weights                                               #
    # ------------------------------------------------------------------ #
    q_w = attn.q_proj.weight.detach().cpu().numpy()  # (num_heads*head_dim, hidden)
    k_w = attn.k_proj.weight.detach().cpu().numpy()  # (num_kv_heads*head_dim, hidden)
    v_w = attn.v_proj.weight.detach().cpu().numpy()  # (num_kv_heads*head_dim, hidden)
    o_w = attn.o_proj.weight.detach().cpu().numpy()  # (hidden, num_heads*head_dim)

    q_b = attn.q_proj.bias.detach().cpu().numpy() if attn.q_proj.bias is not None else None
    k_b = attn.k_proj.bias.detach().cpu().numpy() if attn.k_proj.bias is not None else None
    v_b = attn.v_proj.bias.detach().cpu().numpy() if attn.v_proj.bias is not None else None
    o_b = attn.o_proj.bias.detach().cpu().numpy() if attn.o_proj.bias is not None else None

    num_heads: int = attn.config.num_attention_heads
    num_kv_heads: int = attn.config.num_key_value_heads
    head_dim: int = attn.head_dim
    n_rep: int = attn.num_key_value_groups
    scaling: float = attn.scaling
    hidden_size: int = attn.config.hidden_size

    # ------------------------------------------------------------------ #
    # Determine ONNX dtype from example inputs                            #
    # ------------------------------------------------------------------ #
    if isinstance(args[0], torch.Tensor):
        onnx_dtype = np_dtype_to_tensor_dtype(args[0].detach().cpu().numpy().dtype)
    elif isinstance(args[0], np.ndarray):
        onnx_dtype = np_dtype_to_tensor_dtype(args[0].dtype)
    else:
        onnx_dtype = TensorProto.FLOAT

    np_dtype = _np_dtype_for_onnx(onnx_dtype)

    # Cast weights to the target dtype
    q_w = q_w.astype(np_dtype)
    k_w = k_w.astype(np_dtype)
    v_w = v_w.astype(np_dtype)
    o_w = o_w.astype(np_dtype)
    if q_b is not None:
        q_b = q_b.astype(np_dtype)
    if k_b is not None:
        k_b = k_b.astype(np_dtype)
    if v_b is not None:
        v_b = v_b.astype(np_dtype)
    if o_b is not None:
        o_b = o_b.astype(np_dtype)

    # ------------------------------------------------------------------ #
    # Derive input shapes from examples                                   #
    # ------------------------------------------------------------------ #
    def _get_shape(x: Any):
        if isinstance(x, torch.Tensor):
            return tuple(x.shape)
        if isinstance(x, np.ndarray):
            return tuple(x.shape)
        return None

    hs_shape = _get_shape(args[0])
    has_mask_input = (len(args) > 3 and args[3] is not None) or with_mask
    mask_shape = _get_shape(args[3]) if len(args) > 3 and args[3] is not None else None

    # Build dynamic shape specs
    dyn_hs = ("batch", "seq", hidden_size) if hs_shape is None else (
        "batch", "seq", hs_shape[2]
    )
    dyn_cos = ("batch", "seq", head_dim)
    dyn_sin = ("batch", "seq", head_dim)
    if has_mask_input:
        dyn_mask = ("batch", 1, "seq_q", "total_seq") if mask_shape is None else (
            "batch", 1, mask_shape[2], mask_shape[3]
        )

    # ------------------------------------------------------------------ #
    # Build graph                                                         #
    # ------------------------------------------------------------------ #
    g = GraphBuilder(dict_opset, verbose=0)

    default_names = ("hidden_states", "cos", "sin", "attention_mask")
    if input_names is None:
        input_names = default_names[: 4 if has_mask_input else 3]

    # Declare inputs
    hs_name = input_names[0]
    cos_name = input_names[1] if len(input_names) > 1 else "cos"
    sin_name = input_names[2] if len(input_names) > 2 else "sin"
    mask_name = input_names[3] if len(input_names) > 3 else "attention_mask"

    g.make_tensor_input(hs_name, onnx_dtype, dyn_hs)
    g.make_tensor_input(cos_name, onnx_dtype, dyn_cos)
    g.make_tensor_input(sin_name, onnx_dtype, dyn_sin)
    if has_mask_input:
        g.make_tensor_input(mask_name, onnx_dtype, dyn_mask)
        mask_input: Optional[T] = mask_name
    else:
        mask_input = None

    name = "llama_attention"

    # -------------------------------------------------------------- #
    # Branch: com.microsoft MultiHeadAttention                        #
    # -------------------------------------------------------------- #
    if has_ms:
        # Project (batch, seq, hidden) -> (batch, seq, heads * head_dim)
        def _proj_3d(x_3d: T, w: np.ndarray, b: Optional[np.ndarray], wname: str) -> T:
            wt = g.make_initializer(wname, w.T)
            res = g.op.MatMul(x_3d, wt, name=name)
            if b is not None:
                bv = g.make_initializer(f"{wname}_bias", b)
                res = g.op.Add(res, bv, name=name)
            return res

        q_3d = _proj_3d(hs_name, q_w, q_b, "w_q")  # (batch, seq, num_heads * head_dim)
        k_3d = _proj_3d(hs_name, k_w, k_b, "w_k")  # (batch, seq, num_kv_heads * head_dim)
        v_3d = _proj_3d(hs_name, v_w, v_b, "w_v")  # (batch, seq, num_kv_heads * head_dim)

        # Apply RoPE: reshape to 4D, apply, reshape back to 3D
        def _apply_rope_3d(x_3d: T, n_h: int, cos_n: T, sin_n: T) -> T:
            # (batch, seq, n_h * head_dim) -> (batch, seq, n_h, head_dim)
            sp = np.array([0, 0, n_h, head_dim], dtype=np.int64)
            x4d = g.op.Reshape(x_3d, sp, name=name)
            # (batch, seq, n_h, head_dim) -> (batch, n_h, seq, head_dim)
            x4d = g.op.Transpose(x4d, perm=[0, 2, 1, 3], name=name)
            # cos/sin: (batch, seq, head_dim) -> (batch, 1, seq, head_dim)
            cos4d = g.op.Unsqueeze(cos_n, np.array([1], dtype=np.int64), name=name)
            sin4d = g.op.Unsqueeze(sin_n, np.array([1], dtype=np.int64), name=name)
            x4d = _apply_rope(g, x4d, cos4d, sin4d, head_dim, name)
            # (batch, n_h, seq, head_dim) -> (batch, seq, n_h, head_dim)
            x4d = g.op.Transpose(x4d, perm=[0, 2, 1, 3], name=name)
            # (batch, seq, n_h, head_dim) -> (batch, seq, n_h * head_dim)
            sp3 = np.array([0, 0, -1], dtype=np.int64)
            return g.op.Reshape(x4d, sp3, name=name)

        q_3d = _apply_rope_3d(q_3d, num_heads, cos_name, sin_name)
        k_3d = _apply_rope_3d(k_3d, num_kv_heads, cos_name, sin_name)

        # Repeat KV for GQA so k/v have same head count as q.
        # Use Unsqueeze + Expand (not Tile) to get the same interleaved ordering
        # as torch.repeat_interleave: [h0, h0, h1, h1] not [h0, h1, h0, h1].
        if n_rep > 1:
            def _repeat_kv_3d(x_3d: T) -> T:
                # (batch, seq, kv * head_dim)
                sp_4d = np.array([0, 0, num_kv_heads, head_dim], dtype=np.int64)
                x4d = g.op.Reshape(x_3d, sp_4d, name=name)
                # Unsqueeze at dim 3 → (batch, seq, kv, 1, head_dim)
                x5d = g.op.Unsqueeze(x4d, np.array([3], dtype=np.int64), name=name)
                # Compute dynamic shape (batch, seq, kv, n_rep, head_dim)
                x_shape = g.op.Shape(x_3d, name=name)
                batch_v = g.op.Slice(
                    x_shape, np.array([0], dtype=np.int64), np.array([1], dtype=np.int64), name=name
                )
                seq_v = g.op.Slice(
                    x_shape, np.array([1], dtype=np.int64), np.array([2], dtype=np.int64), name=name
                )
                tgt = g.op.Concat(
                    batch_v, seq_v,
                    np.array([num_kv_heads, n_rep, head_dim], dtype=np.int64),
                    axis=0, name=name,
                )
                x5d = g.op.Expand(x5d, tgt, name=name)
                # Reshape → (batch, seq, kv * n_rep * head_dim) with interleaved heads
                sp_3d = np.array([0, 0, -1], dtype=np.int64)
                return g.op.Reshape(x5d, sp_3d, name=name)

            k_3d = _repeat_kv_3d(k_3d)
            v_3d = _repeat_kv_3d(v_3d)

        # MHA: output is (batch, seq, num_heads * head_dim)
        out_3d = _mha_com_microsoft(
            g, q_3d, k_3d, v_3d, mask_input,
            num_heads, scaling, name,
        )

        # Apply output projection
        o_wt = g.make_initializer("w_o", o_w.T)
        attn_out = g.op.MatMul(out_3d, o_wt, name=name)
        if o_b is not None:
            o_bv = g.make_initializer("w_o_bias", o_b)
            attn_out = g.op.Add(attn_out, o_bv, name=name)

    # -------------------------------------------------------------- #
    # Branch: ONNX Attention (opset >= 24)                            #
    # -------------------------------------------------------------- #
    elif main_opset >= 24:
        # Project to 4D
        q_4d = _project_and_split(g, hs_name, q_w, q_b, num_heads, head_dim, name, "w_q")
        k_4d = _project_and_split(g, hs_name, k_w, k_b, num_kv_heads, head_dim, name, "w_k")
        v_4d = _project_and_split(g, hs_name, v_w, v_b, num_kv_heads, head_dim, name, "w_v")

        # Apply RoPE
        cos4d = g.op.Unsqueeze(cos_name, np.array([1], dtype=np.int64), name=name)
        sin4d = g.op.Unsqueeze(sin_name, np.array([1], dtype=np.int64), name=name)
        q_4d = _apply_rope(g, q_4d, cos4d, sin4d, head_dim, name)
        k_4d = _apply_rope(g, k_4d, cos4d, sin4d, head_dim, name)

        # Expand KV if needed
        if n_rep > 1:
            k_4d = _repeat_kv(g, k_4d, n_rep, num_kv_heads, head_dim, name)
            v_4d = _repeat_kv(g, v_4d, n_rep, num_kv_heads, head_dim, name)

        # ONNX Attention op
        out_4d = _attention_opset24(g, q_4d, k_4d, v_4d, mask_input, scaling, name)

        # (batch, n_heads, seq, head_dim) -> (batch, seq, hidden_size)
        transposed = g.op.Transpose(out_4d, perm=[0, 2, 1, 3], name=name)
        flat_shape = np.array([0, 0, hidden_size], dtype=np.int64)
        flat = g.op.Reshape(transposed, flat_shape, name=name)

        # Output projection
        o_wt = g.make_initializer("w_o", o_w.T)
        attn_out = g.op.MatMul(flat, o_wt, name=name)
        if o_b is not None:
            o_bv = g.make_initializer("w_o_bias", o_b)
            attn_out = g.op.Add(attn_out, o_bv, name=name)

    # -------------------------------------------------------------- #
    # Branch: Standard ONNX ops (opset ≤ 22)                          #
    # -------------------------------------------------------------- #
    else:
        # Project to 4D
        q_4d = _project_and_split(g, hs_name, q_w, q_b, num_heads, head_dim, name, "w_q")
        k_4d = _project_and_split(g, hs_name, k_w, k_b, num_kv_heads, head_dim, name, "w_k")
        v_4d = _project_and_split(g, hs_name, v_w, v_b, num_kv_heads, head_dim, name, "w_v")

        # Apply RoPE
        cos4d = g.op.Unsqueeze(cos_name, np.array([1], dtype=np.int64), name=name)
        sin4d = g.op.Unsqueeze(sin_name, np.array([1], dtype=np.int64), name=name)
        q_4d = _apply_rope(g, q_4d, cos4d, sin4d, head_dim, name)
        k_4d = _apply_rope(g, k_4d, cos4d, sin4d, head_dim, name)

        # Expand KV if needed
        if n_rep > 1:
            k_4d = _repeat_kv(g, k_4d, n_rep, num_kv_heads, head_dim, name)
            v_4d = _repeat_kv(g, v_4d, n_rep, num_kv_heads, head_dim, name)

        # Standard attention
        out_4d = _standard_attention(
            g, q_4d, k_4d, v_4d, mask_input, scaling, onnx_dtype, name
        )

        # (batch, n_heads, seq, head_dim) -> (batch, seq, hidden_size)
        transposed = g.op.Transpose(out_4d, perm=[0, 2, 1, 3], name=name)
        flat_shape = np.array([0, 0, hidden_size], dtype=np.int64)
        flat = g.op.Reshape(transposed, flat_shape, name=name)

        # Output projection
        o_wt = g.make_initializer("w_o", o_w.T)
        attn_out = g.op.MatMul(flat, o_wt, name=name)
        if o_b is not None:
            o_bv = g.make_initializer("w_o_bias", o_b)
            attn_out = g.op.Add(attn_out, o_bv, name=name)

    # Declare output
    g.make_tensor_output(attn_out, onnx_dtype, ("batch", "seq", hidden_size))

    return g.to_onnx(optimize=False)
