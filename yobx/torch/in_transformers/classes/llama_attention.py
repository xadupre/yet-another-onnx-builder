"""
Direct ONNX converter for :class:`transformers.models.llama.modeling_llama.LlamaAttention`.

This converter appends ONNX nodes to an existing :class:`~yobx.xbuilder.GraphBuilder`
from a fitted :class:`transformers.models.llama.modeling_llama.LlamaAttention` module,
without going through :func:`torch.export.export`.

Three computation backends are supported, selected automatically based on
the opsets registered in the graph builder:

* **com.microsoft** (``"com.microsoft"`` domain registered in the builder):
  Uses the ``com.microsoft.RotaryEmbedding`` contrib op for rotary embeddings
  and the ``com.microsoft.MultiHeadAttention`` contrib op for attention from
  *onnxruntime*.
  GQA key/value heads are expanded (via repeat-interleave) to match the query
  head count before the attention op.
  The model runs efficiently on CPU and CUDA with OnnxRuntime.
* **opset ≥ 24** (main opset ≥ 24):
  Uses the standard ONNX ``RotaryEmbedding`` operator (opset ≥ 23) for
  rotary embeddings and the standard ONNX ``Attention`` operator introduced
  in opset 23 (revision 24 fixes a correctness bug; this converter therefore
  requires 24) for attention.
* **opset 23** (main opset == 23):
  Uses the standard ONNX ``RotaryEmbedding`` operator for rotary embeddings
  and basic ONNX ops (``MatMul``, ``Softmax``, ``Transpose``, …) for attention.
* **opset ≤ 22** (default fallback):
  Uses basic ONNX ops (``MatMul``, ``Softmax``, ``Transpose``, …) for both
  rotary embeddings and attention.

Supported dtypes:
    ``float32``, ``float16``, and ``bfloat16`` are all supported.
    The output dtype is inferred from the registered type of *hidden_states*
    in the graph builder.

Expected graph inputs (must be declared in the builder before calling the converter):

* ``hidden_states``: ``(batch, seq, hidden_size)``
* ``cos``: ``(batch, seq, head_dim)``
* ``sin``: ``(batch, seq, head_dim)``
* ``attention_mask`` *(optional)*: ``(batch, 1, seq_q, total_seq)``

Output returned by the converter:

* tensor name ``(batch, seq, hidden_size)`` — the caller must register it as
  a graph output via ``g.make_tensor_output``

Example::

    import torch
    from onnx import TensorProto
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaAttention
    from yobx.xbuilder import GraphBuilder
    from yobx.torch.in_transformers.models import llama_attention_to_onnx

    config = LlamaConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16
    )
    attn = LlamaAttention(config, layer_idx=0).eval()

    # opset 22 — plain ONNX ops (MatMul, Softmax, …)
    g = GraphBuilder({"": 22}, verbose=0)
    g.make_tensor_input("hidden_states", TensorProto.FLOAT, ("batch", "seq", 64))
    g.make_tensor_input("cos", TensorProto.FLOAT, ("batch", "seq", 16))
    g.make_tensor_input("sin", TensorProto.FLOAT, ("batch", "seq", 16))
    out = llama_attention_to_onnx(g, attn, "hidden_states", "cos", "sin")
    g.make_tensor_output(out, TensorProto.FLOAT, ("batch", "seq", 64))
    model = g.to_onnx()

    # opset 24 — ONNX Attention op
    g = GraphBuilder({"": 24}, verbose=0)
    g.make_tensor_input("hidden_states", TensorProto.FLOAT, ("batch", "seq", 64))
    g.make_tensor_input("cos", TensorProto.FLOAT, ("batch", "seq", 16))
    g.make_tensor_input("sin", TensorProto.FLOAT, ("batch", "seq", 16))
    out = llama_attention_to_onnx(g, attn, "hidden_states", "cos", "sin")
    g.make_tensor_output(out, TensorProto.FLOAT, ("batch", "seq", 64))
    model = g.to_onnx()

    # OnnxRuntime contrib ops
    g = GraphBuilder({"": 22, "com.microsoft": 1}, verbose=0)
    g.make_tensor_input("hidden_states", TensorProto.FLOAT, ("batch", "seq", 64))
    g.make_tensor_input("cos", TensorProto.FLOAT, ("batch", "seq", 16))
    g.make_tensor_input("sin", TensorProto.FLOAT, ("batch", "seq", 16))
    out = llama_attention_to_onnx(g, attn, "hidden_states", "cos", "sin")
    g.make_tensor_output(out, TensorProto.FLOAT, ("batch", "seq", 64))
    model = g.to_onnx()
"""

from typing import Optional
import numpy as np
from onnx import TensorProto
from transformers.models.llama.modeling_llama import LlamaAttention

from ....helpers.onnx_helper import tensor_dtype_to_np_dtype
from ....typing import GraphBuilderExtendedProtocol
from ...torch_helper import to_numpy
from ..register import register_transformer_converter

T = str

# Upper bound for open-ended ONNX Slice (INT64_MAX would work but 2^30 is safe and smaller)
_MAX_SLICE_END = 922337203685477580


def _np_dtype_for_onnx(onnx_dtype: int) -> np.dtype:
    return tensor_dtype_to_np_dtype(onnx_dtype)


def _rotate_half(g: GraphBuilderExtendedProtocol, x: T, head_dim: int, name: str) -> T:
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
    g: GraphBuilderExtendedProtocol, states_4d: T, cos_4d: T, sin_4d: T, head_dim: int, name: str
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
    g: GraphBuilderExtendedProtocol,
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
        kv_shape, np.array([0], dtype=np.int64), np.array([1], dtype=np.int64), name=name
    )
    seq_val = g.op.Slice(
        kv_shape, np.array([2], dtype=np.int64), np.array([3], dtype=np.int64), name=name
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
    g: GraphBuilderExtendedProtocol,
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
    g: GraphBuilderExtendedProtocol,
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
    g: GraphBuilderExtendedProtocol,
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
    return g.op.Attention(query_4d, key_4d, value_4d, attention_mask, scale=scaling, name=name)


def _mha_com_microsoft(
    g: GraphBuilderExtendedProtocol,
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
        *inputs, domain="com.microsoft", num_heads=num_heads, scale=scaling, name=name
    )
    # MultiHeadAttention returns (output [, present_key, present_value]); keep only output
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out


def _apply_rope_standard_onnx(
    g: GraphBuilderExtendedProtocol,
    states_4d: T,
    cos: T,
    sin: T,
    head_dim: int,
    num_heads: int,
    name: str,
) -> T:
    """
    Applies Rotary Position Embedding using the standard ONNX
    ``RotaryEmbedding`` operator (opset ≥ 23).

    ``states_4d`` is a 4-D ``(batch, n_heads, seq, head_dim)`` tensor.
    ``cos`` and ``sin`` are 3-D ``(batch, seq, head_dim)`` tensors whose
    first and second halves along the last axis carry the same values
    (LLaMA-style symmetric positional embeddings).  Only the first half
    is passed to the op; the formula is mathematically equivalent to
    the manual ``_apply_rope`` implementation when the symmetry property
    holds.
    """
    half = head_dim // 2
    cos_half = g.op.Slice(
        cos,
        np.array([0], dtype=np.int64),
        np.array([half], dtype=np.int64),
        np.array([-1], dtype=np.int64),
        name=name,
    )
    sin_half = g.op.Slice(
        sin,
        np.array([0], dtype=np.int64),
        np.array([half], dtype=np.int64),
        np.array([-1], dtype=np.int64),
        name=name,
    )
    return g.op.RotaryEmbedding(states_4d, cos_half, sin_half, num_heads=num_heads, name=name)


def _apply_rope_ms_op(
    g: GraphBuilderExtendedProtocol,
    x_3d: T,
    cos: T,
    sin: T,
    head_dim: int,
    num_heads: int,
    name: str,
) -> T:
    """
    Applies Rotary Position Embedding using the
    ``com.microsoft.RotaryEmbedding`` contrib op.

    ``x_3d`` is a 3-D ``(batch, seq, num_heads * head_dim)`` tensor.
    ``cos`` and ``sin`` are 3-D ``(batch, seq, head_dim)`` tensors.
    The first half of the last dimension is extracted; in the LLaMA model
    the two halves are identical (symmetric positional embeddings), so
    taking the first batch element's values as the 2-D cache is safe.

    Position IDs are synthesised as ``[0, 1, …, seq-1]`` broadcast over
    the batch dimension, producing a ``(batch, seq)`` tensor.

    Returns a 3-D ``(batch, seq, num_heads * head_dim)`` tensor.
    """
    half = head_dim // 2

    # Slice cos/sin to (batch, seq, head_dim // 2)
    cos_half = g.op.Slice(
        cos,
        np.array([0], dtype=np.int64),
        np.array([half], dtype=np.int64),
        np.array([-1], dtype=np.int64),
        name=name,
    )
    sin_half = g.op.Slice(
        sin,
        np.array([0], dtype=np.int64),
        np.array([half], dtype=np.int64),
        np.array([-1], dtype=np.int64),
        name=name,
    )

    # Drop the batch dimension by gathering index 0:
    # (batch, seq, head_dim//2) -> (seq, head_dim//2)
    # LLaMA positional embeddings are identical for every batch element,
    # so this is safe regardless of batch size.
    cos_2d = g.op.Gather(cos_half, np.array(0, dtype=np.int64), axis=0, name=name)
    sin_2d = g.op.Gather(sin_half, np.array(0, dtype=np.int64), axis=0, name=name)

    # Build position_ids: (batch, seq) INT64 with values [0, 1, ..., seq-1]
    x_shape = g.op.Shape(x_3d, name=name)
    batch_1d = g.op.Slice(
        x_shape, np.array([0], dtype=np.int64), np.array([1], dtype=np.int64), name=name
    )
    seq_1d = g.op.Slice(
        x_shape, np.array([1], dtype=np.int64), np.array([2], dtype=np.int64), name=name
    )
    seq_scalar = g.op.Squeeze(seq_1d, name=name)
    flat_ids = g.op.Range(
        np.array(0, dtype=np.int64), seq_scalar, np.array(1, dtype=np.int64), name=name
    )
    # Unsqueeze to (1, seq), then Expand to (batch, seq)
    position_ids_1row = g.op.Unsqueeze(flat_ids, np.array([0], dtype=np.int64), name=name)
    expand_shape = g.op.Concat(batch_1d, seq_1d, axis=0, name=name)
    position_ids = g.op.Expand(position_ids_1row, expand_shape, name=name)

    return g.op.RotaryEmbedding(
        x_3d, position_ids, cos_2d, sin_2d, domain="com.microsoft", num_heads=num_heads, name=name
    )


@register_transformer_converter(LlamaAttention)
def llama_attention_to_onnx(
    g: GraphBuilderExtendedProtocol,
    attn: "LlamaAttention",  # noqa: F821
    hidden_states: str,
    cos: str,
    sin: str,
    attention_mask: Optional[str] = None,
    name: str = "llama_attention",
) -> str:
    """
    Appends ONNX nodes implementing
    :class:`transformers.models.llama.modeling_llama.LlamaAttention` to *g*.

    The output dtype (``float32``, ``float16``, or ``bfloat16``) is inferred
    from the registered type of *hidden_states* in *g*.  Model weights are cast
    to match.

    The backend is chosen from the opsets registered in *g*:

    * **com.microsoft** — ``com.microsoft.RotaryEmbedding`` for rotary
      embeddings and ``com.microsoft.MultiHeadAttention`` for attention
      (OnnxRuntime).  The ``"com.microsoft"`` domain must be registered in
      *g*.  GQA KV heads are expanded to match the query head count before
      the attention op.
    * **opset ≥ 24** — standard ONNX ``RotaryEmbedding`` op (opset ≥ 23)
      for rotary embeddings and the standard ONNX ``Attention`` op for
      attention.
    * **opset 23** — standard ONNX ``RotaryEmbedding`` op for rotary
      embeddings and plain ONNX ops (``MatMul``, ``Softmax``,
      ``Transpose``, …) for attention.
    * **opset ≤ 22** — plain ONNX ops for both rotary embeddings and
      attention.  This is the default fallback path.

    .. note::

        The ``cos`` and ``sin`` inputs are expected to carry *symmetric*
        values, i.e. ``cos[..., :head_dim//2] == cos[..., head_dim//2:]``
        (and likewise for ``sin``).  This matches what
        :class:`transformers.models.llama.modeling_llama.LlamaRotaryEmbedding`
        produces.  The dedicated ONNX/ORT ``RotaryEmbedding`` ops use only
        the first half of the last dimension; the plain-op fallback uses
        the full tensor unchanged.

    :param g: an existing graph builder — inputs must already be declared with
        their types; the function appends nodes without creating new graph inputs
        or outputs
    :param attn: a fitted ``LlamaAttention`` module (weights must be
        initialised; the module may be in any of ``float32``, ``float16``, or
        ``bfloat16``)
    :param hidden_states: name of the ``(batch, seq, hidden_size)`` input tensor
        already declared in *g*
    :param cos: name of the ``(batch, seq, head_dim)`` cosine embedding tensor
        already declared in *g*
    :param sin: name of the ``(batch, seq, head_dim)`` sine embedding tensor
        already declared in *g*
    :param attention_mask: optional name of the ``(batch, 1, seq_q, total_seq)``
        attention mask tensor already declared in *g*; pass ``None`` (default)
        when no mask is needed
    :param name: prefix used for all node names added to *g*
    :return: name of the output tensor ``(batch, seq, hidden_size)``; the
        caller is responsible for registering it as a graph output via
        ``g.make_tensor_output``

    Example::

        from onnx import TensorProto
        from yobx.xbuilder import GraphBuilder
        from yobx.torch.in_transformers.models import llama_attention_to_onnx

        g = GraphBuilder({"": 22}, verbose=0)
        g.make_tensor_input("hidden_states", TensorProto.FLOAT, ("batch", "seq", 64))
        g.make_tensor_input("cos", TensorProto.FLOAT, ("batch", "seq", 16))
        g.make_tensor_input("sin", TensorProto.FLOAT, ("batch", "seq", 16))
        out = llama_attention_to_onnx(g, attn, "hidden_states", "cos", "sin")
        g.make_tensor_output(out, TensorProto.FLOAT, ("batch", "seq", 64))
        model = g.to_onnx()
    """
    assert g.has_type(hidden_states), f"Missing type for {hidden_states!r}{g.get_debug_msg()}"

    main_opset = g.main_opset
    has_ms = bool(g.has_opset("com.microsoft"))
    onnx_dtype: int = g.get_type(hidden_states)
    np_dtype = _np_dtype_for_onnx(onnx_dtype)

    # ------------------------------------------------------------------ #
    # Inspect module weights                                               #
    # Use to_numpy() so that bfloat16 weights are handled correctly.      #
    # ------------------------------------------------------------------ #
    q_w = to_numpy(attn.q_proj.weight).astype(np_dtype)  # (num_heads*head_dim, hidden)
    k_w = to_numpy(attn.k_proj.weight).astype(np_dtype)  # (num_kv_heads*head_dim, hidden)
    v_w = to_numpy(attn.v_proj.weight).astype(np_dtype)  # (num_kv_heads*head_dim, hidden)
    o_w = to_numpy(attn.o_proj.weight).astype(np_dtype)  # (hidden, num_heads*head_dim)

    q_b = to_numpy(attn.q_proj.bias).astype(np_dtype) if attn.q_proj.bias is not None else None
    k_b = to_numpy(attn.k_proj.bias).astype(np_dtype) if attn.k_proj.bias is not None else None
    v_b = to_numpy(attn.v_proj.bias).astype(np_dtype) if attn.v_proj.bias is not None else None
    o_b = to_numpy(attn.o_proj.bias).astype(np_dtype) if attn.o_proj.bias is not None else None

    num_heads: int = attn.config.num_attention_heads
    num_kv_heads: int = attn.config.num_key_value_heads
    head_dim: int = attn.head_dim
    n_rep: int = attn.num_key_value_groups
    scaling: float = attn.scaling
    hidden_size: int = attn.config.hidden_size

    # -------------------------------------------------------------- #
    # Branch: com.microsoft RotaryEmbedding + MultiHeadAttention     #
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

        q_3d = _proj_3d(hidden_states, q_w, q_b, "w_q")  # (batch, seq, num_heads * head_dim)
        k_3d = _proj_3d(hidden_states, k_w, k_b, "w_k")  # (batch, seq, num_kv_heads * head_dim)
        v_3d = _proj_3d(hidden_states, v_w, v_b, "w_v")  # (batch, seq, num_kv_heads * head_dim)

        # Apply RoPE using com.microsoft.RotaryEmbedding
        q_3d = _apply_rope_ms_op(g, q_3d, cos, sin, head_dim, num_heads, name)
        k_3d = _apply_rope_ms_op(g, k_3d, cos, sin, head_dim, num_kv_heads, name)

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
                    x_shape,
                    np.array([0], dtype=np.int64),
                    np.array([1], dtype=np.int64),
                    name=name,
                )
                seq_v = g.op.Slice(
                    x_shape,
                    np.array([1], dtype=np.int64),
                    np.array([2], dtype=np.int64),
                    name=name,
                )
                tgt = g.op.Concat(
                    batch_v,
                    seq_v,
                    np.array([num_kv_heads, n_rep, head_dim], dtype=np.int64),
                    axis=0,
                    name=name,
                )
                x5d = g.op.Expand(x5d, tgt, name=name)
                # Reshape → (batch, seq, kv * n_rep * head_dim) with interleaved heads
                sp_3d = np.array([0, 0, -1], dtype=np.int64)
                return g.op.Reshape(x5d, sp_3d, name=name)

            k_3d = _repeat_kv_3d(k_3d)
            v_3d = _repeat_kv_3d(v_3d)

        # MHA: output is (batch, seq, num_heads * head_dim)
        out_3d = _mha_com_microsoft(g, q_3d, k_3d, v_3d, attention_mask, num_heads, scaling, name)

        # Apply output projection
        o_wt = g.make_initializer("w_o", o_w.T)
        attn_out = g.op.MatMul(out_3d, o_wt, name=name)
        if o_b is not None:
            o_bv = g.make_initializer("w_o_bias", o_b)
            attn_out = g.op.Add(attn_out, o_bv, name=name)

    # -------------------------------------------------------------- #
    # Branch: standard ONNX RotaryEmbedding + ONNX Attention (≥ 24) #
    # -------------------------------------------------------------- #
    elif main_opset >= 24:
        # Project to 4D
        q_4d = _project_and_split(g, hidden_states, q_w, q_b, num_heads, head_dim, name, "w_q")
        k_4d = _project_and_split(g, hidden_states, k_w, k_b, num_kv_heads, head_dim, name, "w_k")
        v_4d = _project_and_split(g, hidden_states, v_w, v_b, num_kv_heads, head_dim, name, "w_v")

        # Apply RoPE using standard ONNX RotaryEmbedding op
        q_4d = _apply_rope_standard_onnx(g, q_4d, cos, sin, head_dim, num_heads, name)
        k_4d = _apply_rope_standard_onnx(g, k_4d, cos, sin, head_dim, num_kv_heads, name)

        # Expand KV if needed
        if n_rep > 1:
            k_4d = _repeat_kv(g, k_4d, n_rep, num_kv_heads, head_dim, name)
            v_4d = _repeat_kv(g, v_4d, n_rep, num_kv_heads, head_dim, name)

        # ONNX Attention op
        out_4d = _attention_opset24(g, q_4d, k_4d, v_4d, attention_mask, scaling, name)

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
    # Branch: standard ONNX RotaryEmbedding + plain attention (≥ 23) #
    # -------------------------------------------------------------- #
    elif main_opset >= 23:
        # Project to 4D
        q_4d = _project_and_split(g, hidden_states, q_w, q_b, num_heads, head_dim, name, "w_q")
        k_4d = _project_and_split(g, hidden_states, k_w, k_b, num_kv_heads, head_dim, name, "w_k")
        v_4d = _project_and_split(g, hidden_states, v_w, v_b, num_kv_heads, head_dim, name, "w_v")

        # Apply RoPE using standard ONNX RotaryEmbedding op
        q_4d = _apply_rope_standard_onnx(g, q_4d, cos, sin, head_dim, num_heads, name)
        k_4d = _apply_rope_standard_onnx(g, k_4d, cos, sin, head_dim, num_kv_heads, name)

        # Expand KV if needed
        if n_rep > 1:
            k_4d = _repeat_kv(g, k_4d, n_rep, num_kv_heads, head_dim, name)
            v_4d = _repeat_kv(g, v_4d, n_rep, num_kv_heads, head_dim, name)

        # Standard attention ops
        out_4d = _standard_attention(
            g, q_4d, k_4d, v_4d, attention_mask, scaling, onnx_dtype, name
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

    # -------------------------------------------------------------- #
    # Branch: plain ONNX ops (opset ≤ 22)                            #
    # -------------------------------------------------------------- #
    else:
        # Project to 4D
        q_4d = _project_and_split(g, hidden_states, q_w, q_b, num_heads, head_dim, name, "w_q")
        k_4d = _project_and_split(g, hidden_states, k_w, k_b, num_kv_heads, head_dim, name, "w_k")
        v_4d = _project_and_split(g, hidden_states, v_w, v_b, num_kv_heads, head_dim, name, "w_v")

        # Apply RoPE
        cos4d = g.op.Unsqueeze(cos, np.array([1], dtype=np.int64), name=name)
        sin4d = g.op.Unsqueeze(sin, np.array([1], dtype=np.int64), name=name)
        q_4d = _apply_rope(g, q_4d, cos4d, sin4d, head_dim, name)
        k_4d = _apply_rope(g, k_4d, cos4d, sin4d, head_dim, name)

        # Expand KV if needed
        if n_rep > 1:
            k_4d = _repeat_kv(g, k_4d, n_rep, num_kv_heads, head_dim, name)
            v_4d = _repeat_kv(g, v_4d, n_rep, num_kv_heads, head_dim, name)

        # Standard attention
        out_4d = _standard_attention(
            g, q_4d, k_4d, v_4d, attention_mask, scaling, onnx_dtype, name
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

    return attn_out
