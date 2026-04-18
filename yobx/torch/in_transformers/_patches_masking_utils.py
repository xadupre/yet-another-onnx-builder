"""
Patches for :mod:`transformers.masking_utils` and
:mod:`transformers.integrations.sdpa_attention` to make mask-creation and
attention functions compatible with :class:`~yobx.torch.tracing.CustomTracer`
FX symbolic tracing.

Several functions contain control-flow on symbolic tensor-shape values
(``CustomProxy`` instances) that raises ``TraceError`` during FX tracing:

* :func:`transformers.masking_utils.prepare_padding_mask` — performs
  ``if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:``
  where ``kv_length`` and ``attention_mask.shape[-1]`` are both symbolic.
  For valid model inputs the padding length is always 0, so the function is
  patched to skip the pad step when operating on proxy tensors.

* :func:`transformers.masking_utils._ignore_causal_mask_sdpa` — the first
  line evaluates ``padding_mask.shape[-1] > kv_length`` symbolically before
  it reaches the ``is_tracing`` guard, causing a ``TraceError``.  The patch
  adds an early return of ``False`` for proxy inputs, matching the result
  that the original function produces once it reaches the
  ``not is_tracing(padding_mask)`` branch (which is also ``False``).

* :func:`transformers.integrations.sdpa_attention.sdpa_attention_forward` —
  performs ``is_causal = query.shape[2] > 1 and attention_mask is None and ...``
  where ``query.shape[2]`` is a ``CustomProxyInt``.  ``> 1`` produces a
  ``CustomProxyBool`` which cannot be used in Python boolean logic.  The patch
  detects proxy inputs and reconstructs the SDPA call without that control
  flow.
"""

from typing import Optional, Tuple
import torch


def patched_prepare_padding_mask(
    attention_mask: Optional[torch.Tensor], kv_length: int, kv_offset: int
) -> Optional[torch.Tensor]:
    """
    Trace-safe replacement for
    :func:`transformers.masking_utils.prepare_padding_mask`.

    When the result of the length arithmetic is a
    :class:`torch.fx.proxy.Proxy` (i.e. during FX symbolic tracing), the
    function returns *attention_mask* unchanged.  For valid model inputs the
    padding length is always 0, so skipping the pad is semantically correct.

    In all other cases the original behaviour is preserved.
    """
    if attention_mask is None:
        return None
    local_padding_mask = attention_mask
    padding_length = kv_length + kv_offset - attention_mask.shape[-1]
    # During FX tracing kv_length and shape dimensions are proxy objects.
    # bool() on a proxy raises TraceError, so we skip the padding branch.
    if isinstance(padding_length, torch.fx.proxy.Proxy):
        return local_padding_mask
    if padding_length > 0:
        local_padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
    return local_padding_mask


def patched__ignore_causal_mask_sdpa(
    padding_mask: Optional[torch.Tensor],
    query_length: int,
    kv_length: int,
    kv_offset: int,
    local_attention_size: Optional[int] = None,
) -> bool:
    """
    Trace-safe replacement for
    :func:`transformers.masking_utils._ignore_causal_mask_sdpa`.

    The original function's first statement evaluates
    ``padding_mask.shape[-1] > kv_length`` before it reaches the
    ``is_tracing`` guard, causing a ``TraceError`` when these values are FX
    proxies.  This patch returns ``False`` immediately when *padding_mask* is a
    proxy — the same result the original function produces once it reaches the
    ``not is_tracing(padding_mask)`` branch (which is also ``False``).
    """
    if isinstance(padding_mask, torch.fx.proxy.Proxy):
        return False
    # For non-proxy inputs: replicate the original logic inline to avoid a
    # reference to the (now-replaced) module-level function.
    if padding_mask is not None and padding_mask.shape[-1] > kv_length:
        mask_indices = torch.arange(kv_length, device=padding_mask.device)
        mask_indices = mask_indices + kv_offset
        padding_mask = padding_mask[:, mask_indices]
    # is_tracing check (always False for real tensors)
    if (
        not isinstance(padding_mask, torch.fx.proxy.Proxy)
        and (query_length == 1 or kv_length == query_length)
        and (local_attention_size is None or kv_length < local_attention_size)
        and (padding_mask is None or padding_mask.all())
    ):
        return True
    return False


def patched_sdpa_attention_forward(
    module: "torch.nn.Module",
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """
    Trace-safe replacement for
    :func:`transformers.integrations.sdpa_attention.sdpa_attention_forward`.

    During FX symbolic tracing ``query.shape[2]`` is a
    :class:`~yobx.torch.tracing.CustomProxyInt`, and
    ``query.shape[2] > 1 and ...`` raises ``TraceError`` because a
    ``CustomProxyBool`` cannot be evaluated as a Python bool in the ``and``
    expression.  This patch detects proxy inputs and reconstructs the SDPA
    call without that control flow, using ``is_causal=True`` (prefill
    assumption) when the value cannot be determined statically.

    For non-proxy inputs the original function is called unchanged.
    """
    if not isinstance(query, torch.fx.proxy.Proxy):
        # Not tracing — call the original implementation.
        from transformers.integrations.sdpa_attention import sdpa_attention_forward as _orig

        return _orig(
            module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs
        )

    # ── Proxy / FX-tracing path ───────────────────────────────────────────────
    # Repeat KV heads for grouped-query attention.
    if hasattr(module, "num_key_value_groups") and module.num_key_value_groups != 1:
        from transformers.integrations.sdpa_attention import repeat_kv

        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    # Determine is_causal.  The original code uses ``query.shape[2] > 1``
    # which cannot be evaluated during symbolic tracing.  We fall back to the
    # module attribute (True for a standard causal decoder).
    if is_causal is None:
        is_causal = bool(getattr(module, "is_causal", True))
    if not isinstance(is_causal, bool):
        # Proxy or SymBool — conservatively treat as True (prefill phase).
        is_causal = True

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    return attn_output.transpose(1, 2).contiguous(), None
