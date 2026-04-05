"""
Patches for :mod:`transformers.masking_utils` to make mask-creation functions
compatible with :class:`~yobx.torch.tracing.CustomTracer` FX symbolic tracing.

Two functions contain control-flow on symbolic tensor-shape values
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
"""

from typing import Optional
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
