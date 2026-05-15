import sys
import warnings
from typing import List, Optional
import torch
import transformers
from ...helpers.patch_helper import PatchInfo
from ._patches_model_rope_utils import common_RotaryEmbedding, patched_dynamic_rope_update

_ORIGINAL_IGNORE_CAUSAL_MASK_SDPA = transformers.masking_utils._ignore_causal_mask_sdpa
_ORIGINAL_PREPARE_PADDING_MASK = transformers.masking_utils.prepare_padding_mask
_ORIGINAL_SDPA_MASK = transformers.masking_utils.sdpa_mask
_ORIGINAL_CREATE_CAUSAL_MASK = transformers.masking_utils.create_causal_mask
_ORIGINAL_LLAMA_ATTENTION_FORWARD = (
    transformers.models.llama.modeling_llama.LlamaAttention.forward
)


def patched_ignore_causal_mask_sdpa(
    padding_mask: Optional[torch.Tensor],
    query_length: int,
    kv_length: int,
    kv_offset: int,
    local_attention_size: Optional[int] = None,
) -> bool:
    """Returns False when dynamic shapes prevent eager bool evaluation."""
    try:
        return _ORIGINAL_IGNORE_CAUSAL_MASK_SDPA(
            padding_mask, query_length, kv_length, kv_offset, local_attention_size
        )
    except ValueError as exc:
        if "cannot be converted to a Python bool" in str(exc):
            return False
        raise


def patched_prepare_padding_mask(
    attention_mask: Optional[torch.Tensor], kv_length: int, kv_offset: int
) -> Optional[torch.Tensor]:
    """Skips padding when padding length is symbolic."""
    local_padding_mask = attention_mask
    if attention_mask is None:
        return local_padding_mask
    padding_length = kv_length + kv_offset - attention_mask.shape[-1]
    if isinstance(padding_length, int):
        if padding_length > 0:
            return torch.nn.functional.pad(attention_mask, (0, padding_length))
        return local_padding_mask
    symbolic_value = getattr(padding_length, "value", None)
    if isinstance(symbolic_value, int) and symbolic_value > 0:
        return torch.nn.functional.pad(attention_mask, (0, symbolic_value))
    return local_padding_mask


def patched_sdpa_mask(*args, **kwargs):
    """Creates SDPA mask without requiring symbolic ``expand`` shape arguments."""
    use_vmap = kwargs.get("use_vmap", False)
    if use_vmap:
        return _ORIGINAL_SDPA_MASK(*args, **kwargs)

    batch_size = kwargs.get("batch_size", args[0] if len(args) > 0 else None)
    q_length = kwargs.get("q_length", args[1] if len(args) > 1 else None)
    kv_length = kwargs.get("kv_length", args[2] if len(args) > 2 else None)
    q_offset = kwargs.get("q_offset", 0)
    kv_offset = kwargs.get("kv_offset", 0)
    mask_function = kwargs.get("mask_function", transformers.masking_utils.causal_mask_function)
    attention_mask = kwargs.get("attention_mask", None)
    local_size = kwargs.get("local_size", None)
    allow_is_causal_skip = kwargs.get("allow_is_causal_skip", True)
    allow_is_bidirectional_skip = kwargs.get("allow_is_bidirectional_skip", False)
    allow_torch_fix = kwargs.get("allow_torch_fix", True)
    device = kwargs.get("device", "cpu")

    padding_mask = transformers.masking_utils.prepare_padding_mask(
        attention_mask, kv_length, kv_offset
    )
    if allow_is_causal_skip and transformers.masking_utils._ignore_causal_mask_sdpa(
        padding_mask, q_length, kv_length, kv_offset, local_size
    ):
        return None
    if (
        allow_is_bidirectional_skip
        and transformers.masking_utils._ignore_bidirectional_mask_sdpa(
            padding_mask, kv_length, local_size
        )
    ):
        return None
    if padding_mask is not None:
        mask_function = transformers.masking_utils.and_masks(
            mask_function, transformers.masking_utils.padding_mask_function(padding_mask)
        )

    batch_arange = torch.arange(batch_size, device=device)
    head_arange = torch.arange(1, device=device)
    q_arange = torch.arange(q_length, device=device) + q_offset
    kv_arange = torch.arange(kv_length, device=device) + kv_offset
    built_mask = mask_function(
        *transformers.masking_utils._non_vmap_expansion_sdpa(
            batch_arange, head_arange, q_arange, kv_arange
        )
    )
    # Keep broadcastable shape [1,1,Q,KV] (or equivalent) instead of calling
    # Tensor.expand(batch_size, ..., q_length, kv_length), which requires
    # concrete Python ints and fails with TracingInt dynamic dimensions.
    if not transformers.masking_utils._is_torch_greater_or_equal_than_2_5 and allow_torch_fix:
        built_mask = built_mask | torch.all(~built_mask, dim=-1, keepdim=True)
    return built_mask


def patched_create_causal_mask(
    config,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position=None,
    *,
    past_key_values=None,
    position_ids: Optional[torch.Tensor] = None,
    or_mask_function=None,
    and_mask_function=None,
):
    """Builds a causal mask without symbolic bool checks from masking_utils."""
    if type(inputs_embeds).__name__ != "TracingTensor":
        return _ORIGINAL_CREATE_CAUSAL_MASK(
            config=config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
            or_mask_function=or_mask_function,
            and_mask_function=and_mask_function,
        )
    if position_ids is None:
        return _ORIGINAL_CREATE_CAUSAL_MASK(
            config=config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
            or_mask_function=or_mask_function,
            and_mask_function=and_mask_function,
        )
    if attention_mask is not None:
        kv_length = attention_mask.shape[-1]
    elif past_key_values is not None:
        kv_length = past_key_values.get_seq_length() + position_ids.shape[-1]
    else:
        kv_length = position_ids.shape[-1]
    kv_arange = torch.arange(kv_length, device=inputs_embeds.device)
    causal_mask = kv_arange.view(1, 1, 1, -1) <= position_ids[:, None, :, None]
    if attention_mask is not None:
        causal_mask = causal_mask & attention_mask[:, None, None, :].to(torch.bool)
    if getattr(config, "_attn_implementation", None) == "eager":
        return (~causal_mask).to(inputs_embeds.dtype) * torch.finfo(inputs_embeds.dtype).min
    return causal_mask


def patched_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
):
    """Uses eager attention in transformers>=5.8 to avoid SDPA tracing guard issues."""
    if type(hidden_states).__name__ != "TracingTensor":
        return _ORIGINAL_LLAMA_ATTENTION_FORWARD(
            self,
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = transformers.models.llama.modeling_llama.apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )

    attn_output, attn_weights = transformers.models.llama.modeling_llama.eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _make_patch_info_for_rotary(submodule_class):
    patch = PatchInfo.make(
        # patched_dynamic_rope_update(submodule.forward.__wrapped__.__wrapped__),
        common_RotaryEmbedding.forward,
        submodule_class,
        "forward",
        family="transformers",
    )
    patch.add_dependency(
        PatchInfo.make(
            patched_dynamic_rope_update,
            transformers.modeling_rope_utils,
            "dynamic_rope_update",
            family="transformers",
            _last_patched_function=transformers.modeling_rope_utils.dynamic_rope_update,
        )
    )
    return patch


def get_patches_for(model: Optional[torch.nn.Module] = None) -> List[PatchInfo]:
    """
    Returns the list of patches for a specific model.
    if model is None, patches everything it can.

    .. note::
        The function detects that ``RotaryEmbedding.forward`` is wrapped by checking
        if can find substring ``transformers/modeling_rope_utils.py`` in
        ``RotaryEmbedding.forward.__wrapped__``. It does not seem to be the case
        with Python 3.10.
    """
    if model is None:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        from ...pv_version import PvVersion

        use_llama_attention_patch = PvVersion(transformers.__version__) >= PvVersion("5.8")

        patches = [
            _make_patch_info_for_rotary(LlamaRotaryEmbedding),
            PatchInfo.make(
                patched_ignore_causal_mask_sdpa,
                transformers.masking_utils,
                "_ignore_causal_mask_sdpa",
                family="transformers",
            ),
            PatchInfo.make(
                patched_prepare_padding_mask,
                transformers.masking_utils,
                "prepare_padding_mask",
                family="transformers",
            ),
            PatchInfo.make(
                patched_sdpa_mask, transformers.masking_utils, "sdpa_mask", family="transformers"
            ),
            PatchInfo.make(
                patched_create_causal_mask,
                transformers.masking_utils,
                "create_causal_mask",
                family="transformers",
            ),
            PatchInfo.make(
                patched_create_causal_mask,
                transformers.models.llama.modeling_llama,
                "create_causal_mask",
                family="transformers",
            ),
        ]
        if use_llama_attention_patch:
            patches.append(
                PatchInfo.make(
                    patched_llama_attention_forward,
                    transformers.models.llama.modeling_llama.LlamaAttention,
                    "forward",
                    family="transformers",
                )
            )
        return patches
    patches = [
        PatchInfo.make(
            patched_ignore_causal_mask_sdpa,
            transformers.masking_utils,
            "_ignore_causal_mask_sdpa",
            family="transformers",
        ),
        PatchInfo.make(
            patched_prepare_padding_mask,
            transformers.masking_utils,
            "prepare_padding_mask",
            family="transformers",
        ),
        PatchInfo.make(
            patched_sdpa_mask, transformers.masking_utils, "sdpa_mask", family="transformers"
        ),
        PatchInfo.make(
            patched_create_causal_mask,
            transformers.masking_utils,
            "create_causal_mask",
            family="transformers",
        ),
    ]
    for _name, submodule in model.named_modules():
        if (
            hasattr(submodule.forward, "__wrapped__")
            and hasattr(submodule.forward.__wrapped__, "__code__")
            and (
                "transformers/modeling_rope_utils.py"
                in str(submodule.forward.__wrapped__.__code__)
                or "transformers\\modeling_rope_utils.py"
                in str(submodule.forward.__wrapped__.__code__)
            )
        ):
            # RotaryEmbedding is wrapped and this one includes a control-flow.
            if submodule.__class__.__name__.endswith("RotaryEmbedding"):
                patch = _make_patch_info_for_rotary(submodule.__class__)
            elif submodule.__class__.__name__.endswith("RotaryEmbedding") and sys.version_info[
                :2
            ] < (3, 11):
                warnings.warn(
                    "RotaryEmbedding.forward cannot be patched with python<3.11.", UserWarning
                )
                continue
            else:
                raise NotImplementedError(
                    f"Wrapped {submodule.__class__.__name__} is not implemented yet."
                )
            patches.append(patch)
        if submodule.__class__.__name__ == "LlamaAttention":
            from ...pv_version import PvVersion

            if PvVersion(transformers.__version__) >= PvVersion("5.8"):
                patches.append(
                    PatchInfo.make(
                        patched_llama_attention_forward,
                        submodule.__class__,
                        "forward",
                        family="transformers",
                    )
                )
    try:
        import transformers.models.llama.modeling_llama as modeling_llama

        patches.append(
            PatchInfo.make(
                patched_create_causal_mask,
                modeling_llama,
                "create_causal_mask",
                family="transformers",
            )
        )
    except ImportError:
        pass
    return patches
