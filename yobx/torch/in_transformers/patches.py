import sys
import warnings
from typing import List, Optional
import torch
import transformers
from ...helpers.patch_helper import PatchInfo
from ._patches_model_rope_utils import common_RotaryEmbedding, patched_dynamic_rope_update

PATCHES: List[PatchInfo] = []


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


def _make_masking_patches() -> List[PatchInfo]:
    """
    Returns patches for :mod:`transformers.masking_utils` and
    :class:`~transformers.modeling_utils.AttentionInterface` that make mask-creation
    and attention functions compatible with FX symbolic tracing.

    These patches are only added when the functions exist in the installed
    version of :mod:`transformers`.
    """
    try:
        import transformers.masking_utils as _masking_utils
    except ImportError:
        return []
    from ._patches_masking_utils import (
        patched__ignore_causal_mask_sdpa,
        patched_prepare_padding_mask,
        patched_sdpa_attention_forward,
    )

    patches = []
    if hasattr(_masking_utils, "prepare_padding_mask"):
        patches.append(
            PatchInfo.make(
                patched_prepare_padding_mask,
                _masking_utils,
                "prepare_padding_mask",
                family="transformers",
                _last_patched_function=_masking_utils.prepare_padding_mask,
            )
        )
    if hasattr(_masking_utils, "_ignore_causal_mask_sdpa"):
        patches.append(
            PatchInfo.make(
                patched__ignore_causal_mask_sdpa,
                _masking_utils,
                "_ignore_causal_mask_sdpa",
                family="transformers",
                _last_patched_function=_masking_utils._ignore_causal_mask_sdpa,
            )
        )
    # Models like LLaMA resolve the attention function via
    # ``ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", ...)`` which reads from
    # ``AttentionInterface._global_mapping`` (a class-level dict shared across
    # all instances).  Patching the module-level attribute on
    # ``transformers.integrations.sdpa_attention`` is *not* sufficient because
    # that dict captures the original function reference at class-definition
    # time and is never updated by a module attribute replacement.
    import transformers.modeling_utils as _modeling_utils

    _AttnInterface = getattr(_modeling_utils, "AttentionInterface", None)
    if _AttnInterface is not None and "sdpa" in _AttnInterface._global_mapping:
        _mapping = _AttnInterface._global_mapping
        _patched = patched_sdpa_attention_forward

        def _do_sdpa_mapping():
            old = _mapping["sdpa"]
            _mapping["sdpa"] = _patched
            return old

        def _undo_sdpa_mapping(original):
            _mapping["sdpa"] = original

        patches.append(
            PatchInfo(
                patch=_patched,
                do=_do_sdpa_mapping,
                undo=_undo_sdpa_mapping,
                family="transformers",
                _last_patched_function=_mapping["sdpa"],
            )
        )
    return patches


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

        return [
            *PATCHES,
            _make_patch_info_for_rotary(LlamaRotaryEmbedding),
            *_make_masking_patches(),
        ]
    patches = list(PATCHES)
    patches.extend(_make_masking_patches())
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
    return patches
