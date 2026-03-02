from typing import List, Optional
import torch
import transformers
from ...helpers.patch_helper import PatchInfo
from ._patches_model_rope_utils import common_RotaryEmbedding, patched_dynamic_rope_update

PATCHES: List[PatchInfo] = []


def get_patches_for(model: Optional[torch.nn.Module] = None) -> List[PatchInfo]:
    """Returns the list of patches for a specific model."""
    if model is None:
        return PATCHES
    patches = list(PATCHES)
    for _name, submodule in model.named_modules():
        if (
            hasattr(submodule.forward, "__wrapped__")
            and hasattr(submodule.forward.__wrapped__, "__code__")
            and "transformers/modeling_rope_utils.py"
            in str(submodule.forward.__wrapped__.__code__)
        ):
            # RotaryEmbedding is wrapped and this one includes a control-flow.
            if submodule.__class__.__name__.endswith("RotaryEmbedding"):
                patch = PatchInfo.make(
                    # patched_dynamic_rope_update(submodule.forward.__wrapped__.__wrapped__),
                    common_RotaryEmbedding.forward,
                    submodule.__class__,
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
            else:
                raise NotImplementedError(
                    f"Wrapped {submodule.__class__.__name__} is not implemented yet."
                )
            patches.append(patch)
    return patches
