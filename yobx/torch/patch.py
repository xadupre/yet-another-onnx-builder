import contextlib
import traceback
from typing import Generator, Optional
import torch
from ..helpers.patch_helper import PatchDetails


def retrieve_stacktrace():
    """Retrieves and prints the current stack trace, avoids every torch file."""
    rows = []
    stack_frames = traceback.extract_stack()
    for frame in stack_frames:
        filename, lineno, function_name, code_line = frame
        if "/torch/" in filename:
            continue
        rows.append(f"File: {filename}, Line {lineno}, in {function_name}")
        if code_line:
            rows.append(f"    {code_line}")
    return "\n".join(rows)


@contextlib.contextmanager  # type: ignore
def apply_patches_for_model(
    patch_torch: bool = False,
    patch_transformers: bool = False,
    verbose: int = 0,
    model: Optional[torch.nn.Module] = None,
    tracing: bool = False,
) -> Generator[PatchDetails, None, None]:
    """
    The context manager apply patches, usually before exporting a model.

    .. code-block:: python

        from yobx.torch import apply_patches_for_model

        with apply_patches_for_model(patch_transformers=True, model=model):
            # ...

    :param patch_torch: applies patches for :epkg:`torch`
    :param patch_transformers: applies patch for transformers
    :param verbose: prints out which patch is applies
    :param model: modifies the list of patches for a particular model,
        it is recommended to fill it the used rope is not the default one
    :param tracing: when ``True``, also includes patches required for FX
        symbolic tracing (e.g. for mask-creation utilities and
        ``sdpa_attention_forward``).  These patches must not be applied during
        ordinary eager-mode inference because they alter control-flow in a way
        that is only correct for the symbolic-tracing code path.

    The following shows how to use the output of this function to display
    information about the patches applied to the model.

    .. code-block:: python

        from yobx.torch import apply_patches_for_model

        with apply_patches_for_model(patch_transformers=True, model=model) as details:
            for patch in details:
                diff = patch.make_diff()
                print(f"-- patch {patch!r}")
                print(diff)
    """
    patches = PatchDetails()
    if patch_torch:
        from .in_torch.patches import PATCHES

        patches.extend(PATCHES)
    if patch_transformers:
        from .in_transformers.patches import get_patches_for

        patches.extend(get_patches_for(model, tracing=tracing))
    for patch in patches:
        if verbose:
            print(f"[register_patch_functions] apply {patch}")
        patch.do()
    try:
        yield patches
    finally:
        for patch in patches:
            if verbose:
                print(f"[register_patch_functions] remove {patch!r}")
            patch.undo()
