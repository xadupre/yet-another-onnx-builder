import contextlib
import traceback
from typing import Generator, Iterable, Optional
import torch
from ..helpers.patch_helper import PatchDetails, PatchInfo


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
    extra_patches: Optional[Iterable[PatchInfo]] = None,
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
    :param extra_patches: additional :class:`~yobx.helpers.patch_helper.PatchInfo`
        objects to apply alongside the built-in patches.
        Each patch is applied in order after the built-in ones and removed on exit.

    The following shows how to use the output of this function to display
    information about the patches applied to the model.

    .. code-block:: python

        from yobx.torch import apply_patches_for_model

        with apply_patches_for_model(patch_transformers=True, model=model) as details:
            for patch in details:
                diff = patch.make_diff()
                print(f"-- patch {patch!r}")
                print(diff)

    Custom patches can be passed directly via *extra_patches* without any
    manual ``do``/``undo`` bookkeeping:

    .. code-block:: python

        from yobx.helpers.patch_helper import PatchInfo
        from yobx.torch import apply_patches_for_model

        my_patch = PatchInfo.make(my_fn, some_module, "fn_name", family="custom")

        with apply_patches_for_model(patch_torch=True, extra_patches=[my_patch]) as details:
            ep = torch.export.export(model, ...)
    """
    patches = PatchDetails()
    if patch_torch:
        from .in_torch.patches import get_patches

        patches.extend(get_patches())
    if patch_transformers:
        from .in_transformers.patches import get_patches_for

        patches.extend(get_patches_for(model))
    if extra_patches is not None:
        patches.extend(extra_patches)
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
