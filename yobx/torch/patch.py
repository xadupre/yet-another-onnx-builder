import contextlib
import traceback
from typing import Iterator, List
from ..helpers.patch_helper import PatchInfo


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


@contextlib.contextmanager
def apply_patches(
    patch_torch: bool = False, patch_transformers: bool = False, verbose: int = 0
) -> Iterator[List[PatchInfo]]:
    """
    The context manager apply patches, usually before exporting a model.

    .. code-block:: python

        from yobx.torch import apply_patches

        with apply_patches(patch_transformers=True):
            # ...
    """
    patches = []
    if patch_torch:
        from .torch.patches import PATCHES

        patches.extend(PATCHES)
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
