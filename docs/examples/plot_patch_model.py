"""
.. _l-plot-patch-model:

Applying patches to a model and displaying the diff
=====================================================

Before exporting a PyTorch model with :func:`torch.export.export`, a set of
**patches** must be applied to work around limitations in the PyTorch exporter.
This example shows how to:

1. Apply those patches with
   :func:`apply_patches_for_model <yobx.torch.patch.apply_patches_for_model>`.
2. Inspect the registered
   :class:`PatchDetails <yobx.helpers.patch_helper.PatchDetails>` object that
   is yielded by the context manager.
3. Display a unified diff for each
   :class:`PatchInfo <yobx.helpers.patch_helper.PatchInfo>` so you can see
   exactly what changed in the original PyTorch internals.

The context manager both **applies** the patches on entry and **removes** them
on exit, so the original functions are restored once the ``with`` block ends.
"""

import matplotlib.pyplot as plt
import torch
from yobx.helpers.patch_helper import PatchDetails
from yobx.torch import apply_patches_for_model

# %%
# 1. Apply patches and inspect PatchDetails
# ------------------------------------------
#
# :func:`apply_patches_for_model` accepts two boolean flags:
#
# * ``patch_torch=True``  — patches several internal PyTorch functions that
#   prevent successful dynamic-shape export.
# * ``patch_transformers=True`` — adds extra patches for 🤗 Transformers models.
#
# The context manager yields a :class:`PatchDetails` instance that lists every
# :class:`PatchInfo` that was applied.

with apply_patches_for_model(patch_torch=True) as details:
    assert isinstance(details, PatchDetails)
    print(f"Number of patches applied: {details.n_patches}")
    for patch in details:
        print(f"  [{patch.family}] {patch.name}")

# %%
# 2. Display the diff for each patch
# ------------------------------------
#
# After the ``with`` block the patches have been removed, but
# :meth:`PatchInfo.format_diff` still works because the original function
# reference is retained internally.
#
# Each diff is a standard ``unified diff`` — lines starting with ``-`` were
# in the original function; lines starting with ``+`` are in the patched
# version.

for patch in details:
    print(patch.format_diff(format="raw"))
    print()

# %%
# 3. Verify the patches are removed after the context exits
# ----------------------------------------------------------
#
# Applying the same patches a second time succeeds, which proves the first
# ``with`` block cleanly removed them.

with apply_patches_for_model(patch_torch=True) as details2:
    assert details2.n_patches == details.n_patches, (
        f"Expected {details.n_patches} patches, got {details2.n_patches}"
    )
print("Patches removed and re-applied successfully.")

# %%
# 4. Visualise patch sizes
# -------------------------
#
# The chart below shows the total number of changed lines (additions +
# removals) for each patch, giving a quick sense of how invasive each
# rewrite is.

patch_names = []
diff_sizes = []

for patch in details:
    diff = patch.make_diff()
    added = sum(
        1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    removed = sum(
        1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")
    )
    patch_names.append(patch.name)
    diff_sizes.append(added + removed)

fig, ax = plt.subplots(figsize=(max(6, len(patch_names) * 1.5), 4))
bars = ax.bar(range(len(patch_names)), diff_sizes, color="#4c72b0")
ax.set_xticks(range(len(patch_names)))
ax.set_xticklabels(patch_names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Changed lines (added + removed)")
ax.set_title("Size of each torch patch")
for bar, val in zip(bars, diff_sizes):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        str(val),
        ha="center",
        va="bottom",
        fontsize=9,
    )
plt.tight_layout()
plt.show()
