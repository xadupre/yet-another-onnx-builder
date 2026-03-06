"""
.. _l-plot-patch-model-diff:

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
4. Render the diff text as a matplotlib figure so that sphinx-gallery
   captures the example.
5. Show which patches were actually exercised when exporting a real model
   (`arnir0/Tiny-LLM`).

The context manager both **applies** the patches on entry and **removes** them
on exit, so the original functions are restored once the ``with`` block ends.
"""

import torch
from yobx.helpers.patch_helper import PatchDetails
from yobx.torch import apply_patches_for_model, register_flattening_functions, use_dyn_not_str
from yobx.torch.tiny_models import get_tiny_model

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
# 3. Plot the diff text as an image
# -----------------------------------
#
# Each unified diff is rendered as a matplotlib figure with colour-coded lines:
# ``-`` lines in red, ``+`` lines in green, and ``@@`` hunk headers in blue.
# This makes the figure capturable by sphinx-gallery.

import matplotlib.pyplot as plt  # noqa: E402

_LINE_COLORS = {"+": "#2a9d2a", "-": "#cc2222", "@": "#1a6fbf"}
_DEFAULT_LINE_COLOR = "#333333"
_MAX_FIG_HEIGHT = 30  # cap total figure height in inches


def _plot_diff(patch_info, ax):
    """Render one unified-diff as coloured text on *ax*."""
    title = patch_info.name
    diff_text = patch_info.make_diff()
    lines = diff_text.splitlines()

    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(len(lines), 1))
    ax.axis("off")
    ax.set_title(title, fontsize=8, loc="left", pad=3)

    for i, line in enumerate(lines):
        color = _LINE_COLORS.get(line[:1], _DEFAULT_LINE_COLOR)
        ax.text(
            0.01,
            len(lines) - i - 0.5,
            line,
            fontsize=6,
            color=color,
            fontfamily="monospace",
            va="center",
            transform=ax.transData,
        )


n = details.n_patches
fig, axes = plt.subplots(n, 1, figsize=(10, min(n * 4, _MAX_FIG_HEIGHT)))
if n == 1:
    axes = [axes]
for ax, patch in zip(axes, details):
    _plot_diff(patch, ax)
plt.tight_layout()
plt.show()

# %%
# 4. Show which patches apply when exporting arnir0/Tiny-LLM
# -----------------------------------------------------------
#
# When exporting a real transformers model we can find out exactly which
# patched functions were exercised by calling
# :meth:`PatchDetails.patches_involved_in_graph` after
# :func:`torch.export.export`.
#
# :func:`register_flattening_functions` must also be active so that the
# :class:`~transformers.DynamicCache` pytree structure is understood by the
# exporter.

data = get_tiny_model("arnir0/Tiny-LLM")
model, inputs, ds = data.model, data.export_inputs, data.dynamic_shapes

with (
    register_flattening_functions(patch_transformers=True),
    apply_patches_for_model(patch_torch=True, patch_transformers=True, model=model) as details2,
):
    ep = torch.export.export(model, (), kwargs=inputs, dynamic_shapes=use_dyn_not_str(ds))

patches = details2.patches_involved_in_graph(ep.graph)
print(f"\nPatches involved in the exported graph: {len(patches)}")
print(details2.make_report(patches))
