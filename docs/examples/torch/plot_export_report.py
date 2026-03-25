"""
.. _l-plot-export-report:

Excel report produced by the torch exporter
============================================

Every call to :func:`to_onnx <yobx.torch.interpreter.to_onnx>` with a
*filename* argument saves two artifacts next to the ``.onnx`` file:

* the ONNX model itself, and
* a companion ``.xlsx`` workbook that contains up to **six sheets** covering
  different aspects of the export process.

This example exports a small model, reads the workbook back, and visualises
the content of every sheet so you can see what each page looks like.

The six sheets are:

``stats``
    One row per optimisation rule application — pattern name, number of nodes
    added/removed, and time spent.
``stats_agg``
    The same data aggregated by rule name and sorted by nodes removed
    (descending).
``extra``
    Scalar key/value pairs recorded during the export: timing entries,
    counters, export-option flags, etc.
``build_stats``
    Timing and counter entries collected by the low-level
    :class:`~yobx.container.BuildStats` object embedded in the model
    container (written only for *large_model* exports).
``node_stats``
    Per-op-type breakdown: how many nodes of each type are in the exported
    model and the estimated FLOPs for each type.
``symbolic_flops``
    Per-node symbolic FLOPs expressions computed by
    :class:`~yobx.xshape.BasicShapeBuilder` with ``InferenceMode.COST``.
    When the model's input shapes contain symbolic dimensions the values are
    symbolic arithmetic strings; for fully static shapes they are integers.
"""

# %%
# Imports
# -------

import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from yobx.torch.interpreter import to_onnx

# %%
# 1. Define and export a model
# ----------------------------
#
# We use a small two-layer MLP so that the export produces a non-trivial
# set of ONNX nodes and a visible optimisation report.


class SmallMLP(torch.nn.Module):
    """Two-layer MLP: Linear → ReLU → Linear."""

    def __init__(self, in_features: int = 16, hidden: int = 32, out_features: int = 8):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden)
        self.fc2 = torch.nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


model = SmallMLP()
x = torch.randn(4, 16)

onnx_path = "plot_export_report.onnx"
xlsx_path = os.path.splitext(onnx_path)[0] + ".xlsx"

# ``filename`` triggers both the ONNX save and the Excel report.
artifact = to_onnx(model, (x,), filename=onnx_path)

print(f"ONNX saved  : {onnx_path}")
print(f"Report saved: {xlsx_path}")
print(f"Nodes in graph: {len(artifact.graph.node)}")
print(f"Report repr  : {artifact.report!r}")

# %%
# 2. Read every sheet from the workbook
# --------------------------------------
#
# :func:`pandas.read_excel` with ``sheet_name=None`` returns an
# ``{sheet_name: DataFrame}`` mapping so we can inspect every page.

sheets: dict[str, pd.DataFrame] = pd.read_excel(xlsx_path, sheet_name=None)
print(f"\nSheets in workbook: {list(sheets)}")
for name, df in sheets.items():
    print(f"\n--- {name} ({df.shape[0]} rows × {df.shape[1]} cols) ---")
    print(df.to_string(index=False))

# %%
# 3. Plot the sheet content
# -------------------------
#
# We render each sheet as a matplotlib table so sphinx-gallery captures
# the output.  Sheets that are absent (e.g. ``build_stats`` for a
# standard-size model) are silently skipped.

ordered_sheets = ["extra", "stats", "stats_agg", "node_stats", "symbolic_flops", "build_stats"]
present = [s for s in ordered_sheets if s in sheets]
n = len(present)

fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
if n == 1:
    axes = [axes]

for ax, sheet_name in zip(axes, present):
    df = sheets[sheet_name]
    ax.axis("off")
    ax.set_title(sheet_name, fontsize=11, fontweight="bold", pad=6)
    if df.empty:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center", transform=ax.transAxes)
        continue
    # Truncate to at most 10 rows for readability
    display_df = df.head(10)
    tbl = ax.table(
        cellText=display_df.values.tolist(),
        colLabels=list(display_df.columns),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.auto_set_column_width(col=list(range(len(display_df.columns))))
    if len(df) > 10:
        ax.text(
            0.5,
            0.01,
            f"… {len(df) - 10} more rows not shown",
            ha="center",
            va="bottom",
            fontsize=7,
            transform=ax.transAxes,
        )

fig.suptitle("Excel report sheets produced by to_onnx()", fontsize=12)
plt.tight_layout()
plt.show()
