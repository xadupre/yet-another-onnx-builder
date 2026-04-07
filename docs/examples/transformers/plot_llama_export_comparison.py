"""
.. _l-plot-llama-export-comparison:

Compare custom exporter and torch.onnx.export on LlamaModel
============================================================

This example builds a small, **untrained** (randomly initialised)
:class:`transformers.LlamaModel` with 2 hidden layers directly from
:class:`transformers.LlamaConfig` — no HuggingFace hub download required.

The same model is then exported to ONNX using two different exporters:

* **custom** — :func:`yobx.torch.to_onnx` (the custom FX-based exporter
  provided by this library).
* **dynamo** — :func:`torch.onnx.export` with the ``dynamo`` backend
  (``torch >= 2.4``).

The resulting ONNX graphs are compared by counting how many times each
``op_type`` appears.  The counts are rendered as grouped horizontal bar charts
so the two exporters can be inspected side-by-side.

**No command-line arguments are needed.**  The model is always created with
random weights so the example finishes quickly and does not require internet
access.
"""

# %%
# Imports
# -------

import warnings

import matplotlib.pyplot as plt
import numpy as np
import onnx
import torch
from transformers import LlamaConfig, LlamaModel

from yobx.torch import apply_patches_for_model, register_flattening_functions, to_onnx

# %%
# Build a tiny, untrained 2-layer LlamaModel
# ------------------------------------------
#
# :class:`transformers.LlamaConfig` lets us set every architectural
# hyperparameter explicitly.  By choosing small hidden/intermediate sizes we
# keep the model lightweight while still having a realistic two-layer
# transformer body.  No pre-trained weights are downloaded — the parameters
# are initialised at random.

config = LlamaConfig(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=32,
    max_position_embeddings=256,
    vocab_size=256,
)

model = LlamaModel(config)
model.eval()

print(
    f"LlamaModel — hidden_size={config.hidden_size}, "
    f"num_hidden_layers={config.num_hidden_layers}, "
    f"#params={sum(p.numel() for p in model.parameters()):,}"
)

# %%
# Define inputs and dynamic shapes
# ---------------------------------
#
# ``LlamaModel.forward`` requires ``input_ids`` and accepts an optional
# ``attention_mask``.  We declare the batch and sequence dimensions as dynamic
# so the exported model can be used with arbitrary input sizes.

batch, seq = 2, 8

inputs = dict(
    input_ids=torch.randint(0, config.vocab_size, (batch, seq)),
    attention_mask=torch.ones((batch, seq), dtype=torch.int64),
)

dynamic_shapes = {
    "input_ids": {0: "batch", 1: "seq_length"},
    "attention_mask": {0: "batch", 1: "seq_length"},
}

# Compute a reference output for the discrepancy check later.
with torch.no_grad():
    expected = model(**inputs)

# %%
# Export with the custom exporter (yobx)
# ----------------------------------------
#
# :func:`yobx.torch.to_onnx` drives a custom FX interpreter that translates
# every ATen / Torch op to the corresponding ONNX primitive.
# :func:`register_flattening_functions` and :func:`apply_patches_for_model`
# ensure that any Transformers-specific helper classes (e.g. KV-cache) and ops
# are handled correctly during the FX graph capture.

custom_filename = "plot_llama_export_comparison.custom.onnx"

try:
    with (
        register_flattening_functions(patch_transformers=True),
        apply_patches_for_model(patch_transformers=True, model=model),
    ):
        to_onnx(model, kwargs=inputs, filename=custom_filename, dynamic_shapes=dynamic_shapes)
    custom_ok = True
    print("custom exporter — export ok")
except Exception as e:
    custom_ok = False
    print(f"custom exporter — export failed: {e}")

# %%
# Export with torch.onnx.export (dynamo backend)
# -----------------------------------------------
#
# :func:`torch.onnx.export` with the *dynamo* backend (available since
# ``torch >= 2.4``) uses :func:`torch.export.export` internally to capture a
# fully-lowered FX graph and then translates it to ONNX.

dynamo_filename = "plot_llama_export_comparison.dynamo.onnx"

try:
    with (
        register_flattening_functions(patch_transformers=True),
        apply_patches_for_model(patch_transformers=True, model=model),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model, (), dynamo_filename, kwargs=inputs, dynamic_shapes=dynamic_shapes
        )
    dynamo_ok = True
    print("torch.onnx.export (dynamo) — export ok")
except Exception as e:
    dynamo_ok = False
    print(f"torch.onnx.export (dynamo) — export failed: {e}")

# %%
# Summarise: node-type counts
# ----------------------------
#
# For each successful export we load the ONNX protobuf and count how many times
# each ``op_type`` appears in the top-level graph.  The resulting frequency
# tables are printed for reference.

successful_exports: dict[str, str] = {}
if custom_ok:
    successful_exports["custom"] = custom_filename
if dynamo_ok:
    successful_exports["dynamo"] = dynamo_filename

counts_per_exporter: dict[str, dict[str, int]] = {}
for exp_name, fname in successful_exports.items():
    proto = onnx.load(fname)
    freq: dict[str, int] = {}
    for node in proto.graph.node:
        freq[node.op_type] = freq.get(node.op_type, 0) + 1
    counts_per_exporter[exp_name] = freq
    print(f"\n=== {exp_name} — {len(proto.graph.node)} nodes total ===")
    for op, cnt in sorted(freq.items(), key=lambda x: -x[1]):
        print(f"  {op:<30} {cnt:>4}")

# %%
# Visualise: grouped bar chart
# ----------------------------
#
# The bar chart below places the two exporters side-by-side for every
# ``op_type`` that appears in at least one of the exported graphs.  Op types
# present in only one export show a zero bar for the other exporter, making it
# easy to spot coverage differences.

if counts_per_exporter:
    all_op_types = sorted({op for freq in counts_per_exporter.values() for op in freq})
    n_ops = len(all_op_types)
    n_exp = len(counts_per_exporter)
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2"]

    fig, ax = plt.subplots(figsize=(8, max(4, n_ops * 0.4 + 2)))
    y = np.arange(n_ops)
    height = 0.8 / max(n_exp, 1)

    for idx, (exp_name, freq) in enumerate(counts_per_exporter.items()):
        vals = [freq.get(op, 0) for op in all_op_types]
        offset = (idx - (n_exp - 1) / 2) * height
        bars = ax.barh(y + offset, vals, height, label=exp_name, color=colors[idx % len(colors)])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    str(val),
                    ha="left",
                    va="center",
                    fontsize=7,
                )

    ax.set_yticks(y)
    ax.set_yticklabels(all_op_types, fontsize=8)
    ax.set_xlabel("Number of nodes")
    ax.set_title(
        "ONNX node frequencies — 2-layer LlamaModel\n(custom vs torch.onnx.export)", fontsize=10
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.show()
else:
    print("No successful exports — nothing to plot.")
