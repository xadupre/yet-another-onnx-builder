"""
.. _l-plot-tracing-replace-attention:

Tracing a model and replacing LlamaAttention with a direct ONNX implementation
===============================================================================

This example shows how to export a model using symbolic tracing
while replacing a specific submodule — here
:class:`~transformers.models.llama.modeling_llama.LlamaAttention` — with a
hand-written ONNX implementation via
:class:`~yobx.torch.interpreter.Dispatcher`.

**Why this matters**

:func:`torch.export.export` (the dynamo path) traces *through* the attention
implementation and produces dozens of individual ``MatMul``, ``Transpose``, and
``Softmax`` nodes.  If you already have a well-tested, optimised ONNX
implementation for the module, you can bypass the tracer entirely for that
submodule and inline your own nodes instead.

**Two-step strategy**

1. **Trace** the outer model with
   :class:`~yobx.torch.tracing.CustomTracer`, declaring
   ``FlatLlamaAttention`` (see below) as a *leaf* so the tracer records a
   single ``call_module`` node rather than tracing inside the attention code.

2. **Translate** that ``call_module`` node to a sequence of ONNX ops using
   :func:`~yobx.torch.in_transformers.models.llama_attention_to_onnx` — the
   same direct converter that is already unit-tested for all three backends
   (plain ops, ONNX ``Attention``, OnnxRuntime ``MultiHeadAttention``).

**Model wrapper**

:class:`~transformers.models.llama.modeling_llama.LlamaAttention` expects
``position_embeddings`` as a ``(cos, sin)`` tuple, which complicates tracing
(the tuple would appear as a non-tensor arg in the FX graph).  The thin
``FlatLlamaAttention`` wrapper unpacks the tuple so that the leaf module's
forward signature is ``forward(hidden_states, cos, sin)`` — three plain
tensors that the tracer handles naturally.

**Command-line options**

Run with ``--opset`` to select the ONNX backend::

    python plot_tracing_replace_attention.py --opset 22    # plain ONNX ops (default)
    python plot_tracing_replace_attention.py --opset 24    # ONNX Attention op
"""

# %%
# Imports
# -------

import argparse
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from onnx import TensorProto
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from yobx import doc
from yobx.reference import ExtendedReferenceEvaluator
from yobx.torch import ExportOptions, to_onnx
from yobx.torch.in_transformers.models import llama_attention_to_onnx
from yobx.torch.interpreter import Dispatcher
from yobx.xbuilder import GraphBuilder

# %%
# Command-line arguments
# ----------------------

parser = argparse.ArgumentParser(description="Tracing + LlamaAttention ONNX replacement demo.")
parser.add_argument(
    "--opset",
    type=int,
    default=22,
    help="Target ONNX opset (default: 22 — plain ops). Use 24 for the ONNX Attention op.",
)
args, _ = parser.parse_known_args(sys.argv[1:])

# %%
# Tiny LlamaAttention configuration
# ----------------------------------
#
# We build a small model so the example runs fast without a GPU.

HIDDEN = 64
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 16

config = LlamaConfig(
    hidden_size=HIDDEN,
    num_attention_heads=NUM_HEADS,
    num_key_value_heads=NUM_KV_HEADS,
    head_dim=HEAD_DIM,
)
print(f"LlamaConfig: hidden={HIDDEN}, heads={NUM_HEADS}, kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")


# %%
# Flat-interface wrapper
# ----------------------
#
# :class:`~transformers.models.llama.modeling_llama.LlamaAttention` accepts
# ``position_embeddings=(cos, sin)`` as a tuple argument.  When the tracer
# encounters a leaf call with a tuple arg, the FX node records a tuple in its
# ``args``, which the interpreter cannot flatten to individual tensor names.
#
# The ``FlatLlamaAttention`` wrapper exposes ``(hidden_states, cos, sin)`` as
# three separate tensors, making the ``call_module`` node fully traceable.
# This is the module we mark as a leaf and register in the dispatcher.


class FlatLlamaAttention(torch.nn.Module):
    """Thin wrapper: accepts ``cos`` and ``sin`` as separate tensors."""

    def __init__(self, attn: LlamaAttention):
        super().__init__()
        self.attn = attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        out, _ = self.attn(hidden_states, position_embeddings=(cos, sin))
        return out


class SimpleAttentionLayer(torch.nn.Module):
    """A minimal model containing a single ``FlatLlamaAttention`` layer."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        inner = LlamaAttention(config, layer_idx=0).eval()
        self.attn = FlatLlamaAttention(inner)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return self.attn(hidden_states, cos, sin)


model = SimpleAttentionLayer(config).eval()

# %%
# Sample inputs
# -------------

torch.manual_seed(42)
hs = torch.randn(2, 10, HIDDEN)
cos = torch.randn(2, 10, HEAD_DIM)
sin = torch.randn(2, 10, HEAD_DIM)

with torch.no_grad():
    expected = model(hs, cos, sin).numpy()

print(f"Input shapes: hidden_states={tuple(hs.shape)}, cos={tuple(cos.shape)}, sin={tuple(sin.shape)}")
print(f"Output shape: {expected.shape}")

# %%
# Reference: direct ONNX construction
# ------------------------------------
#
# Before tracing, let's build the ONNX model by calling
# :func:`~yobx.torch.in_transformers.models.llama_attention_to_onnx` directly.
# This gives us a reference to compare against.

g = GraphBuilder({"": args.opset}, verbose=0)
g.make_tensor_input("hidden_states", TensorProto.FLOAT, ("batch", "seq", HIDDEN))
g.make_tensor_input("cos", TensorProto.FLOAT, ("batch", "seq", HEAD_DIM))
g.make_tensor_input("sin", TensorProto.FLOAT, ("batch", "seq", HEAD_DIM))
out_direct = llama_attention_to_onnx(g, model.attn.attn, "hidden_states", "cos", "sin")
g.make_tensor_output(out_direct, TensorProto.FLOAT, ("batch", "seq", HIDDEN))
onnx_direct = g.to_onnx()

feeds = {"hidden_states": hs.numpy(), "cos": cos.numpy(), "sin": sin.numpy()}
ref_direct = ExtendedReferenceEvaluator(onnx_direct)
got_direct = ref_direct.run(None, feeds)[0]
diff_direct = float(np.abs(expected - got_direct).max())
print(f"Direct ONNX ({args.opset}): {len(onnx_direct.graph.node)} nodes, max_diff={diff_direct:.2e}")

# %%
# Dispatcher function
# --------------------
#
# The dispatcher function has signature ``f(g, module, *tensor_name_args)``.
# When :meth:`~yobx.xbuilder.GraphBuilder.process` encounters a
# ``call_module`` node whose module type is registered in the dispatcher,
# it calls ``f(builder, module_instance, input_name_1, input_name_2, ...)``.
# The function must return the output tensor name (or a list of names).


def flat_llama_to_onnx(
    g: GraphBuilder,
    module: FlatLlamaAttention,
    hidden_states: str,
    cos: str,
    sin: str,
) -> str:
    """Dispatcher: delegates to :func:`llama_attention_to_onnx`."""
    return llama_attention_to_onnx(g, module.attn, hidden_states, cos, sin)


dispatcher = Dispatcher({FlatLlamaAttention: flat_llama_to_onnx})

# %%
# Export with tracing + dispatcher
# ---------------------------------
#
# :class:`~yobx.torch.export_options.ExportOptions` is created with
# ``tracing=True`` to use :class:`~yobx.torch.tracing.CustomTracer` instead
# of :func:`torch.export.export`, and ``tracing_module_leaves`` to declare
# ``FlatLlamaAttention`` as a leaf module (the tracer stops here and records
# a single ``call_module`` node).
#
# The ``dispatcher`` maps that ``call_module`` node to ``flat_llama_to_onnx``.

onnx_traced = to_onnx(
    model,
    kwargs={"hidden_states": hs, "cos": cos, "sin": sin},
    export_options=ExportOptions(
        tracing=True,
        tracing_module_leaves={
            FlatLlamaAttention: lambda m, module_qualified_name=None: True
        },
    ),
    dispatcher=dispatcher,
    target_opset=args.opset,
)

got_traced = ExtendedReferenceEvaluator(onnx_traced).run(None, feeds)[0]
diff_traced = float(np.abs(expected - got_traced).max())

print(f"Traced ONNX ({args.opset}): {len(onnx_traced.graph.node)} nodes, max_diff={diff_traced:.2e}")
assert diff_traced < 1e-4, f"Traced ONNX output differs too much: {diff_traced:.2e}"
print("Outputs match — tracing + dispatcher works correctly.")

# %%
# Compare node counts
# --------------------
#
# Both the direct and the traced export produce the same ONNX ops.

direct_ops = Counter(n.op_type for n in onnx_direct.graph.node)
traced_ops = Counter(n.op_type for n in onnx_traced.graph.node)

op_types = sorted(set(direct_ops) | set(traced_ops))
x = np.arange(len(op_types))
width = 0.35

fig, ax = plt.subplots(figsize=(max(8, len(op_types) * 0.8 + 2), 4))
bars1 = ax.bar(x - width / 2, [direct_ops.get(t, 0) for t in op_types], width, label="direct", color="#4c72b0")
bars2 = ax.bar(x + width / 2, [traced_ops.get(t, 0) for t in op_types], width, label="traced", color="#dd8452")

for bars in (bars1, bars2):
    for bar in bars:
        h = bar.get_height()
        if h:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.1,
                str(int(h)),
                ha="center",
                va="bottom",
                fontsize=8,
            )

ax.set_xticks(x)
ax.set_xticklabels(op_types, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Node count")
ax.set_title(f"ONNX node frequencies — opset {args.opset} (direct vs traced)", fontsize=10)
ax.legend()
fig.tight_layout()
plt.show()

# %%
# Visualise the ONNX graph
# ------------------------
#
# Render the traced ONNX model as a DOT graph.
# This step requires ``graphviz`` to be installed; it is skipped silently
# if the ``dot`` executable is not found.

try:
    doc.save_fig(doc.plot_dot(onnx_traced), "plot_tracing_replace_attention.onnx.png", dpi=400)
except FileNotFoundError:
    pass  # graphviz not installed — skip visualisation
