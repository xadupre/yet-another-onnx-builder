"""
.. _l-plot-einsum-cost-comparison:

Comparing Computational Cost of Two Einsum→ONNX Decomposition Approaches
=========================================================================

:func:`~yobx.helpers.einsum_helper.decompose_einsum` and
:func:`~yobx.helpers.einsum_helper.decompose_einsum_2inputs` both convert an
einsum equation into a graph of primitive ONNX operators, but they follow
different strategies:

* **decompose_einsum** (strategy A) — uses the
  :class:`~yobx.helpers._einsum.EinsumSubOp` /
  :class:`~yobx.helpers._einsum.GraphEinsumSubOp` framework (``"numpy"``
  strategy by default).  It builds a step-by-step decomposition that handles
  an arbitrary number of input operands.

* **decompose_einsum_2inputs** (strategy B) — a completely independent
  implementation restricted to exactly two input operands.  It classifies
  every index letter into one of four roles (*batch*, *contract*, *left*,
  *right*) and emits a fixed
  ``Transpose → Reshape → MatMul → Reshape → Transpose`` pipeline.

This example loops over seven representative equations — including 2-D,
3-D, and 4-D cases as well as equations with reduction (contracted indices
absent from the output) — and for each one:

1. Builds both ONNX models.
2. Computes the **symbolic FLOPs** cost (using string-typed input dimensions
   so the cost formula stays general) where cost inference is supported.
3. Evaluates **concrete FLOPs** by substituting actual tensor shapes.
4. Counts the **distribution of operator types** in each graph.
5. Runs a short **runtime benchmark** with :mod:`onnxruntime`.
"""

import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime

from yobx.helpers.einsum_helper import decompose_einsum, decompose_einsum_2inputs
from yobx.xshape import BasicShapeBuilder, InferenceMode

# %%
# 1. Equation registry
# --------------------
#
# Each entry specifies:
#
# * *equation* — the einsum string
# * *sym0 / sym1* — symbolic dimension names for the two inputs (used when
#   computing symbolic FLOPs; ``None`` means the equation is skipped for
#   symbolic analysis)
# * *sh0 / sh1* — concrete shapes used for numerical checks and benchmarks

EQUATIONS = [
    # ── 2-D equations ──────────────────────────────────────────────────────
    {
        "equation": "ij,jk->ik",
        "label": "matmul 2D",
        "sym0": ("M", "K"),
        "sym1": ("K", "N"),
        "sh0": (64, 128),
        "sh1": (128, 32),
    },
    {
        "equation": "ij,ij->i",
        "label": "row dot (reduction)",
        "sym0": ("M", "K"),
        "sym1": ("M", "K"),
        "sh0": (64, 128),
        "sh1": (64, 128),
    },
    # ── 3-D equations ──────────────────────────────────────────────────────
    {
        "equation": "bij,bjk->bik",
        "label": "batched matmul 3D",
        "sym0": ("B", "M", "K"),
        "sym1": ("B", "K", "N"),
        "sh0": (4, 64, 128),
        "sh1": (4, 128, 32),
    },
    {
        "equation": "bij,bj->bi",
        "label": "batched matvec (reduction)",
        "sym0": ("B", "M", "K"),
        "sym1": ("B", "K"),
        "sh0": (4, 64, 128),
        "sh1": (4, 128),
    },
    {
        "equation": "bik,bjk->bij",
        "label": "batch pairwise dot",
        "sym0": ("B", "I", "K"),
        "sym1": ("B", "J", "K"),
        "sh0": (4, 16, 32),
        "sh1": (4, 24, 32),
    },
    # ── 4-D equations ──────────────────────────────────────────────────────
    # Symbolic cost inference is not supported for multi-batch 4-D equations
    # (the shape-inference engine cannot evaluate multi-element Gather nodes
    # that arise from ReduceProd over 2+ batch dimensions).  These entries are
    # still included for node-count and benchmark comparisons.
    {
        "equation": "abij,abjk->abik",
        "label": "multi-batch matmul 4D",
        "sym0": None,
        "sym1": None,
        "sh0": (2, 3, 16, 32),
        "sh1": (2, 3, 32, 8),
    },
    {
        "equation": "abij,ij->ab",
        "label": "4D→2D reduction",
        "sym0": None,
        "sym1": None,
        "sh0": (2, 3, 16, 32),
        "sh1": (16, 32),
    },
]

# %%
# 2. Analysis loop
# ----------------
#
# For each equation we build both ONNX models, attempt symbolic and concrete
# FLOPs estimation, collect node-type distributions, and run a micro-benchmark.

N_BENCH = 50  # repetitions for the timing benchmark

rng = np.random.default_rng(42)
results = []

for spec in EQUATIONS:
    eq = spec["equation"]
    sh0, sh1 = spec["sh0"], spec["sh1"]
    sym0, sym1 = spec["sym0"], spec["sym1"]
    label = spec["label"]

    feeds = {
        "X0": rng.standard_normal(sh0).astype(np.float32),
        "X1": rng.standard_normal(sh1).astype(np.float32),
    }

    row = {"equation": eq, "label": label}

    for key, fn in [("A", decompose_einsum), ("B", decompose_einsum_2inputs)]:
        model = fn(eq, sh0, sh1)

        # --- node type distribution ---
        type_dist = Counter(n.op_type for n in model.graph.node)
        row[f"nodes_{key}"] = sum(type_dist.values())
        row[f"dist_{key}"] = dict(type_dist)

        # --- symbolic + concrete FLOPs ---
        sym_total = None
        conc_total = None
        if sym0 is not None:
            # Build a second model with symbolic (string) dimension names to get
            # symbolic FLOPs expressions.
            try:
                sym_model = fn(eq, sym0, sym1)
                bld_sym = BasicShapeBuilder()
                cost_sym = bld_sym.run_model(sym_model, inference=InferenceMode.COST)
                # Pick the node whose symbolic formula contains the most dimension
                # products (longest string with '*') as a proxy for the most
                # compute-intensive node.  Constant integer FLOPs (no '*') are
                # scalar ops with negligible cost and are excluded.
                sym_totals = [
                    (op, fl) for op, fl, _ in cost_sym if isinstance(fl, str) and "*" in fl
                ]
                if sym_totals:
                    sym_total = max(sym_totals, key=lambda t: len(t[1]))
                # Evaluate with concrete feeds.
                bld_conc = BasicShapeBuilder()
                cost_conc_raw = bld_conc.run_model(model, inference=InferenceMode.COST)
                cost_conc = bld_conc.evaluate_cost_with_true_inputs(feeds, cost_conc_raw)
                conc_total = sum(f or 0 for _, f, _ in cost_conc)
            except RuntimeError:
                # BasicShapeBuilder does not yet handle multi-element Gather nodes
                # that arise when multiple batch/left/right dimensions need to be
                # gathered at once (4-D+ multi-batch equations).  Fall back to
                # reporting cost as unavailable.
                pass

        row[f"sym_{key}"] = sym_total
        row[f"flops_{key}"] = conc_total

        # --- ORT numerical check ---
        sess = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (out,) = sess.run(None, feeds)
        expected = np.einsum(eq, feeds["X0"], feeds["X1"])
        assert np.allclose(out, expected, atol=1e-4), f"Mismatch for {eq} strategy {key}"

        # --- ORT benchmark ---
        # Warm up, then time N_BENCH inference calls.
        for _ in range(3):
            sess.run(None, feeds)
        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            sess.run(None, feeds)
        elapsed_ms = (time.perf_counter() - t0) / N_BENCH * 1000
        row[f"ms_{key}"] = elapsed_ms

    results.append(row)

# %%
# 3. Symbolic FLOPs formulas
# --------------------------
#
# For equations where symbolic shape inference is supported, we print the
# largest symbolic-cost node for each strategy.  The formula uses the symbolic
# dimension names supplied in the equation registry (e.g. ``M``, ``K``, ``N``).

print(f"{'Equation':<28s}  {'Strategy':<4s}  {'Most expensive node FLOPs'}")
print("-" * 75)
for row in results:
    for key in ("A", "B"):
        sym = row[f"sym_{key}"]
        if sym is not None:
            op_name, formula = sym
            label = "A=decompose_einsum" if key == "A" else "B=decompose_einsum_2inp"
            print(f"{row['equation']:<28s}  {label:<22s}  {op_name}: {formula}")
        else:
            print(f"{row['equation']:<28s}  {key:<22s}  (not available)")

# %%
# 4. Summary table: node count, FLOPs, and benchmark
# ---------------------------------------------------

print(
    "\n{:<28s}  {:>5s}  {:>5s}  {:>12s}  {:>12s}  {:>8s}  {:>8s}".format(
        "Equation", "#n(A)", "#n(B)", "FLOPs(A)", "FLOPs(B)", "ms(A)", "ms(B)"
    )
)
print("-" * 90)
for row in results:
    fa = f"{row['flops_A']:>12,}" if row["flops_A"] is not None else f"{'N/A':>12s}"
    fb = f"{row['flops_B']:>12,}" if row["flops_B"] is not None else f"{'N/A':>12s}"
    print(
        f"{row['equation']:<28s}  {row['nodes_A']:>5d}  {row['nodes_B']:>5d}"
        f"  {fa}  {fb}  {row['ms_A']:>7.3f}  {row['ms_B']:>7.3f}"
    )
print("\n(A) = decompose_einsum  (B) = decompose_einsum_2inputs   ms = ms/inference")

# %%
# 5. Operator-type distribution
# -----------------------------
#
# We look at the node-type distributions for the two strategies on a
# representative equation (batched matmul ``bij,bjk->bik``).

target_eq = "bij,bjk->bik"
target_row = next(r for r in results if r["equation"] == target_eq)

all_op_types = sorted(set(target_row["dist_A"]) | set(target_row["dist_B"]))
counts_a = [target_row["dist_A"].get(op, 0) for op in all_op_types]
counts_b = [target_row["dist_B"].get(op, 0) for op in all_op_types]

print(f"\nNode-type distribution for '{target_eq}':")
print(f"  {'Op type':<18s}  {'A':>4s}  {'B':>4s}")
print("  " + "-" * 28)
for op, ca, cb in zip(all_op_types, counts_a, counts_b):
    print(f"  {op:<18s}  {ca:>4d}  {cb:>4d}")

# %%
# 6. Charts
# ---------
#
# **Top-left** — node count per equation (both strategies).
# **Top-right** — concrete FLOPs per equation (where available).
# **Bottom-left** — operator-type distribution for the representative equation.
# **Bottom-right** — mean inference time per equation.

equations_labels = [f"{r['equation']}\n({r['label']})" for r in results]
x = np.arange(len(results))
width = 0.35

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: node count
ax = axes[0, 0]
ba = ax.bar(x - width / 2, [r["nodes_A"] for r in results], width, label="A", color="#4c72b0")
bb = ax.bar(x + width / 2, [r["nodes_B"] for r in results], width, label="B", color="#dd8452")
ax.set_xticks(x)
ax.set_xticklabels(equations_labels, fontsize=7)
ax.set_ylabel("ONNX node count")
ax.set_title("Graph complexity (node count)", fontsize=9)
ax.legend(fontsize=8)
for bar in list(ba) + list(bb):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.15,
        str(int(bar.get_height())),
        ha="center",
        va="bottom",
        fontsize=6,
    )

# Top-right: concrete FLOPs (skip N/A)
ax2 = axes[0, 1]
x_flops = [i for i, r in enumerate(results) if r["flops_A"] is not None]
fa_vals = [results[i]["flops_A"] for i in x_flops]
fb_vals = [results[i]["flops_B"] for i in x_flops]
flop_labels = [equations_labels[i] for i in x_flops]
xf = np.arange(len(x_flops))
ax2.bar(xf - width / 2, fa_vals, width, label="A", color="#4c72b0")
ax2.bar(xf + width / 2, fb_vals, width, label="B", color="#dd8452")
ax2.set_xticks(xf)
ax2.set_xticklabels(flop_labels, fontsize=7)
ax2.set_ylabel("Total FLOPs")
ax2.set_title("Estimated FLOPs (symbolic-capable equations)", fontsize=9)
ax2.legend(fontsize=8)

# Bottom-left: op-type distribution for the representative equation
ax3 = axes[1, 0]
xo = np.arange(len(all_op_types))
ax3.bar(xo - width / 2, counts_a, width, label="A", color="#4c72b0")
ax3.bar(xo + width / 2, counts_b, width, label="B", color="#dd8452")
ax3.set_xticks(xo)
ax3.set_xticklabels(all_op_types, rotation=25, ha="right", fontsize=7)
ax3.set_ylabel("Node count")
ax3.set_title(f"Operator-type distribution — '{target_eq}'", fontsize=9)
ax3.legend(fontsize=8)

# Bottom-right: benchmark (ms/inference)
ax4 = axes[1, 1]
ax4.bar(x - width / 2, [r["ms_A"] for r in results], width, label="A", color="#4c72b0")
ax4.bar(x + width / 2, [r["ms_B"] for r in results], width, label="B", color="#dd8452")
ax4.set_xticks(x)
ax4.set_xticklabels(equations_labels, fontsize=7)
ax4.set_ylabel("Inference time (ms)")
ax4.set_title("OnnxRuntime benchmark (ms / inference)", fontsize=9)
ax4.legend(fontsize=8)

plt.suptitle(
    "Einsum→ONNX: node count, FLOPs, operator distribution and benchmark\n"
    "(A = decompose_einsum, B = decompose_einsum_2inputs)",
    fontsize=10,
)
plt.tight_layout()
plt.show()
