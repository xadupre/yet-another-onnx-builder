"""
.. _l-plot-einsum-cost-comparison:

Comparing Computational Cost of Two Einsum→ONNX Decomposition Approaches
=========================================================================

:func:`~yobx.helpers.einsum_helper.decompose_einsum` and
:func:`~yobx.helpers.einsum_helper.decompose_einsum_2inputs` both convert an
einsum equation into a graph of primitive ONNX operators, but they follow
different strategies:

* **decompose_einsum** — uses the :class:`~yobx.helpers._einsum.EinsumSubOp` /
  :class:`~yobx.helpers._einsum.GraphEinsumSubOp` framework (``"numpy"``
  strategy by default).  It builds a step-by-step decomposition that can
  handle an arbitrary number of input operands and uses ``Mul``,
  ``ReduceSum``, ``Unsqueeze``, and ``Squeeze`` in addition to ``MatMul``.

* **decompose_einsum_2inputs** — a completely independent implementation
  restricted to exactly two input operands.  It classifies every index
  letter into one of four roles (*batch*, *contract*, *left*, *right*) and
  emits a fixed ``Transpose → Reshape → MatMul → Reshape → Transpose``
  pipeline.

This example compares the two approaches on three representative einsum
equations by counting the number of ONNX nodes each produces and estimating
the floating-point operation (FLOPs) cost with
:class:`~yobx.xshape.BasicShapeBuilder` using ``inference=InferenceMode.COST``.
"""

import numpy as np
import onnxruntime

from yobx.helpers.einsum_helper import decompose_einsum, decompose_einsum_2inputs
from yobx.xshape import BasicShapeBuilder, InferenceMode

# %%
# Helper functions
# ----------------
#
# ``total_flops`` runs :meth:`~yobx.xshape.BasicShapeBuilder.run_model` with
# ``inference=InferenceMode.COST`` to get per-node FLOPs estimates (symbolic
# when the model has dynamic/string input shapes, or integer when the shapes
# are fully concrete), then sums them up.
# ``evaluate_cost_with_true_inputs`` substitutes actual tensor shapes into any
# remaining symbolic expressions.


def total_flops(model, feeds):
    """Computes the total FLOPs for *model* given concrete input *feeds*.

    :param model: ONNX model to evaluate.
    :param feeds: mapping ``{name: array}`` of actual input tensors.
    :returns: Sum of estimated FLOPs across all nodes.
    """
    builder = BasicShapeBuilder()
    cost_sym = builder.run_model(model, inference=InferenceMode.COST)
    cost_conc = builder.evaluate_cost_with_true_inputs(feeds, cost_sym)
    return sum(flops or 0 for _, flops, _ in cost_conc)


def node_count(model):
    """Returns the number of nodes in *model*'s graph.

    :param model: ONNX model to count nodes for.
    :returns: Integer count of nodes in the graph.
    """
    return len(model.graph.node)


rng = np.random.default_rng(0)

# %%
# 1. Matrix multiplication — ``ij,jk->ik``
# -----------------------------------------
#
# The simplest useful einsum: the plain 2-D matrix product.
# Input shapes: ``(64, 128)`` and ``(128, 32)``.

eq1 = "ij,jk->ik"
M, K, N = 64, 128, 32

model1_a = decompose_einsum(eq1, (M, K), (K, N))
model1_b = decompose_einsum_2inputs(eq1, (M, K), (K, N))

feeds1 = {
    "X0": rng.standard_normal((M, K)).astype(np.float32),
    "X1": rng.standard_normal((K, N)).astype(np.float32),
}

flops1_a = total_flops(model1_a, feeds1)
flops1_b = total_flops(model1_b, feeds1)
nodes1_a = node_count(model1_a)
nodes1_b = node_count(model1_b)

print(f"Equation: {eq1}  (M={M}, K={K}, N={N})")
print(f"  decompose_einsum      : {nodes1_a:3d} nodes,  {flops1_a:>12,} FLOPs")
print(f"  decompose_einsum_2inp : {nodes1_b:3d} nodes,  {flops1_b:>12,} FLOPs")

# Numerical sanity check
(r1_a,) = onnxruntime.InferenceSession(
    model1_a.SerializeToString(), providers=["CPUExecutionProvider"]
).run(None, feeds1)
(r1_b,) = onnxruntime.InferenceSession(
    model1_b.SerializeToString(), providers=["CPUExecutionProvider"]
).run(None, feeds1)
assert np.allclose(r1_a, r1_b, atol=1e-5), "Results differ for equation 1"

# %%
# 2. Batched matrix multiplication — ``bij,bjk->bik``
# ----------------------------------------------------
#
# A 3-D batched matrix product that exercises the *batch* letter role in both
# decompositions.
# Input shapes: ``(4, 64, 128)`` and ``(4, 128, 32)``.

eq2 = "bij,bjk->bik"
B = 4

model2_a = decompose_einsum(eq2, (B, M, K), (B, K, N))
model2_b = decompose_einsum_2inputs(eq2, (B, M, K), (B, K, N))

feeds2 = {
    "X0": rng.standard_normal((B, M, K)).astype(np.float32),
    "X1": rng.standard_normal((B, K, N)).astype(np.float32),
}

flops2_a = total_flops(model2_a, feeds2)
flops2_b = total_flops(model2_b, feeds2)
nodes2_a = node_count(model2_a)
nodes2_b = node_count(model2_b)

print(f"\nEquation: {eq2}  (B={B}, M={M}, K={K}, N={N})")
print(f"  decompose_einsum      : {nodes2_a:3d} nodes,  {flops2_a:>12,} FLOPs")
print(f"  decompose_einsum_2inp : {nodes2_b:3d} nodes,  {flops2_b:>12,} FLOPs")

(r2_a,) = onnxruntime.InferenceSession(
    model2_a.SerializeToString(), providers=["CPUExecutionProvider"]
).run(None, feeds2)
(r2_b,) = onnxruntime.InferenceSession(
    model2_b.SerializeToString(), providers=["CPUExecutionProvider"]
).run(None, feeds2)
assert np.allclose(r2_a, r2_b, atol=1e-5), "Results differ for equation 2"

# %%
# 3. Transposed second input — ``ij,kj->ik``
# -------------------------------------------
#
# Same dimensions as equation 1, but the second input has its axes swapped
# so ``j`` (the contracted dimension) is the *last* axis of ``X1`` instead of
# the first.  The decompositions must therefore introduce extra ``Transpose``
# nodes to bring the contracted axes into the right position before the
# ``MatMul``.
# Input shapes: ``(64, 128)`` and ``(32, 128)``.

eq3 = "ij,kj->ik"

model3_a = decompose_einsum(eq3, (M, K), (N, K))
model3_b = decompose_einsum_2inputs(eq3, (M, K), (N, K))

feeds3 = {
    "X0": rng.standard_normal((M, K)).astype(np.float32),
    "X1": rng.standard_normal((N, K)).astype(np.float32),
}

flops3_a = total_flops(model3_a, feeds3)
flops3_b = total_flops(model3_b, feeds3)
nodes3_a = node_count(model3_a)
nodes3_b = node_count(model3_b)

print(f"\nEquation: {eq3}  (M={M}, K={K}, N={N})")
print(f"  decompose_einsum      : {nodes3_a:3d} nodes,  {flops3_a:>12,} FLOPs")
print(f"  decompose_einsum_2inp : {nodes3_b:3d} nodes,  {flops3_b:>12,} FLOPs")

(r3_a,) = onnxruntime.InferenceSession(
    model3_a.SerializeToString(), providers=["CPUExecutionProvider"]
).run(None, feeds3)
(r3_b,) = onnxruntime.InferenceSession(
    model3_b.SerializeToString(), providers=["CPUExecutionProvider"]
).run(None, feeds3)
assert np.allclose(r3_a, r3_b, atol=1e-5), "Results differ for equation 3"

# %%
# 4. Summary table
# ----------------
#
# We collect the results for all three equations and display them.

equations = [eq1, eq2, eq3]
nodes_a = [nodes1_a, nodes2_a, nodes3_a]
nodes_b = [nodes1_b, nodes2_b, nodes3_b]
flops_a = [flops1_a, flops2_a, flops3_a]
flops_b = [flops1_b, flops2_b, flops3_b]

print(
    "\n{:<25s}  {:>6s}  {:>6s}  {:>14s}  {:>14s}".format(
        "Equation", "#n(A)", "#n(B)", "FLOPs(A)", "FLOPs(B)"
    )
)
print("-" * 72)
for eq, na, nb, fa, fb in zip(equations, nodes_a, nodes_b, flops_a, flops_b):
    print(f"{eq:<25s}  {na:>6d}  {nb:>6d}  {fa:>14,}  {fb:>14,}")
print("\n(A) = decompose_einsum  (B) = decompose_einsum_2inputs")

# %%
# 5. Bar chart comparison
# -----------------------
#
# The charts below compare the two approaches side by side for each equation.
# The left panel shows the **node count** (a proxy for graph complexity) and
# the right panel shows the **total FLOPs** (a proxy for theoretical compute
# work).
#
# Both approaches produce graphs with identical FLOPs for the core
# ``MatMul`` computation; any difference in total FLOPs comes from the
# auxiliary ``Transpose``, ``Reshape``, and element-wise nodes that wrap it.

import matplotlib.pyplot as plt  # noqa: E402

labels = [f"eq{i + 1}\n{eq}" for i, eq in enumerate(equations)]
x = np.arange(len(labels))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: node count
ax = axes[0]
bars_a = ax.bar(x - width / 2, nodes_a, width, label="decompose_einsum", color="#4c72b0")
bars_b = ax.bar(x + width / 2, nodes_b, width, label="decompose_einsum_2inputs", color="#dd8452")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Number of ONNX nodes")
ax.set_title("Graph complexity (node count)", fontsize=9)
ax.legend(fontsize=8)
for bar in list(bars_a) + list(bars_b):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.2,
        str(int(bar.get_height())),
        ha="center",
        va="bottom",
        fontsize=7,
    )

# Right: total FLOPs
ax2 = axes[1]
ax2.bar(x - width / 2, flops_a, width, label="decompose_einsum", color="#4c72b0")
ax2.bar(x + width / 2, flops_b, width, label="decompose_einsum_2inputs", color="#dd8452")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel("Total FLOPs")
ax2.set_title("Total estimated FLOPs", fontsize=9)
ax2.legend(fontsize=8)

plt.suptitle("Einsum→ONNX decomposition: node count and FLOPs comparison", fontsize=10)
plt.tight_layout()
plt.show()
