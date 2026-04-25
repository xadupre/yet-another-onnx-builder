"""
.. _l-plot-einsum-decomposition:

Decompose Einsum into Regular ONNX Operators
============================================

The ONNX ``Einsum`` operator is very expressive but not all runtimes
support it.  :func:`decompose_einsum
<yobx.helpers.einsum_helper.decompose_einsum>` converts an einsum
equation into a sub-graph built from simpler operators
(``Transpose``, ``Reshape``, ``MatMul``, ``Mul``, ``ReduceSum``, …)
that every compliant ONNX runtime understands.

The decomposition is delegated to the
:mod:`onnx_extended.tools.einsum` package
(``pip install onnx-extended``).
"""

import numpy as np
import onnxruntime
from yobx.helpers.einsum_helper import decompose_einsum

# %%
# 1. Matrix multiplication — ``ij,jk->ik``
# -----------------------------------------
#
# The simplest useful einsum: multiply two 2-D matrices.

model_mm = decompose_einsum("ij,jk->ik", (3, 4), (4, 5))

# Inspect the generated node types.
print("Node types:", [n.op_type for n in model_mm.graph.node])

# Validate the result numerically.
sess = onnxruntime.InferenceSession(
    model_mm.SerializeToString(), providers=["CPUExecutionProvider"]
)
a = np.random.rand(3, 4).astype(np.float32)
b = np.random.rand(4, 5).astype(np.float32)
(result,) = sess.run(None, {"X0": a, "X1": b})
expected = np.einsum("ij,jk->ik", a, b)
print("max |error|:", np.max(np.abs(result - expected)))
assert np.allclose(result, expected, atol=1e-5)

# %%
# 2. Batched matrix multiplication — ``bij,bjk->bik``
# ----------------------------------------------------
#
# A 3-D batched version of the matrix product.

model_bmm = decompose_einsum("bij,bjk->bik", (2, 3, 4), (2, 4, 5))

print("Node types:", [n.op_type for n in model_bmm.graph.node])

sess2 = onnxruntime.InferenceSession(
    model_bmm.SerializeToString(), providers=["CPUExecutionProvider"]
)
a = np.random.rand(2, 3, 4).astype(np.float32)
b = np.random.rand(2, 4, 5).astype(np.float32)
(result,) = sess2.run(None, {"X0": a, "X1": b})
expected = np.einsum("bij,bjk->bik", a, b)
print("max |error|:", np.max(np.abs(result - expected)))
assert np.allclose(result, expected, atol=1e-5)

# %%
# 3. Three-operand contraction — ``bac,cd,def->ebc``
# ---------------------------------------------------
#
# A more complex equation involving three input tensors.

model_3op = decompose_einsum("bac,cd,def->ebc", (2, 2, 2), (2, 2), (2, 2, 2))

print("Node types:", [n.op_type for n in model_3op.graph.node])

sess3 = onnxruntime.InferenceSession(
    model_3op.SerializeToString(), providers=["CPUExecutionProvider"]
)
x0 = np.random.rand(2, 2, 2).astype(np.float32)
x1 = np.random.rand(2, 2).astype(np.float32)
x2 = np.random.rand(2, 2, 2).astype(np.float32)
(result,) = sess3.run(None, {"X0": x0, "X1": x1, "X2": x2})
expected = np.einsum("bac,cd,def->ebc", x0, x1, x2)
print("max |error|:", np.max(np.abs(result - expected)))
assert np.allclose(result, expected, atol=1e-5)

# %%
# 4. Operator counts comparison
# ------------------------------
#
# The bar chart below shows how many ONNX nodes each decomposed graph
# contains compared to the single ``Einsum`` node it replaces.

import matplotlib.pyplot as plt  # noqa: E402

equations = {
    "ij,jk->ik": [(3, 4), (4, 5)],
    "bij,bjk->bik": [(2, 3, 4), (2, 4, 5)],
    "bac,cd,def->ebc": [(2, 2, 2), (2, 2), (2, 2, 2)],
}

node_counts = {}
for eq, shapes in equations.items():
    model = decompose_einsum(eq, *shapes)
    node_counts[eq] = len(model.graph.node)

labels = list(node_counts.keys())
counts = list(node_counts.values())

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(labels, counts, color="#4c72b0")
ax.axvline(1, color="#dd8452", linestyle="--", label="1 Einsum node")
ax.set_xlabel("Number of ONNX nodes after decomposition")
ax.set_title("Einsum decomposition: node count")
ax.legend()
for bar, count in zip(bars, counts):
    ax.text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height() / 2,
        str(count),
        va="center",
        fontsize=9,
    )
plt.tight_layout()
plt.show()
