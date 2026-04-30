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

The decomposition is implemented in :mod:`yobx.helpers._einsum`, a
self-contained sub-package — no external dependency is required.
"""

import numpy as np
import onnxruntime
from yobx.doc import plot_dot
from yobx.helpers.einsum_helper import decompose_einsum

# %%
# 1. Matrix multiplication — ``ij,jk->ik``
# -----------------------------------------
#
# The simplest useful einsum: multiply two 2-D matrices.

model_mm = decompose_einsum("ij,jk->ik", (3, 4), (4, 5))

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
# Graph of ``ij,jk->ik``
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# :func:`~yobx.doc.plot_dot` renders the decomposed ONNX graph so you can
# see every node and edge at a glance.

plot_dot(model_mm)

# %%
# 2. Batched matrix multiplication — ``bij,bjk->bik``
# ----------------------------------------------------
#
# A 3-D batched version of the matrix product.

model_bmm = decompose_einsum("bij,bjk->bik", (2, 3, 4), (2, 4, 5))

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
# Graph of ``bij,bjk->bik``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

plot_dot(model_bmm)

# %%
# 3. Three-operand contraction — ``bac,cd,def->ebc``
# ---------------------------------------------------
#
# A more complex equation involving three input tensors.

model_3op = decompose_einsum("bac,cd,def->ebc", (2, 2, 2), (2, 2), (2, 2, 2))

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
# Graph of ``bac,cd,def->ebc``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plot_dot(model_3op)
