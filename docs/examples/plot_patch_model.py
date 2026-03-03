"""
.. _l-plot-patch-model:

Patching an ONNX model and displaying the diff
===============================================

This example shows how to **patch** an existing ONNX model — replacing one
operator with another — and then visualise exactly what changed using
:func:`make_diff_code <yobx.helpers.patch_helper.make_diff_code>` combined
with :func:`translate <yobx.translate.translate>`.

Workflow:

1. Build an original model.
2. Patch it (replace ``Relu`` with ``LeakyRelu``).
3. Translate both models to Python source code with ``translate()``.
4. Call ``make_diff_code()`` to produce a unified diff of the two code strings.

The diff immediately shows what changed in the model at the code level, which
is useful when reviewing optimisation passes or debugging graph transformations.
"""

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh

from yobx.helpers.patch_helper import make_diff_code
from yobx.translate import translate

TFLOAT = onnx.TensorProto.FLOAT

# %%
# 1. Build the original model
# ----------------------------
#
# The graph computes ``Y = Relu(X @ W + b)`` — a single linear layer followed
# by a ReLU activation.

W = onh.from_array(np.random.randn(8, 4).astype(np.float32), name="W")
b = onh.from_array(np.zeros(4, dtype=np.float32), name="b")

original = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Gemm", ["X", "W", "b"], ["Z"]),
            oh.make_node("Relu", ["Z"], ["Y"]),
        ],
        "linear_relu",
        [oh.make_tensor_value_info("X", TFLOAT, [None, 8])],
        [oh.make_tensor_value_info("Y", TFLOAT, [None, 4])],
        [W, b],
    ),
    opset_imports=[oh.make_opsetid("", 17)],
    ir_version=9,
)
onnx.checker.check_model(original)
print(f"Original model: {[n.op_type for n in original.graph.node]}")

# %%
# 2. Patch the model
# -------------------
#
# Replace the ``Relu`` node with a ``LeakyRelu`` node (``alpha=0.01``).
# We rebuild the graph from scratch, keeping everything the same except for
# the activation node.

patched = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Gemm", ["X", "W", "b"], ["Z"]),
            oh.make_node("LeakyRelu", ["Z"], ["Y"], alpha=0.01),
        ],
        "linear_relu",
        [oh.make_tensor_value_info("X", TFLOAT, [None, 8])],
        [oh.make_tensor_value_info("Y", TFLOAT, [None, 4])],
        [W, b],
    ),
    opset_imports=[oh.make_opsetid("", 17)],
    ir_version=9,
)
onnx.checker.check_model(patched)
print(f"Patched  model: {[n.op_type for n in patched.graph.node]}")

# %%
# 3. Translate both models to Python source
# ------------------------------------------
#
# :func:`translate <yobx.translate.translate>` converts each
# :class:`onnx.ModelProto` into a self-contained Python snippet.
# We use the ``"onnx-short"`` API so that large initialisers are replaced
# by compact ``np.random.randn(…)`` calls, keeping the diff readable.

code_original = translate(original, api="onnx-short")
code_patched = translate(patched, api="onnx-short")

print("=== original ===")
print(code_original)

# %%
# 4. Display the diff
# --------------------
#
# :func:`make_diff_code <yobx.helpers.patch_helper.make_diff_code>` produces
# a standard unified diff between two source strings.  Lines starting with
# ``-`` were removed; lines starting with ``+`` were added.

diff = make_diff_code(code_original, code_patched)
print("=== diff (original -> patched) ===")
print(diff)

assert "Relu" in diff, "expected 'Relu' to appear in the diff"
assert "LeakyRelu" in diff, "expected 'LeakyRelu' to appear in the diff"
assert "-" in diff and "+" in diff, "expected both additions and removals in the diff"

# %%
# 5. Visualise the diff as a bar chart
# -------------------------------------
#
# The chart below counts how many lines were added, removed, or unchanged in
# the diff.  For a single-node replacement the numbers are small, but the same
# approach scales to complex graph rewrites.

import matplotlib.pyplot as plt  # noqa: E402

added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
removed = sum(
    1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")
)
context = sum(
    1
    for line in diff.splitlines()
    if line and not line.startswith(("+", "-", "@", "\\"))
)

fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(
    ["added", "removed", "context"],
    [added, removed, context],
    color=["#55a868", "#c44e52", "#4c72b0"],
)
ax.set_ylabel("Number of lines")
ax.set_title("Unified diff: Relu → LeakyRelu")
for bar, val in zip(bars, [added, removed, context]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        str(val),
        ha="center",
        va="bottom",
        fontsize=10,
    )
plt.tight_layout()
plt.show()
