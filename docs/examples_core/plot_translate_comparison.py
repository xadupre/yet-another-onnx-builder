"""
.. _l-plot-translate-comparison:

Comparing the four ONNX translation APIs
=========================================

:func:`translate <yobx.translate.translate>` converts an
:class:`onnx.ModelProto` into Python source code that, when executed,
recreates the same model.  Four output APIs are available:

* ``"onnx"`` — uses :mod:`onnx.helper` (``oh.make_node``, ``oh.make_graph``, …)
  via :class:`~yobx.translate.inner_emitter.InnerEmitter`.
* ``"onnx-short"`` — same as ``"onnx"`` but replaces large initializers with
  random values to keep the snippet compact, via
  :class:`~yobx.translate.inner_emitter.InnerEmitterShortInitializer`.
* ``"light"`` — fluent ``start(…).vin(…).…`` chain,
  via :class:`~yobx.translate.light_emitter.LightEmitter`.
* ``"builder"`` — ``GraphBuilder``-based function wrapper,
  via :class:`~yobx.translate.builder_emitter.BuilderEmitter`.

This example builds a small model, translates it with every API, shows the
generated code, and verifies that the ``"onnx"`` snippet can be re-executed to
reproduce the original model.
"""

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.translate import translate, translate_header

# %%
# Build the model
# ----------------
#
# We use ``Z = Relu(X @ W + b)`` as a running example:
# a single ``Gemm`` followed by ``Relu``.

TFLOAT = onnx.TensorProto.FLOAT
INT64 = onnx.TensorProto.INT64

W = onh.from_array(np.random.randn(8, 5).astype(np.float32), name="W")
b = onh.from_array(np.random.randn(5).astype(np.float32), name="b")

model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Gemm", ["X", "W", "b"], ["T"]),
            oh.make_node("Relu", ["T"], ["Z"]),
        ],
        "gemm_relu",
        [oh.make_tensor_value_info("X", TFLOAT, [None, 8])],
        [oh.make_tensor_value_info("Z", TFLOAT, [None, 5])],
        [W, b],
    ),
    opset_imports=[oh.make_opsetid("", 17)],
    ir_version=9,
)

print(f"Model: {len(model.graph.node)} node(s), {len(model.graph.initializer)} initializer(s)")

# %%
# 1. ``"onnx"`` API — full initializer values
# ---------------------------------------------
#
# The generated code uses :func:`onnx.helper.make_node`,
# :func:`onnx.helper.make_graph`, and :func:`onnx.helper.make_model`.
# Every initializer is serialised as an exact ``np.array(…)`` literal.

code_onnx = translate(model, api="onnx")
print("=== api='onnx' ===")
print(code_onnx)

# %%
# 2. ``"onnx-short"`` API — large initializers replaced by random values
# -----------------------------------------------------------------------
#
# Identical to ``"onnx"`` except that initializers with more than 16 elements
# are replaced by ``np.random.randn(…)`` / ``np.random.randint(…)`` calls.
# This keeps the snippet readable when dealing with large weight tensors.

code_short = translate(model, api="onnx-short")
print("=== api='onnx-short' ===")
print(code_short)

# %%
# Size comparison between the two onnx variants:

print(f"\nFull code length  : {len(code_onnx):>6} characters")
print(f"Short code length : {len(code_short):>6} characters")

# %%
# 3. ``"light"`` API — fluent chain
# -----------------------------------
#
# The output is a single method-chain expression (``start(…).vin(…).…``).

code_light = translate(model, api="light")
print("=== api='light' ===")
print(code_light)

# %%
# 4. ``"builder"`` API — GraphBuilder
# -------------------------------------
#
# The output uses ``GraphBuilder`` to wrap the graph nodes in a Python function.

code_builder = translate(model, api="builder")
print("=== api='builder' ===")
print(code_builder)

# %%
# Round-trip verification
# -----------------------
#
# The ``"onnx"`` snippet is fully self-contained and executable.
# Running it should recreate a model with the same graph structure.

header = translate_header("onnx")
full_code = header + "\n" + code_onnx
ns: dict = {}
exec(compile(full_code, "<translate>", "exec"), ns)  # noqa: S102
recreated = ns["model"]

assert isinstance(recreated, onnx.ModelProto)
assert len(recreated.graph.node) == len(
    model.graph.node
), f"Expected {len(model.graph.node)} nodes, got {len(recreated.graph.node)}"
assert len(recreated.graph.initializer) == len(model.graph.initializer), (
    f"Expected {len(model.graph.initializer)} initializers, "
    f"got {len(recreated.graph.initializer)}"
)
print("\nRound-trip succeeded ✓")

# %%
# Plot: code size by API
# -----------------------
#
# The bar chart compares the number of characters produced by each API for the
# same model.  ``"onnx-short"`` is always ≤ ``"onnx"`` because it compresses
# large initializers.

import matplotlib.pyplot as plt  # noqa: E402

api_labels = ["onnx", "onnx-short", "light", "builder"]
code_sizes = [len(code_onnx), len(code_short), len(code_light), len(code_builder)]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(api_labels, code_sizes, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
ax.set_ylabel("Generated code size (characters)")
ax.set_title("ONNX translation: code size by API")
for bar, size in zip(bars, code_sizes):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.01,
        str(size),
        ha="center",
        va="bottom",
        fontsize=9,
    )
plt.tight_layout()
plt.show()
