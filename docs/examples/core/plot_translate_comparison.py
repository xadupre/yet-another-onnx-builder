"""
.. _l-plot-translate-comparison:

Translating ONNX models to source code
======================================

:func:`onnx_light.tools.translate` converts an ONNX model into source code
that rebuilds it. The native implementation supports a compact
``onnx_light.onnx.helper`` expression and an incremental ``GraphBuilder``
script.
"""

import numpy as np
import onnx_light.onnx as onnx
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.tools import translate, translate_header

model = oh.make_model(
    oh.make_graph(
        [oh.make_node("Gemm", ["X", "W", "b"], ["T"]), oh.make_node("Relu", ["T"], ["Z"])],
        "gemm_relu",
        [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, 8])],
        [oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [None, 5])],
        [
            onh.from_array(np.random.randn(8, 5).astype(np.float32), name="W"),
            onh.from_array(np.random.randn(5).astype(np.float32), name="b"),
        ],
    ),
    opset_imports=[oh.make_opsetid("", 17)],
    ir_version=9,
)

# %%
# Compact helper expression
# -------------------------

code_compact = translate(model, api="onnx-compact")
print(code_compact)

# %%
# Native GraphBuilder script
# --------------------------

code_builder = translate(model, api="builder")
print(code_builder)

# %%
# Round-trip verification
# -----------------------

namespace = {}
exec(
    compile(translate_header("onnx-compact") + code_compact, "<translate>", "exec"), namespace
)  # noqa: S102
recreated = namespace["model"]
assert [node.op_type for node in recreated.graph.node] == [
    node.op_type for node in model.graph.node
]

# %%
# Compare generated source sizes
# ------------------------------

import matplotlib.pyplot as plt  # noqa: E402

labels = ["onnx-compact", "builder"]
sizes = [len(code_compact), len(code_builder)]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(labels, sizes, color=["#8172b2", "#c44e52"])
ax.set_ylabel("Generated code size (characters)")
ax.set_title("onnx-light translation APIs")
plt.tight_layout()
plt.show()
