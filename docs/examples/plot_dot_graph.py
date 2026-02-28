"""
.. _l-plot-dot-graph:

ONNX Graph Visualization with to_dot
======================================

:func:`to_dot <yobx.helpers.dot_helper.to_dot>` converts an
:class:`onnx.ModelProto` into a `DOT <https://graphviz.org/doc/info/lang.html>`_
string that can be rendered by `Graphviz <https://graphviz.org/>`_.

The function:

* assigns different fill colors to well-known op-types (``Shape``,
  ``MatMul``, ``Reshape``, …),
* inlines small scalar constants and 1-D initializers whose length is ≤ 9
  directly onto the node label so the graph stays compact,
* uses :class:`BasicShapeBuilder <yobx.xshape.shape_builder_impl.BasicShapeBuilder>`
  to annotate every edge with its inferred dtype and shape (when available),
* handles ``Scan`` / ``Loop`` / ``If`` sub-graphs by drawing dotted edges for
  outer-scope values consumed by the sub-graph.

The output is a plain DOT string; it can be saved to a ``.dot`` file or passed
to any graphviz renderer (``dot -Tsvg``, ``dot -Tpng``, …).
"""

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.helpers.dot_helper import to_dot

TFLOAT = onnx.TensorProto.FLOAT

# %%
# Build a small model
# --------------------
#
# The graph performs the following operations:
#
# 1. ``Add(X, Y)``  — element-wise sum with shape ``(batch, seq, d)``.
# 2. ``MatMul(added, W)``  — project the last dimension to ``d//2``.
# 3. ``Relu(Z)``  — element-wise ReLU activation.

model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Add", ["X", "Y"], ["added"]),
            oh.make_node("MatMul", ["added", "W"], ["mm"]),
            oh.make_node("Relu", ["mm"], ["Z"]),
        ],
        "add_matmul_relu",
        [
            oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", 4]),
            oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", 4]),
        ],
        [oh.make_tensor_value_info("Z", TFLOAT, ["batch", "seq", 2])],
        [
            onh.from_array(
                np.random.randn(4, 2).astype(np.float32),
                name="W",
            )
        ],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=10,
)

# %%
# Convert to DOT
# ---------------
#
# :func:`to_dot <yobx.helpers.dot_helper.to_dot>` returns the DOT source as a
# plain string.  You can write it to a file and render it with
# ``dot -Tsvg graph.dot > graph.svg``.

dot_src = to_dot(model)
print(dot_src)

# %%
# Display the graph
# ------------------
#
# The DOT source produced above describes the following graph.
#
# .. gdot::
#     :script: DOT-SECTION
#
#     import numpy as np
#     import onnx
#     import onnx.helper as oh
#     import onnx.numpy_helper as onh
#     from yobx.helpers.dot_helper import to_dot
#
#     TFLOAT = onnx.TensorProto.FLOAT
#     model = oh.make_model(
#         oh.make_graph(
#             [
#                 oh.make_node("Add", ["X", "Y"], ["added"]),
#                 oh.make_node("MatMul", ["added", "W"], ["mm"]),
#                 oh.make_node("Relu", ["mm"], ["Z"]),
#             ],
#             "add_matmul_relu",
#             [
#                 oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", 4]),
#                 oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", 4]),
#             ],
#             [oh.make_tensor_value_info("Z", TFLOAT, ["batch", "seq", 2])],
#             [
#                 onh.from_array(
#                     np.random.randn(4, 2).astype(np.float32),
#                     name="W",
#                 )
#             ],
#         ),
#         opset_imports=[oh.make_opsetid("", 18)],
#         ir_version=10,
#     )
#     dot = to_dot(model)
#     print("DOT-SECTION", dot)

# %%
# Plot: graph topology diagram
# -----------------------------
#
# The schematic below shows the same Add → MatMul → Relu pipeline as the
# DOT representation above, drawn with :mod:`matplotlib` patches and
# arrows so that the gallery thumbnail illustrates the data flow.

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

fig, ax = plt.subplots(figsize=(7, 4))
ax.set_xlim(-0.5, 8.5)
ax.set_ylim(-0.5, 5)
ax.axis("off")
ax.set_title("ONNX graph: Add → MatMul → Relu", fontsize=11)

# Node definitions: (x, y, label, face-color)
nodes = [
    (0.5, 4.0, "X\n(batch,seq,4)", "#aaeeaa"),
    (0.5, 2.0, "Y\n(batch,seq,4)", "#aaeeaa"),
    (2.5, 3.0, "Add", "#cccccc"),
    (4.0, 4.3, "W\n(4×2)", "#cccc00"),
    (4.5, 3.0, "MatMul", "#ee9999"),
    (6.5, 3.0, "Relu", "#cccccc"),
    (8.0, 3.0, "Z\n(batch,seq,2)", "#aaaaee"),
]

node_pos = {}
for x, y, label, color in nodes:
    box = mpatches.FancyBboxPatch(
        (x - 0.6, y - 0.45),
        1.2,
        0.9,
        boxstyle="round,pad=0.08",
        linewidth=1.2,
        edgecolor="#888888",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=7.5)
    node_pos[label.split("\n")[0]] = (x, y)

# Arrows
edges = [
    ("X", "Add"),
    ("Y", "Add"),
    ("Add", "MatMul"),
    ("W", "MatMul"),
    ("MatMul", "Relu"),
    ("Relu", "Z"),
]
for src, dst in edges:
    x0, y0 = node_pos[src]
    x1, y1 = node_pos[dst]
    ax.annotate(
        "",
        xy=(x1 - 0.6, y1),
        xytext=(x0 + 0.6, y0),
        arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2),
    )

plt.tight_layout()
plt.show()
