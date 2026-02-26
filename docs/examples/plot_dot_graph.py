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
#     :process:
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
