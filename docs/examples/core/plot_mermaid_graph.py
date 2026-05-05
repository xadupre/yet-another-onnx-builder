"""
.. _l-plot-mermaid-graph:

ONNX Graph Visualization with to_mermaid
==========================================

:func:`to_mermaid <yobx.helpers.mermaid_helper.to_mermaid>` converts an
:class:`onnx.ModelProto` into a `Mermaid <https://mermaid.js.org/>`_
``flowchart TD`` string that can be rendered by any Mermaid-compatible viewer
(e.g. GitHub Markdown, the Mermaid live editor, or Sphinx with the
``sphinxcontrib-mermaid`` extension).

The function:

* assigns different CSS classes to different node kinds (inputs are green,
  initializers are yellow, operators are light-grey, outputs are light-blue),
* inlines small scalar constants and 1-D initializers whose length is ≤ 9
  directly onto the node label so the graph stays compact,
* uses :class:`BasicShapeBuilder <yobx.xshape.shape_builder_impl.BasicShapeBuilder>`
  to annotate every edge with its inferred dtype and shape (when available),
* handles ``Scan`` / ``Loop`` / ``If`` sub-graphs by drawing dotted edges for
  outer-scope values consumed by the sub-graph.

The output is a plain Mermaid string; it can be embedded directly in Markdown
or saved to a ``.mmd`` file.
"""

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
from IPython.display import SVG as IPythonSVG
from yobx.doc import draw_graph_mermaid
from yobx.helpers.mermaid_helper import to_mermaid

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
        [onh.from_array(np.random.randn(4, 2).astype(np.float32), name="W")],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=10,
)

# %%
# Convert to Mermaid
# -------------------
#

mermaid_src = to_mermaid(model)
print(mermaid_src)

# %%
# Display the graph
# ------------------
#
# The diagram is rendered to SVG via the ``mermaid.ink`` online service (through
# :epkg:`mermaid-py`) and displayed using :class:`IPython.display.SVG`.
# When the service is unreachable the raw Mermaid source printed above is
# shown instead so the example never hard-fails in an offline environment.

try:
    IPythonSVG(draw_graph_mermaid(model))
except Exception:
    print(mermaid_src)
