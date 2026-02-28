"""
.. _l-plot-print-command:

Command Line: ``python -m yobx print``
=======================================

This example builds a small ONNX model, saves it to a temporary file, and
then demonstrates the different output formats produced by the
``python -m yobx print`` command.

The same result can be achieved from the terminal with::

    python -m yobx print pretty  model.onnx
    python -m yobx print printer model.onnx
    python -m yobx print dot     model.onnx
"""

import os
import tempfile
import onnx
import onnx.helper as oh
from yobx._command_lines_parser import _cmd_print

TFLOAT = onnx.TensorProto.FLOAT

# %%
# Build a small ONNX model
# ------------------------
#
# The graph computes ``Z = Relu(X + Y)`` with static shapes ``(2, 3)``.

model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Add", ["X", "Y"], ["added"]),
            oh.make_node("Relu", ["added"], ["Z"]),
        ],
        "add_relu",
        [
            oh.make_tensor_value_info("X", TFLOAT, [2, 3]),
            oh.make_tensor_value_info("Y", TFLOAT, [2, 3]),
        ],
        [oh.make_tensor_value_info("Z", TFLOAT, [2, 3])],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=10,
)

# Save to a temporary file so the CLI helpers can load it.
fd, tmp = tempfile.mkstemp(suffix=".onnx")
os.close(fd)
onnx.save(model, tmp)

# %%
# ``pretty`` format
# -----------------
#
# The default format produced by
# :func:`yobx.helpers.onnx_helper.pretty_onnx`.
# It shows opset, inputs/outputs, and every node in a compact,
# human-readable layout.

print("python -m yobx print pretty model.onnx")
print("-" * 40)
_cmd_print(["print", "pretty", tmp])

# %%
# ``printer`` format
# ------------------
#
# Uses the built-in ``onnx.printer.to_text`` renderer which produces the
# official ONNX text representation.

print("python -m yobx print printer model.onnx")
print("-" * 40)
_cmd_print(["print", "printer", tmp])

# %%
# ``dot`` format
# --------------
#
# Dumps the DOT graph source.  Pipe the output to ``dot -Tsvg`` to get
# a visual representation of the graph
# (see also :func:`yobx.helpers.dot_helper.to_dot`).

print("python -m yobx print dot model.onnx")
print("-" * 40)
_cmd_print(["print", "dot", tmp])
