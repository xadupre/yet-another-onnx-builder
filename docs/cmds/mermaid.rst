-m yobx mermaid ... convert an onnx model into Mermaid flowchart format
========================================================================

The command converts an onnx model into a
:epkg:`Mermaid` flowchart string that can be embedded
in Markdown, GitHub READMEs, or any tool that renders Mermaid diagrams.

Description
+++++++++++

See :func:`yobx.helpers.mermaid_helper.to_mermaid`.

.. runpython::

    from yobx._command_lines_parser import get_parser_mermaid

    get_parser_mermaid().print_help()

Examples
++++++++

Print the Mermaid source to the terminal:

.. code-block:: bash

    python -m yobx mermaid model.onnx

Save the Mermaid source to a file:

.. code-block:: bash

    python -m yobx mermaid model.onnx -o model.mmd

Output on a Dummy Model
+++++++++++++++++++++++

The following block builds a small ``Add + Relu`` model on the fly and runs
the command to show what the output actually looks like.

.. runpython::

    import os
    import tempfile
    import onnx
    import onnx.helper as oh
    from yobx._command_lines_parser import _cmd_mermaid

    TFLOAT = onnx.TensorProto.FLOAT
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
    fd, tmp = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    onnx.save(model, tmp)

    _cmd_mermaid(["mermaid", tmp])
