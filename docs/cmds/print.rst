-m yobx print ... print an onnx model on standard output
=========================================================

The command loads an onnx model and prints it on standard output in
one of several formats.

Description
+++++++++++

See :func:`yobx.helpers.onnx_helper.pretty_onnx` and
:class:`yobx.xshape.BasicShapeBuilder`.

.. runpython::

    from yobx._command_lines_parser import get_parser_print

    get_parser_print().print_help()

Examples
++++++++

Print a model in the default *pretty* format:

.. code-block:: bash

    python -m yobx print pretty model.onnx

Print every node with its inferred input/output shapes:

.. code-block:: bash

    python -m yobx print shape model.onnx

Print the raw protobuf representation:

.. code-block:: bash

    python -m yobx print raw model.onnx

Print the onnx-native text representation:

.. code-block:: bash

    python -m yobx print printer model.onnx

Dump the dot graph source:

.. code-block:: bash

    python -m yobx print dot model.onnx

Output on a Dummy Model
+++++++++++++++++++++++

The following block builds a small ``Add + Relu`` model on the fly and runs
the command to show what the output actually looks like.

.. runpython::

    import os
    import tempfile
    import onnx
    import onnx.helper as oh
    from yobx._command_lines_parser import _cmd_print

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

    print("--- pretty ---")
    _cmd_print(["print", "pretty", tmp])
    print()
    print("--- printer ---")
    _cmd_print(["print", "printer", tmp])
