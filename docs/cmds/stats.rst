-m yobx stats ... compute statistics on an onnx model
======================================================

The command analyses an ONNX model and reports the number of nodes per
operator type together with an estimation of the computational cost (FLOPs).
Results can be printed to the terminal or saved to a CSV or Excel file.

Description
+++++++++++

See :func:`yobx.helpers.stats_helper.model_statistics`.

.. runpython::

    from yobx._command_lines_parser import get_parser_stats

    get_parser_stats().print_help()

Examples
++++++++

Print statistics to the terminal:

.. code-block:: bash

    python -m yobx stats model.onnx

Save statistics to a CSV file:

.. code-block:: bash

    python -m yobx stats model.onnx -o stats.csv

Save statistics to an Excel workbook:

.. code-block:: bash

    python -m yobx stats model.onnx -o stats.xlsx

Output on a Dummy Model
+++++++++++++++++++++++

The following block builds a small ``MatMul + Relu`` model on the fly and
runs the command to show what the output actually looks like.

.. runpython::

    import os
    import tempfile
    import onnx
    import onnx.helper as oh
    from yobx._command_lines_parser import _cmd_stats

    TFLOAT = onnx.TensorProto.FLOAT
    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("MatMul", ["X", "W"], ["mm"]),
                oh.make_node("Relu", ["mm"], ["Z"]),
            ],
            "matmul_relu",
            [
                oh.make_tensor_value_info("X", TFLOAT, [4, 8]),
                oh.make_tensor_value_info("W", TFLOAT, [8, 4]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, [4, 4])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )
    fd, tmp = tempfile.mkstemp(suffix=".onnx")
    os.close(fd)
    onnx.save(model, tmp)

    _cmd_stats(["stats", tmp])
