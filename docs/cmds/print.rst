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
