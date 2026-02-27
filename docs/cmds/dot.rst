-m yobx dot ... convert an onnx model into dot format
=====================================================

The command converts an onnx model into a `dot` file that can be
rendered with `Graphviz <https://graphviz.org/>`_ into an image (SVG, PNG, …).

Description
+++++++++++

See :func:`yobx.helpers.dot_helper.to_dot`.

.. runpython::

    from yobx._command_lines_parser import get_parser_dot

    get_parser_dot().print_help()

Examples
++++++++

Convert a model to dot and render it as SVG in one step:

.. code-block:: bash

    python -m yobx dot model.onnx -o model.dot --run svg

Or just print the dot source to the terminal:

.. code-block:: bash

    python -m yobx dot model.onnx
