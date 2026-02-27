-m yobx find ... find nodes consuming or producing a result
============================================================

The command looks into an onnx model and searches for a set of names,
reporting which node is consuming or producing each of them.
It can also detect *shadowing names* – results that are defined more than once.

Description
+++++++++++

See :func:`yobx.helpers.onnx_helper.onnx_find` and
:func:`yobx.helpers.onnx_helper.enumerate_results`.

.. runpython::

    from yobx._command_lines_parser import get_parser_find

    get_parser_find().print_help()

Examples
++++++++

Look for specific result names in a model:

.. code-block:: bash

    python -m yobx find -i model.onnx -n "result1,result2"

Detect shadowing names (results defined more than once):

.. code-block:: bash

    python -m yobx find -i model.onnx -n SHADOW

Use the alternative ``enumerate_results`` back-end:

.. code-block:: bash

    python -m yobx find -i model.onnx -n "result1" --v2
