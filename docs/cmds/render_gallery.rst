-m yobx render-gallery ... convert a sphinx-gallery example to RST
====================================================================

The command parses a sphinx-gallery Python example file (``.py``) and writes
the equivalent RST source to the corresponding ``auto_examples_<category>/``
folder without executing any code.

For an input file at ``docs/examples/<category>/plot_foo.py`` the output is
written to ``docs/auto_examples_<category>/plot_foo.rst``.  The output
directory is created automatically if it does not exist.

Description
+++++++++++

See :func:`yobx.helpers._gallery_helper.gallery_to_rst` and
:func:`yobx._command_lines_parser._gallery_auto_output_path`.

.. runpython::

    from yobx._command_lines_parser import get_parser_render_gallery

    get_parser_render_gallery().print_help()

Examples
++++++++

Convert a single gallery example:

.. code-block:: bash

    python -m yobx render-gallery docs/examples/core/plot_dot_graph.py

Convert several examples at once:

.. code-block:: bash

    python -m yobx render-gallery \\
        docs/examples/core/plot_dot_graph.py \\
        docs/examples/sklearn/plot_sklearn_pipeline.py
