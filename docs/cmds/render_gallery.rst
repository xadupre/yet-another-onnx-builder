-m yobx render-gallery ... convert a sphinx-gallery example to RST
====================================================================

The command parses a sphinx-gallery Python example file (``.py``) and emits
the equivalent RST source without executing any code.  The output can be
previewed in a text editor, checked into version control, or piped into other
tools.

Description
+++++++++++

See :func:`yobx.helpers._gallery_helper.gallery_to_rst`.

.. runpython::

    from yobx._command_lines_parser import get_parser_render_gallery

    get_parser_render_gallery().print_help()

Examples
++++++++

Print RST for a single gallery example to stdout:

.. code-block:: bash

    python -m yobx render-gallery docs/examples/core/plot_dot_graph.py

Save the RST to a file:

.. code-block:: bash

    python -m yobx render-gallery docs/examples/core/plot_dot_graph.py \\
        -o /tmp/plot_dot_graph.rst

Write all examples in a gallery directory to an output directory:

.. code-block:: bash

    python -m yobx render-gallery docs/examples/core/*.py -o /tmp/gallery/
