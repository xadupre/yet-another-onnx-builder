-m yobx run-doc-examples ... run all runpython:: and gdot:: examples
======================================================================

The command scans RST documentation files and/or Python source files for
``.. runpython::`` and ``.. gdot::`` directives, executes each embedded code
block in an isolated subprocess, and reports which blocks pass and which fail.

Description
+++++++++++

See :func:`yobx.helpers._check_runpython.extract_runpython_blocks` and
:func:`yobx.helpers._check_runpython.run_runpython_blocks`.

.. runpython::

    from yobx._command_lines_parser import get_parser_run_doc_examples

    get_parser_run_doc_examples().print_help()

Examples
++++++++

Check all examples in a single RST file:

.. code-block:: bash

    python -m yobx run-doc-examples docs/design/misc/helpers.rst

Check all RST files inside a directory (recursively):

.. code-block:: bash

    python -m yobx run-doc-examples docs/ -v 1

Check Python docstring examples in a source file:

.. code-block:: bash

    python -m yobx run-doc-examples yobx/helpers/helper.py -v 2

Scan several paths at once with a per-block timeout:

.. code-block:: bash

    python -m yobx run-doc-examples \\
        docs/design/misc/helpers.rst \\
        yobx/helpers/helper.py \\
        --timeout 60 -v 1

Exit code
+++++++++

The command exits with **0** when every block passes (or when no blocks are
found), and with **1** when at least one block fails or times out.
This makes it easy to use in CI pipelines:

.. code-block:: bash

    python -m yobx run-doc-examples docs/ || exit 1
