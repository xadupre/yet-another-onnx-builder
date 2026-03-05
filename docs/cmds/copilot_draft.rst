-m yobx copilot-draft ... draft a sklearn→ONNX converter via GitHub Copilot
===========================================================================

The command queries the GitHub Copilot chat API to generate a first-draft
ONNX converter for any :epkg:`scikit-learn` estimator and writes the result
into ``yobx/sklearn/``.

Description
+++++++++++

See :func:`yobx.helpers.copilot.draft_converter_with_copilot` for the full
API and :ref:`l-design-copilot-draft` for a design overview.

.. runpython::

    from yobx._command_lines_parser import get_parser_copilot_draft

    get_parser_copilot_draft().print_help()

Examples
++++++++

Preview the generated code without writing any file (``--dry-run``):

.. code-block:: bash

    python -m yobx copilot-draft sklearn.linear_model.Ridge --dry-run

Pass the GitHub token explicitly and write to the default location
(``yobx/sklearn/linear_model/ridge.py``):

.. code-block:: bash

    python -m yobx copilot-draft sklearn.linear_model.Ridge \
        --token ghp_...

Write to a custom directory and enable verbose output:

.. code-block:: bash

    python -m yobx copilot-draft sklearn.preprocessing.MinMaxScaler \
        --output-dir /tmp/my_converters -v 1

The GitHub token can also be provided via the ``GITHUB_TOKEN`` or
``GH_TOKEN`` environment variable instead of ``--token``.
