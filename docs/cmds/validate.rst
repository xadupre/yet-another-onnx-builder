-m yobx validate ... validate a HuggingFace model exported to ONNX
===================================================================

The command validates an ONNX export for a HuggingFace model.
It captures real inputs by running the model on a text prompt with
:class:`~yobx.torch.InputObserver`, exports to ONNX, and checks
numerical discrepancies between the original PyTorch model and the
ONNX runtime outputs.

Description
+++++++++++

See :func:`yobx.torch.validate.validate_model`.

.. runpython::

    from yobx._command_lines_parser import get_parser_validate

    get_parser_validate().print_help()

Examples
++++++++

Basic validation with default settings:

.. code-block:: bash

    python -m yobx validate -m arnir0/Tiny-LLM -v 1

Save ONNX artifacts to a folder for further inspection:

.. code-block:: bash

    python -m yobx validate -m arnir0/Tiny-LLM -v 1 -o dump_validate

Export without applying patches and without running discrepancy checks:

.. code-block:: bash

    python -m yobx validate -m arnir0/Tiny-LLM --no-patch --no-run

Full set of options — target CUDA with float32 at opset 22:

.. code-block:: bash

    python -m yobx validate -m arnir0/Tiny-LLM \
           -e yobx --opt default --opset 22 --device cuda --dtype float32 \
           --patch -r -o dump_test -v 1

Fast validation with random weights (useful for CI):

.. code-block:: bash

    python -m yobx validate -m arnir0/Tiny-LLM --random-weights \
           --config-override num_hidden_layers=2

Override multiple config attributes at once:

.. code-block:: bash

    python -m yobx validate -m arnir0/Tiny-LLM --random-weights \
           --config-override num_hidden_layers=2 \
           --config-override hidden_size=64
