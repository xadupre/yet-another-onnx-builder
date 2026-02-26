-m yobx agg ... aggregate statistics from benchmark runs
=========================================================

The command aggregates statistics produced by benchmarks.
It reads one or more CSV files (or ZIP archives containing CSV files),
combines them into a single dataset, and writes an Excel workbook with
multiple tabs – one per *view*.

Description
+++++++++++

See :class:`yobx.helpers.cube_helper.CubeLogsPerformance`.

.. runpython::

    from yobx._command_lines_parser import get_parser_agg

    get_parser_agg().print_help()

Examples
++++++++

Basic aggregation from a set of ZIP archives:

.. code-block:: bash

    python -m yobx agg test_agg.xlsx raw/*.zip -v 1

Drop the raw-data sheet, keep only the most recent run per key set, and
filter out a specific exporter:

.. code-block:: bash

    python -m yobx agg agg.xlsx raw/*.zip raw/*.csv -v 1 \
        --no-raw --keep-last-date --filter-out "exporter:test-exporter"

Create a time-series view (no recent-only filtering):

.. code-block:: bash

    python -m yobx agg history.xlsx raw/*.csv -v 1 --no-raw \
        --no-recent
