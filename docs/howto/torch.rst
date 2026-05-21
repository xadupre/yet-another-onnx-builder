.. _l-howto-torch:

torch
=====

This page answers common *"how do I…"* questions for converting
:epkg:`torch` models to ONNX with :func:`yobx.torch.interpreter.to_onnx`.

----

How to choose how the FX graph is produced
------------------------------------------

:class:`~yobx.torch.export_options.ExportOptions` exposes three practical
ways to obtain the FX graph before ONNX conversion.

**1) torch.export.export path (default)**

Use ``strategy="nostrict"`` (default) or ``strategy="strict"``:

.. code-block:: python

    from yobx.torch.export_options import ExportOptions
    from yobx.torch.interpreter import to_onnx

    artifact = to_onnx(model, (x,), export_options=ExportOptions(strategy="nostrict"))

**2) Symbolic FX tracing (CustomTracer)**

Use ``strategy="tracing"``:

.. code-block:: python

    artifact = to_onnx(model, (x,), export_options=ExportOptions(strategy="tracing"))

**3) Dispatch-based tracing (new-tracing)**

Use ``strategy="new-tracing"``:

.. code-block:: python

    artifact = to_onnx(model, (x,), export_options=ExportOptions(strategy="new-tracing"))

Quick summary
^^^^^^^^^^^^^

The table below summarizes not only capture/backend routing, but also how
decomposition is handled and when each option is usually the most practical.

.. list-table::
    :header-rows: 1

    * - Option
      - FX graph capture
      - ONNX conversion backend
      - Decomposition behavior
      - Typical use
    * - ``strategy="nostrict"`` / ``"strict"``
      - :func:`torch.export.export`
      - yobx converter
      - no explicit decomposition table by default; may still decompose only if
        inplace nodes cannot be removed directly
      - preferred default when you want yobx ATen-level conversion coverage
    * - ``strategy="tracing"``
      - symbolic :class:`~yobx.torch.tracing.CustomTracer`
      - yobx converter
      - no explicit decomposition table by default; tracing normalizes many
        inplace patterns up front
      - useful when ``torch.export.export`` fails on a model but FX tracing works
    * - ``strategy="new-tracing"``
      - dispatch-based tracing
      - yobx converter
      - no explicit decomposition table by default; relies on new-tracing
        rewrites/patches for operator coverage
      - useful for models/operators better captured through dispatch-level tracing
    * - ``strategy="onnxscript"`` or ``tracing=TracingMode.ONNXSCRIPT``
      - handled by :func:`torch.onnx.export`
      - official exporter
      - decomposition behavior follows the official exporter pipeline
      - use when you want direct parity with :func:`torch.onnx.export`
    * - ``converting_library=ConvertingLibrary.ONNXSCRIPT`` (+ chosen strategy)
      - chosen strategy still captures the graph first
      - official exporter on that captured graph
      - decomposition depends on the selected capture strategy used before
        calling :func:`torch.onnx.export`
      - use when you want strategy-controlled capture plus official ONNX lowering

----

How to trigger the official exporter
------------------------------------

The official exporter path is :func:`torch.onnx.export`. You can trigger it
through :class:`~yobx.torch.export_options.ExportOptions`:

.. code-block:: python

    from yobx.torch.export_options import ExportOptions, TracingMode, ConvertingLibrary
    from yobx.torch.interpreter import to_onnx

    # direct torch.onnx.export(..., dynamo=True)
    a1 = to_onnx(model, (x,), export_options=ExportOptions(strategy="onnxscript"))
    a2 = to_onnx(model, (x,), export_options=ExportOptions(tracing=TracingMode.ONNXSCRIPT))

    # first capture with the selected strategy, then call torch.onnx.export
    a3 = to_onnx(
        model,
        (x,),
        export_options=ExportOptions(
            strategy="new-tracing",
            converting_library=ConvertingLibrary.ONNXSCRIPT,
        ),
    )

----

How to read export and optimization statistics
----------------------------------------------

The returned :class:`~yobx.container.ExportArtifact` carries an
:class:`~yobx.container.ExportReport` in ``artifact.report``.

.. code-block:: python

    artifact = to_onnx(model, (x,), return_optimize_report=True)
    print(artifact.report)
    print(artifact.report.stats[:2])      # per-pattern optimization rows
    print(artifact.report.extra.keys())   # timings, selected export options, counters

    artifact.compute_node_stats()
    print(artifact.report.node_stats[:2])  # per-op-type counts and estimated FLOPs

The ``extra`` dictionary typically includes timing keys such as
``time_export_graph_module`` and flags describing which export options won
the strategy fallback (for example ``onnx_export_options_strict`` or
``onnx_export_options_decomp``).

If ``filename="model.onnx"`` is passed, the converter also writes
``model.xlsx`` with sheets such as ``extra``, ``stats``, ``stats_agg``,
``node_stats``, and ``symbolic_flops``.

----

How this differs from torch.onnx.export
---------------------------------------

By default, :func:`yobx.torch.interpreter.to_onnx` does **not** apply an
explicit decomposition table, so ATen operators stay close to the original
graph and are translated directly by yobx converters. A decomposition pass may
still run if needed to eliminate inplace operations that could not be removed
directly.

The official exporter path is more decomposition-oriented. To align behaviors,
set decomposition options explicitly in yobx:

.. code-block:: python

    # run default decomposition table
    artifact_dec = to_onnx(model, (x,), export_options="nostrict-dec")

    # run full decomposition table
    artifact_decall = to_onnx(model, (x,), export_options="nostrict-decall")

When you need the official behavior directly, use one of the
``onnxscript`` options shown above.

.. seealso::

    :ref:`l-not-torch-onnx-export` for the full design-level comparison.
