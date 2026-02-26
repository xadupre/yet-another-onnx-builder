.. _l-cube:

=========================================
Statistics on Logs Through a Cube of Data
=========================================

:class:`CubeLogs <yobx.helpers.cube_helper.CubeLogs>` is a lightweight
data-analysis layer built on top of :mod:`pandas`.  It treats experiment
logs as a *cube* — a structured table whose columns play one of four roles:

* **time** — a single date/timestamp column (default ``"date"``).  It is
  optional but strongly recommended.
* **keys** — categorical identifiers (model name, exporter, hardware device, …).
  Together with *time*, the key tuple must uniquely identify every row.
* **values** — numerical or string measurements that can be aggregated
  (latency, memory, error counts, …).
* **ignored** — columns explicitly excluded from all three roles above.

Everything else is silently dropped after loading.

:class:`CubeLogsPerformance <yobx.helpers.cube_helper.CubeLogsPerformance>`
is a ready-made subclass whose defaults match the conventions used by
benchmarking scripts in this project (columns prefixed ``time_``, ``disc_``,
``onnx_``, etc.).

Loading data
============

Pass any of the following to the constructor, then call
:meth:`load <yobx.helpers.cube_helper.CubeLogs.load>`:

* a :class:`pandas.DataFrame`
* a list of :class:`pandas.DataFrame` objects (concatenated automatically)
* a list of dicts (each dict becomes one row)
* a list of file paths / directory paths to CSV files

.. runpython::
    :showcode:

    import io
    import textwrap
    import pandas
    from yobx.helpers.cube_helper import CubeLogs

    raw = pandas.read_csv(
        io.StringIO(
            textwrap.dedent(
                """
                date,version_python,model_name,model_exporter,time_latency,time_baseline
                2025/01/01,3.13,phi3,export,0.10,0.10
                2025/01/02,3.13,phi3,export,0.11,0.10
                2025/01/01,3.13,phi4,export,0.10,0.105
                2025/01/01,3.12,phi4,onnx-dynamo,0.14,0.999
                """
            )
        )
    )

    cube = CubeLogs(
        raw,
        time="date",
        keys=["version_.*", "model_.*"],
        values=["time_.*"],
        recent=True,      # keep only the most recent row per key tuple
    ).load()

    print("shape :", cube.shape)
    print("time  :", cube.time)
    print("keys  :", cube.keys_no_time)
    print("values:", cube.values)

When ``recent=True`` the cube keeps only the most recent row (highest *time*
value) for each combination of key columns.  This is useful when experiment
logs are appended over time and you only care about the latest run.

Column patterns
---------------

The ``keys``, ``values``, and ``ignored`` arguments accept plain column names
**or** Python regular expressions.  A string is treated as a regular
expression when it contains any of the characters ``"^.*+{}"``::

    CubeLogs(df, keys=["^version_.*", "model_name", "device"], ...)

Adding computed columns (formulas)
====================================

Pass a dictionary of ``{new_column_name: callable}`` as the ``formulas``
argument.  Each callable receives the full :class:`~pandas.DataFrame` and
must return a :class:`~pandas.Series`.  Computed columns are appended to the
*values* list.

.. runpython::
    :showcode:

    import io
    import textwrap
    import pandas
    from yobx.helpers.cube_helper import CubeLogs

    raw = pandas.read_csv(
        io.StringIO(
            textwrap.dedent(
                """
                date,version_python,model_name,model_exporter,time_latency,time_baseline
                2025/01/01,3.13,phi3,export,0.10,0.12
                2025/01/01,3.13,phi4,export,0.10,0.105
                2025/01/01,3.12,phi4,onnx-dynamo,0.14,0.999
                """
            )
        )
    )

    cube = CubeLogs(
        raw,
        time="date",
        keys=["version_.*", "model_.*"],
        values=["time_.*"],
        formulas={"speedup": lambda df: df["time_baseline"] / df["time_latency"]},
    ).load()

    print("values:", cube.values)
    print(cube.data[["model_name", "model_exporter", "speedup"]])

Pivot views
===========

:meth:`CubeLogs.view <yobx.helpers.cube_helper.CubeLogs.view>` creates a
pivot table from the cube.  Rows and columns of the pivot are controlled by a
:class:`CubeViewDef <yobx.helpers.cube_helper.CubeViewDef>` object.

.. runpython::
    :showcode:

    import io
    import textwrap
    import pandas
    from yobx.helpers.cube_helper import CubeLogs, CubeViewDef

    raw = pandas.read_csv(
        io.StringIO(
            textwrap.dedent(
                """
                date,version_python,model_name,model_exporter,time_latency,time_baseline
                2025/01/01,3.13,phi3,export,0.10,0.12
                2025/01/01,3.13,phi4,export,0.10,0.105
                2025/01/01,3.12,phi4,onnx-dynamo,0.14,0.999
                """
            )
        )
    )

    cube = CubeLogs(
        raw,
        time="date",
        keys=["version_.*", "model_.*"],
        values=["time_latency", "time_baseline"],
    ).load()

    view = cube.view(
        CubeViewDef(
            key_index=["version_python", "model_name"],
            values=["time_latency", "time_baseline"],
            ignore_columns=["date"],
        )
    )
    print(view.to_string())

The result is a :class:`~pandas.DataFrame` with a :class:`~pandas.MultiIndex`
on both the row index (``key_index``) and the columns
(metrics × remaining key values).

CubeViewDef options
--------------------

+---------------------------+------------------------------------------------------------+
| Parameter                 | Purpose                                                    |
+===========================+============================================================+
| ``key_index``             | Key columns placed in the row index of the pivot table     |
+---------------------------+------------------------------------------------------------+
| ``values``                | Metric columns to include in the pivot                     |
+---------------------------+------------------------------------------------------------+
| ``ignore_unique``         | Drop key columns that have only one distinct value         |
+---------------------------+------------------------------------------------------------+
| ``key_agg``               | Keys to aggregate away before pivoting                     |
+---------------------------+------------------------------------------------------------+
| ``agg_args``              | Aggregation function(s) passed to ``groupby(...).agg()``   |
|                           | (can be a callable ``column_name → agg_func``)             |
+---------------------------+------------------------------------------------------------+
| ``agg_multi``             | Extra aggregations over multiple columns simultaneously    |
+---------------------------+------------------------------------------------------------+
| ``order``                 | Explicit ordering of column levels                         |
+---------------------------+------------------------------------------------------------+
| ``ignore_columns``        | Columns to exclude from the view                           |
+---------------------------+------------------------------------------------------------+
| ``keep_columns_in_index`` | Keep a column even if it has only one distinct value       |
+---------------------------+------------------------------------------------------------+
| ``transpose``             | Transpose rows and columns                                 |
+---------------------------+------------------------------------------------------------+
| ``plots``                 | Attach a :class:`CubePlot` to this view in Excel export    |
+---------------------------+------------------------------------------------------------+
| ``no_index``              | Reset the index, returning a flat DataFrame                |
+---------------------------+------------------------------------------------------------+

Aggregated views
----------------

Set ``key_agg`` to collapse one or more key dimensions before building the
pivot.  This is useful for summarising across all models or all dates:

.. code-block:: python

    view = cube.view(
        CubeViewDef(
            key_index=["version_python"],
            values=["time_latency"],
            key_agg=["model_name", "date"],
            agg_args=lambda col: "mean",
            name="aggregated",
        )
    )

Describing the cube
===================

:meth:`CubeLogs.describe <yobx.helpers.cube_helper.CubeLogs.describe>` returns
a summary :class:`~pandas.DataFrame` with one row per column, showing its
role (time / keys / values / ignored), dtype, missing-value count, and basic
statistics:

.. runpython::
    :showcode:

    import io
    import textwrap
    import pandas
    from yobx.helpers.cube_helper import CubeLogs

    raw = pandas.read_csv(
        io.StringIO(
            textwrap.dedent(
                """
                date,version_python,model_name,model_exporter,time_latency,time_baseline
                2025/01/01,3.13,phi3,export,0.10,0.12
                2025/01/01,3.13,phi4,export,0.10,0.105
                2025/01/01,3.12,phi4,onnx-dynamo,0.14,0.999
                """
            )
        )
    )

    cube = CubeLogs(
        raw,
        time="date",
        keys=["version_.*", "model_.*"],
        values=["time_.*"],
    ).load()

    print(cube.describe()[["kind", "dtype", "missing", "min", "max"]].to_string())

Exporting to Excel
==================

:meth:`CubeLogs.to_excel <yobx.helpers.cube_helper.CubeLogs.to_excel>` writes
an ``.xlsx`` workbook.  Each view is placed on its own sheet.  A *raw* sheet
with the full cube data and a *main* sheet with per-column statistics can be
added automatically.

.. code-block:: python

    cube.to_excel(
        "results.xlsx",
        views={
            "latency": CubeViewDef(
                key_index=["version_python", "model_name"],
                values=["time_latency"],
                ignore_columns=["date"],
                name="latency",
                plots=True,
            ),
            "baseline": "time_baseline",   # shorthand: auto-generated view
        },
        main="main",    # sheet with per-column statistics
        raw="raw",      # sheet with the complete cube data
    )

Passing a plain string as a view is a shorthand: the cube calls
:meth:`make_view_def <yobx.helpers.cube_helper.CubeLogs.make_view_def>` to
produce a sensible default view for that metric.

CubeLogsPerformance
===================

:class:`CubeLogsPerformance <yobx.helpers.cube_helper.CubeLogsPerformance>`
is a subclass of :class:`CubeLogs <yobx.helpers.cube_helper.CubeLogs>` with
defaults tuned for ML benchmarking logs.  Its constructor pre-configures:

* **time** column: ``"DATE"`` (uppercase; contrast with the ``"date"``
  default in the base :class:`CubeLogs <yobx.helpers.cube_helper.CubeLogs>`)
* **key** patterns: ``version_.*``, ``model_.*``, ``device``, ``exporter``,
  ``suite``, ``machine``, ``dtype``, ``architecture``, and several others.
* **value** patterns: ``time_.*``, ``disc.*``, ``ERR_.*``, ``onnx_.*``, and
  related prefixes.
* **formulas**: ``speedup``, ``ERR1``, ``n_models``, ``n_model_faster``, a
  full suite of ``n_node_*`` counts, and more — all computed automatically
  from the raw columns.
* ``recent=True``: only the most recent row per key tuple is kept.

Usage is identical to :class:`CubeLogs <yobx.helpers.cube_helper.CubeLogs>`:

.. code-block:: python

    from yobx.helpers.cube_helper import CubeLogsPerformance

    cube = CubeLogsPerformance(df).load()
    view = cube.view(CubeViewDef(...))

.. seealso::

    :class:`CubeLogs API <yobx.helpers.cube_helper.CubeLogs>` —
    full API reference generated from docstrings.

    :class:`CubeLogsPerformance API <yobx.helpers.cube_helper.CubeLogsPerformance>` —
    subclass pre-configured for ML performance logs.
