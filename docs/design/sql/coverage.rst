.. _l-design-sql-coverage:

===========================
SQL / DataFrame Coverage
===========================

This page summarises which operations are supported across the three
input paths — direct SQL strings, pandas-inspired DataFrame tracing, and
polars ``LazyFrame`` — when converting tabular data manipulations to ONNX.

SQL string coverage
===================

The following SQL constructs are handled by :func:`~yobx.sql.sql_to_onnx`:

.. list-table::
    :header-rows: 1
    :widths: 30 20 50

    * - SQL construct
      - Status
      - Notes
    * - ``SELECT col``
      - ✔ supported
      - column pass-through via ``Identity``
    * - ``SELECT expr AS alias``
      - ✔ supported
      - arithmetic: ``Add``, ``Sub``, ``Mul``, ``Div``
    * - ``SELECT AGG(col)``
      - ✔ supported
      - ``SUM``, ``AVG``, ``MIN``, ``MAX``, ``COUNT``
    * - ``WHERE condition``
      - ✔ supported
      - comparisons + ``AND`` / ``OR``
    * - ``GROUP BY cols``
      - ⚠ partial
      - whole-dataset aggregation only; no per-group rows
    * - ``INNER / LEFT / RIGHT / FULL JOIN … ON col = col``
      - ✔ supported
      - equi-join on a single key column
    * - ``SELECT DISTINCT``
      - ✘ not supported
      - parsed but raises ``NotImplementedError``
    * - ``HAVING``
      - ✘ not supported
      - not yet implemented
    * - ``ORDER BY``
      - ✘ not supported
      - not yet implemented
    * - ``LIMIT``
      - ✘ not supported
      - not yet implemented
    * - Subqueries
      - ✘ not supported
      - not yet implemented
    * - String equality (``WHERE col = 'val'``)
      - ✘ not supported
      - requires special ONNX string handling

DataFrame tracer coverage
=========================

The following :class:`~yobx.xtracing.dataframe_trace.TracedDataFrame` and
:class:`~yobx.xtracing.dataframe_trace.TracedSeries` operations are handled by
:func:`~yobx.sql.dataframe_to_onnx`:

.. list-table::
    :header-rows: 1
    :widths: 30 20 50

    * - Operation
      - Status
      - Notes
    * - ``df["col"]``
      - ✔ supported
      - column access
    * - ``df.filter(condition)``
      - ✔ supported
      - maps to ``WHERE``
    * - ``df.select([series, …])``
      - ✔ supported
      - maps to ``SELECT``
    * - ``df.assign(name=series)``
      - ✔ supported
      - maps to ``SELECT … AS name``
    * - ``df.groupby(cols)``
      - ⚠ partial
      - whole-dataset aggregation only
    * - Series arithmetic (``+``, ``-``, ``*``, ``/``)
      - ✔ supported
      - ``Add``, ``Sub``, ``Mul``, ``Div``
    * - Series comparisons (``>``, ``<``, ``>=``, ``<=``, ``==``, ``!=``)
      - ✔ supported
      - ``Greater``, ``Less``, ``Equal``, …
    * - ``series.sum()`` / ``.mean()`` / ``.min()`` / ``.max()`` / ``.count()``
      - ✔ supported
      - ``ReduceSum``, ``ReduceMean``, ``ReduceMin``, ``ReduceMax``
    * - ``cond1 & cond2`` / ``cond1 | cond2``
      - ✔ supported
      - ``And``, ``Or``
    * - ``series.alias("name")``
      - ✔ supported
      - output rename
    * - Conditional branches (``if``/``else``)
      - ✘ not supported
      - tracing captures one execution path only

Polars LazyFrame coverage
=========================

The following polars operations are handled by :func:`~yobx.sql.lazyframe_to_onnx`:

.. list-table::
    :header-rows: 1
    :widths: 30 20 50

    * - Polars operation
      - Status
      - Notes
    * - ``lf.select([…])``
      - ✔ supported
      - column selection and arithmetic expressions
    * - ``lf.filter(condition)``
      - ✔ supported
      - comparison and boolean predicates
    * - ``lf.group_by(cols).agg([…])``
      - ⚠ partial
      - whole-dataset aggregation only
    * - Arithmetic (``+``, ``-``, ``*``, ``/``)
      - ✔ supported
      - inlined into ``SELECT`` expressions
    * - Comparisons (``>``, ``<``, ``>=``, ``<=``, ``==``, ``!=``)
      - ✔ supported
      - ``WHERE`` predicates
    * - ``&`` / ``|`` compound predicates
      - ✔ supported
      - ``AND`` / ``OR`` in ``WHERE``
    * - ``.alias("name")``
      - ✔ supported
      - output rename
    * - Aggregation methods (``.sum()``, ``.mean()``, ``.min()``, ``.max()``, ``.count()``)
      - ✔ supported
      - ``ReduceSum``, ``ReduceMean``, ``ReduceMin``, ``ReduceMax``
    * - ``lf.join(…)``
      - ✘ not supported
      - not yet implemented
    * - ``lf.sort(…)``
      - ✘ not supported
      - not yet implemented
    * - ``lf.limit(…)`` / ``lf.head(…)``
      - ✘ not supported
      - not yet implemented
    * - ``lf.unique(…)``
      - ✘ not supported
      - not yet implemented

Further reading
===============

* :ref:`l-design-sql-converter` — SQL parser and emission details.
* :ref:`l-design-sql-dataframe` — DataFrame function tracer design.
* :ref:`l-design-sql-polars` — Polars LazyFrame conversion design.
