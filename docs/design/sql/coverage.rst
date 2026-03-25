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

.. runpython::
    :rst:

    from yobx.sql.coverage import get_sql_coverage
    print(get_sql_coverage("sql"))

DataFrame tracer coverage
=========================

The following :class:`~yobx.xtracing.dataframe_trace.TracedDataFrame` and
:class:`~yobx.xtracing.dataframe_trace.TracedSeries` operations are handled by
:func:`~yobx.sql.dataframe_to_onnx`:

.. runpython::
    :rst:

    from yobx.sql.coverage import get_sql_coverage
    print(get_sql_coverage("dataframe"))

Polars LazyFrame coverage
=========================

The following polars operations are handled by :func:`~yobx.sql.lazyframe_to_onnx`:

.. runpython::
    :rst:

    from yobx.sql.coverage import get_sql_coverage
    print(get_sql_coverage("polars"))

Further reading
===============

* :ref:`l-design-sql-converter` — SQL parser and emission details.
* :ref:`l-design-sql-dataframe` — DataFrame function tracer design.
* :ref:`l-design-sql-polars` — Polars LazyFrame conversion design.
