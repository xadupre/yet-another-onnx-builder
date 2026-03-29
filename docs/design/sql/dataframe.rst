.. _l-design-sql-dataframe:

==============================
DataFrame Function Tracer
==============================

Overview
========

In addition to accepting SQL strings and :class:`polars.LazyFrame` objects,
``yobx.sql`` provides a lightweight pandas-inspired API for tracing Python
functions that operate on a virtual DataFrame.

.. seealso::

    :ref:`l-plot-dataframe-to-onnx` — runnable end-to-end examples of the
    DataFrame tracing API.

The tracer works by passing a :class:`~yobx.xtracing.dataframe_trace.TracedDataFrame`
proxy to the user function.  Every operation performed on the proxy —
column access, arithmetic, filtering, aggregation — is recorded as an AST
node rather than being executed.  The resulting AST is assembled into a
:class:`~yobx.xtracing.parse.ParsedQuery` which is then compiled to ONNX by the
existing SQL converter.

Architecture
============

.. code-block:: text

    Python function (TracedDataFrame)
        │
        ▼
    trace_dataframe()   ─── ParsedQuery ──► operations list
                                                │
                                                ▼
    parsed_query_to_onnx() ─ GraphBuilder ──► ExportArtifact

Key classes and functions
=========================

* :class:`~yobx.xtracing.dataframe_trace.TracedDataFrame` — proxy DataFrame with
  ``.filter()``, ``.select()``, ``.assign()``, ``.groupby()`` operations.
* :class:`~yobx.xtracing.dataframe_trace.TracedSeries` — proxy for a column or
  expression; supports arithmetic (``+``, ``-``, ``*``, ``/``),
  comparisons (``>``, ``<``, ``>=``, ``<=``, ``==``, ``!=``),
  aggregations (``.sum()``, ``.mean()``, ``.min()``, ``.max()``, ``.count()``)
  and ``.alias()``.
* :class:`~yobx.xtracing.dataframe_trace.TracedCondition` — boolean predicate proxy;
  supports ``&`` (AND) and ``|`` (OR) combination.
* :func:`~yobx.sql.trace_dataframe` — traces a function and returns a
  :class:`~yobx.xtracing.parse.ParsedQuery`.
* :func:`~yobx.sql.dataframe_to_onnx` — end-to-end converter: function →
  :class:`~yobx.container.ExportArtifact`.
* :func:`~yobx.sql.parsed_query_to_onnx` — convert an already-built
  :class:`~yobx.xtracing.parse.ParsedQuery` to ONNX (bypasses SQL string parsing).

Tracing a function
==================

The following example traces a function and prints the list of captured
operations before compiling to ONNX:

.. runpython::
    :showcode:

    import numpy as np
    from yobx.sql import trace_dataframe

    def transform(df):
        df = df.filter(df["a"] > 0)
        return df.select([(df["a"] + df["b"]).alias("total")])

    pq = trace_dataframe(transform, {"a": np.float32, "b": np.float32})
    for op in pq.operations:
        print(type(op).__name__, "—", op)

End-to-end conversion
=====================

The :func:`~yobx.sql.dataframe_to_onnx` function combines tracing and ONNX
emission in a single call:

.. runpython::
    :showcode:

    import numpy as np
    from yobx.sql import dataframe_to_onnx
    from yobx.reference import ExtendedReferenceEvaluator

    def transform(df):
        df = df.filter(df["a"] > 0)
        return df.select([(df["a"] + df["b"]).alias("total")])

    dtypes = {"a": np.float32, "b": np.float32}
    artifact = dataframe_to_onnx(transform, dtypes)

    ref = ExtendedReferenceEvaluator(artifact)
    a = np.array([ 1.0, -2.0,  3.0], dtype=np.float32)
    b = np.array([ 4.0,  5.0,  6.0], dtype=np.float32)
    (total,) = ref.run(None, {"a": a, "b": b})
    print(total)   # → [5.  9.]   (rows where a > 0)

Supported operations
====================

The following pandas-inspired operations can be traced:

.. list-table::
    :header-rows: 1

    * - Operation
      - TracedDataFrame / TracedSeries API
      - ONNX nodes emitted
    * - Row filter
      - ``df.filter(condition)``
      - ``Compress``, ``Equal``, ``Less``, ``Greater``, …
    * - Column selection
      - ``df.select([series, …])``
      - ``Identity``, ``Add``, ``Sub``, ``Mul``, ``Div``
    * - Column addition
      - ``df.assign(name=series)``
      - ``Add``, ``Sub``, ``Mul``, ``Div``
    * - Aggregation
      - ``.sum()``, ``.mean()``, ``.min()``, ``.max()``, ``.count()``
      - ``ReduceSum``, ``ReduceMean``, ``ReduceMin``, ``ReduceMax``
    * - Group by
      - ``df.groupby(cols)``
      - (aggregation expressions in ``select``)
    * - Boolean AND / OR
      - ``cond1 & cond2``, ``cond1 | cond2``
      - ``And``, ``Or``

Limitations
===========

* Tracing captures only the operations performed during the single forward
  pass through the function.  Conditional branches (``if``/``else``) are
  not supported.
* ``GROUP BY`` uses whole-dataset aggregation, not true per-group
  semantics (same limitation as the SQL converter).
* The ``TracedDataFrame`` API is a subset of the full pandas/polars API;
  operations outside this subset will raise :class:`NotImplementedError`.
