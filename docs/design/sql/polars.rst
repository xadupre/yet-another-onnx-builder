.. _l-design-sql-polars:

================================
Polars LazyFrame to ONNX
================================

Overview
========

``yobx.sql`` can convert a :class:`polars.LazyFrame` execution plan directly
into a self-contained ONNX model.  The conversion works by extracting the
logical plan from the ``LazyFrame`` via :meth:`polars.LazyFrame.explain`,
translating that plan into an intermediate SQL query, and then delegating to
the SQL-to-ONNX pipeline (:func:`~yobx.sql.sql_to_onnx`).

Architecture
============

.. code-block:: text

    polars.LazyFrame
        │
        ▼
    lf.explain()        ─── execution plan string
        │
        ▼
    _parse_polars_plan()─── _PolarsPlan (select / filter / group_by)
        │
        ▼
    _plan_to_sql()      ─── SQL query string
        │
        ▼
    sql_to_onnx()       ─── GraphBuilder ──► ExportArtifact

Supported LazyFrame operations
==============================

.. list-table::
    :header-rows: 1

    * - Polars operation
      - SQL clause generated
      - ONNX nodes emitted
    * - ``select([col, expr, …])``
      - ``SELECT expr [AS alias], …``
      - ``Identity``, ``Add``, ``Sub``, ``Mul``, ``Div``
    * - ``filter(condition)``
      - ``WHERE condition``
      - ``Compress``, ``Equal``, ``Less``, ``Greater``, …
    * - ``group_by(cols).agg([…])``
      - ``SELECT agg … GROUP BY cols``
      - ``ReduceSum``, ``ReduceMean``, ``ReduceMin``, ``ReduceMax``
    * - Arithmetic (``+``, ``-``, ``*``, ``/``)
      - Inlined into ``SELECT`` expressions
      - ``Add``, ``Sub``, ``Mul``, ``Div``
    * - Comparisons (``>``, ``<``, ``>=``, ``<=``, ``==``, ``!=``)
      - ``WHERE`` predicates
      - ``Greater``, ``Less``, ``Equal``, …
    * - Boolean compound (``&``, ``|``)
      - ``AND`` / ``OR`` in ``WHERE``
      - ``And``, ``Or``
    * - ``.alias("name")``
      - ``… AS name`` in ``SELECT``
      - (rename only)
    * - Aggregation methods (``.sum()``, ``.mean()``, ``.min()``, ``.max()``, ``.count()``)
      - ``SUM(…)``, ``AVG(…)``, ``MIN(…)``, ``MAX(…)``, ``COUNT(…)``
      - ``ReduceSum``, ``ReduceMean``, ``ReduceMin``, ``ReduceMax``

Columnar input convention
=========================

As with the SQL converter, each source column of the plan is represented as a
separate 1-D ONNX input tensor.  The ``input_dtypes`` parameter maps source
column names to numpy dtypes and must include every column that appears in the
plan.

Polars dtype mapping
--------------------

The following polars data types are mapped to numpy equivalents:

.. list-table::
    :header-rows: 1

    * - Polars type
      - numpy dtype
    * - ``pl.Float32``
      - ``float32``
    * - ``pl.Float64``
      - ``float64``
    * - ``pl.Int8`` / ``pl.Int16`` / ``pl.Int32`` / ``pl.Int64``
      - ``int8`` / ``int16`` / ``int32`` / ``int64``
    * - ``pl.UInt8`` / ``pl.UInt16`` / ``pl.UInt32`` / ``pl.UInt64``
      - ``uint8`` / ``uint16`` / ``uint32`` / ``uint64``
    * - ``pl.Boolean``
      - ``bool``
    * - ``pl.String`` / ``pl.Utf8``
      - ``object``

Example
=======

.. code-block:: python

    import numpy as np
    import polars as pl
    from yobx.sql import lazyframe_to_onnx
    from yobx.reference import ExtendedReferenceEvaluator

    lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    lf = lf.filter(pl.col("a") > 0).select(
        [(pl.col("a") + pl.col("b")).alias("total")]
    )

    dtypes = {"a": np.float64, "b": np.float64}
    artifact = lazyframe_to_onnx(lf, dtypes)

    ref = ExtendedReferenceEvaluator(artifact)
    a = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    b = np.array([4.0,  5.0, 6.0], dtype=np.float64)
    (total,) = ref.run(None, {"a": a, "b": b})
    # total contains rows where a > 0: [5.0, 9.0]

Limitations
===========

* ``GROUP BY`` on multiple columns casts the key columns to ``float64`` before
  combining them, which causes precision loss for integer keys greater than 2**53.
* Only a single ``filter`` step, a single ``select`` step, and a single
  ``group_by``/``agg`` step are handled.  Complex multi-step plans may
  not translate correctly.
* ``join``, ``sort``, ``limit``, ``distinct``, ``pivot``, ``melt``, and
  other advanced polars operations are not yet supported.
* The plan text produced by :meth:`polars.LazyFrame.explain` may change
  between polars versions; the parser targets the format used by polars ≥ 0.19.
