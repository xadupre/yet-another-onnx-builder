.. _l-design-sql:

=================================
SQL-to-ONNX Converter Design
=================================

Overview
========

``yobx.sql`` converts a SQL query string into a self-contained
:class:`onnx.ModelProto`.

The primary design principle is:

  **Every column referenced in the query is a distinct 1-D ONNX input tensor.**

This mirrors the columnar representation used in tabular data pipelines: the
caller feeds one vector per column rather than a single 2-D matrix.  The ONNX
model can therefore be applied to any subset of rows without copying the full
table.

Architecture
============

The conversion pipeline consists of two stages:

1. **Parsing** (:func:`~yobx.sql.parse.parse_sql`) — turns the SQL string
   into a :class:`~yobx.sql.parse.ParsedQuery` containing an ordered list of
   :class:`~yobx.sql.parse.SqlOperation` objects, one per SQL clause.

2. **Emission** (:func:`~yobx.sql.sql_to_onnx`) — iterates over the
   operations in execution order and appends ONNX nodes to a
   :class:`~yobx.xbuilder.GraphBuilder`.

.. code-block:: text

    SQL string
        │
        ▼
    parse_sql()         ─── ParsedQuery ──► operations list
                                              │
                                              ▼
    sql_to_onnx()       ─── GraphBuilder ──► ModelProto

Supported SQL clauses
=====================

.. list-table::
    :header-rows: 1

    * - Clause
      - SqlOperation
      - ONNX nodes emitted
    * - ``SELECT expr [AS alias], …``
      - :class:`~yobx.sql.parse.SelectOp`
      - ``Identity``, ``Add``, ``Sub``, ``Mul``, ``Div``,
        ``ReduceSum``, ``ReduceMean``, ``ReduceMin``, ``ReduceMax``
    * - ``WHERE condition``
      - :class:`~yobx.sql.parse.FilterOp`
      - ``Compress``, ``Equal``, ``Less``, ``Greater``,
        ``LessOrEqual``, ``GreaterOrEqual``, ``And``, ``Or``, ``Not``
    * - ``GROUP BY col, …``
      - :class:`~yobx.sql.parse.GroupByOp`
      - (groups are processed together with ``SelectOp`` aggregations)
    * - ``[INNER|LEFT|RIGHT|FULL] JOIN … ON col = col``
      - :class:`~yobx.sql.parse.JoinOp`
      - ``Unsqueeze``, ``Equal``, ``ArgMax``, ``ReduceMax``,
        ``Compress``, ``Gather``

Columnar input convention
=========================

Each column becomes a separate ONNX graph input whose name matches the
column name in the SQL query.  Column names are normalised to lower-case.

.. code-block:: python

    import numpy as np
    from yobx.sql import sql_to_onnx

    # Columns "a" and "b" → two separate ONNX inputs
    dtypes = {"a": np.float32, "b": np.float32}
    onx = sql_to_onnx(
        "SELECT a + b AS total FROM t WHERE a > 0",
        dtypes,
    )

    # Inputs of the ONNX model
    for inp in onx.graph.input:
        print(inp.name)   # → "a", then "b"

Execution order
===============

Operations are applied in the following logical order:

1. **JoinOp** — merge left and right tables using an equi-join key.
2. **FilterOp** — apply the ``WHERE`` predicate as a boolean mask
   (``Compress``) to all column tensors simultaneously.
3. **GroupByOp** — record the group keys; referenced by aggregation
   expressions in the ``SelectOp``.
4. **SelectOp** — compute output expressions (arithmetic, aggregations)
   over the filtered/joined columns.

.. code-block:: text

    inputs: col_a, col_b, col_key_left, col_key_right
        │
        ├── JoinOp:   Unsqueeze/Equal/ArgMax → Compress/Gather aligned columns
        │
        ├── FilterOp: Greater/And/… → Compress (row mask applied to all cols)
        │
        ├── GroupByOp: (records group columns, used by aggregations)
        │
        └── SelectOp: Add/ReduceSum/… → Identity → outputs

Parser design
=============

The parser (:mod:`yobx.sql.parse`) is a hand-written recursive-descent
parser using a single-pass tokeniser.  No third-party SQL library is
required.

Tokens
------

The tokeniser (:func:`yobx.sql.parse._tokenize`) classifies input characters
into four token kinds:

* ``"num"`` — integer or floating-point literals.
* ``"str"`` — single- or double-quoted string literals.
* ``"op"`` — operators and punctuation (``+``, ``-``, ``*``, ``/``,
  ``=``, ``<``, ``>``, ``<=``, ``>=``, ``<>``, ``!=``, ``(``, ``)``,
  ``,``).
* ``"id"`` — identifiers (SQL keywords and column/table names).

All identifiers are lowercased so that parsing is case-insensitive.

Expression grammar
------------------

.. code-block:: text

    primary  ::= agg_func "(" expr ")"
               | "(" expr ")"
               | number
               | string
               | identifier

    multiplicative ::= primary ( ("*" | "/") primary )*
    additive       ::= multiplicative ( ("+" | "-") multiplicative )*
    expr           ::= additive

    comparison ::= expr ( "=" | "<" | ">" | "<=" | ">=" | "<>" | "!=" ) expr
    and_pred   ::= comparison ( "AND" comparison )*
    condition  ::= and_pred ( "OR" and_pred )*

Limitations and future work
============================

* ``GROUP BY`` uses a whole-dataset aggregation (``ReduceSum`` etc.) rather
  than true per-group aggregation.  True per-group semantics require an ONNX
  ``Loop`` or a custom kernel.
* ``SELECT DISTINCT`` is parsed but raises :class:`NotImplementedError` during conversion.
* Only equi-joins on a single key column are supported for ``JOIN``.
* ``HAVING``, ``ORDER BY``, ``LIMIT``, and subqueries are not yet supported.
* String equality (``WHERE name = 'alice'``) is not yet supported
  (string literals are parsed but ONNX ``Equal`` on strings may need a
  separate handling path).

Example
=======

.. runpython::
    :showcode:

    import numpy as np
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sql import sql_to_onnx, parse_sql
    from yobx.reference import ExtendedReferenceEvaluator

    # ── parse ────────────────────────────────────────────────────────
    pq = parse_sql("SELECT a + b AS total FROM t WHERE a > 0")
    for op in pq.operations:
        print(type(op).__name__, "—", op)

    # ── convert ──────────────────────────────────────────────────────
    dtypes = {"a": np.float32, "b": np.float32}
    onx = sql_to_onnx(
        "SELECT a + b AS total FROM t WHERE a > 0",
        dtypes,
    )
    print(pretty_onnx(onx))

    # ── run ──────────────────────────────────────────────────────────
    ref = ExtendedReferenceEvaluator(onx)
    a = np.array([ 1.0, -2.0,  3.0], dtype=np.float32)
    b = np.array([ 4.0,  5.0,  6.0], dtype=np.float32)
    (total,) = ref.run(None, {"a": a, "b": b})
    print(total)   # → [5.  9.]   (rows where a > 0)

DataFrame function tracer
=========================

In addition to accepting SQL strings and :class:`polars.LazyFrame` objects,
``yobx.sql`` provides a lightweight pandas-inspired API for tracing Python
functions that operate on a virtual DataFrame.

The tracer works by passing a :class:`~yobx.sql.dataframe_trace.TracedDataFrame`
proxy to the user function.  Every operation performed on the proxy —
column access, arithmetic, filtering, aggregation — is recorded as an AST
node rather than being executed.  The resulting AST is assembled into a
:class:`~yobx.sql.parse.ParsedQuery` which is then compiled to ONNX by the
existing SQL converter.

.. code-block:: text

    Python function (TracedDataFrame)
        │
        ▼
    trace_dataframe()   ─── ParsedQuery ──► operations list
                                                │
                                                ▼
    parsed_query_to_onnx() ─ GraphBuilder ──► ExportArtifact

Key classes and functions
-------------------------

* :class:`~yobx.sql.dataframe_trace.TracedDataFrame` — proxy DataFrame with
  ``.filter()``, ``.select()``, ``.assign()``, ``.groupby()`` operations.
* :class:`~yobx.sql.dataframe_trace.TracedSeries` — proxy for a column or
  expression; supports arithmetic (``+``, ``-``, ``*``, ``/``),
  comparisons (``>``, ``<``, ``>=``, ``<=``, ``==``, ``!=``),
  aggregations (``.sum()``, ``.mean()``, ``.min()``, ``.max()``, ``.count()``)
  and ``.alias()``.
* :class:`~yobx.sql.dataframe_trace.TracedCondition` — boolean predicate proxy;
  supports ``&`` (AND) and ``|`` (OR) combination.
* :func:`~yobx.sql.trace_dataframe` — traces a function and returns a
  :class:`~yobx.sql.parse.ParsedQuery`.
* :func:`~yobx.sql.dataframe_to_onnx` — end-to-end converter: function →
  :class:`~yobx.container.ExportArtifact`.
* :func:`~yobx.sql.parsed_query_to_onnx` — convert an already-built
  :class:`~yobx.sql.parse.ParsedQuery` to ONNX (bypasses SQL string parsing).

Example
-------

The following example traces the function and prints the list of captured
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

