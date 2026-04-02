.. _l-design-sql-converter:

==========================
SQL-to-ONNX Converter
==========================

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

1. **Parsing** (:func:`~yobx.xtracing.parse.parse_sql`) — turns the SQL string
   into a :class:`~yobx.xtracing.parse.ParsedQuery` containing an ordered list of
   :class:`~yobx.xtracing.parse.SqlOperation` objects, one per SQL clause.

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
      - :class:`~yobx.xtracing.parse.SelectOp`
      - ``Identity``, ``Add``, ``Sub``, ``Mul``, ``Div``,
        ``ReduceSum``, ``ReduceMean``, ``ReduceMin``, ``ReduceMax``
    * - ``WHERE condition``
      - :class:`~yobx.xtracing.parse.FilterOp`
      - ``Compress``, ``Equal``, ``Less``, ``Greater``,
        ``LessOrEqual``, ``GreaterOrEqual``, ``And``, ``Or``, ``Not``
    * - ``GROUP BY col, …``
      - :class:`~yobx.xtracing.parse.GroupByOp`
      - ``Unique``, ``ScatterElements`` (groups are processed together
        with ``SelectOp`` aggregations)
    * - ``[INNER|LEFT|RIGHT|FULL] JOIN … ON col = col``
      - :class:`~yobx.xtracing.parse.JoinOp`
      - ``Unsqueeze``, ``Equal``, ``ArgMax``, ``ReduceMax``,
        ``Compress``, ``Gather``
    * - Subqueries (``SELECT … FROM (SELECT …)``)
      - inner :class:`~yobx.xtracing.parse.ParsedQuery`
      - (inner query outputs become outer query column tensors)

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

1. **Subquery** — if present, the inner query is processed first; its SELECT
   outputs become the column tensors for the outer query.
2. **JoinOp** — merge left and right tables using an equi-join key.
3. **FilterOp** — apply the ``WHERE`` predicate as a boolean mask
   (``Compress``) to all column tensors simultaneously.
4. **GroupByOp** — use ``Unique`` + ``ScatterElements`` to compute per-group
   aggregations; referenced by aggregation expressions in the ``SelectOp``.
5. **SelectOp** — compute output expressions (arithmetic, aggregations)
   over the filtered/joined columns.

.. code-block:: text

    inputs: col_a, col_b, col_key_left, col_key_right
        │
        ├── Subquery:  inner ParsedQuery → intermediate column tensors
        │
        ├── JoinOp:   Unsqueeze/Equal/ArgMax → Compress/Gather aligned columns
        │
        ├── FilterOp: Greater/And/… → Compress (row mask applied to all cols)
        │
        ├── GroupByOp: Unique → ScatterElements (per-group aggregations)
        │
        └── SelectOp: Add/ReduceSum/… → Identity → outputs

Parser design
=============

The parser (:mod:`yobx.xtracing.parse`) is a hand-written recursive-descent
parser using a single-pass tokeniser.  No third-party SQL library is
required.

Tokens
------

The tokeniser (:func:`yobx.xtracing.parse._tokenize`) classifies input characters
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

* ``GROUP BY`` on multiple columns casts the key columns to ``float64`` before
  combining them, which causes precision loss for integer keys greater than 2**53.
* ``SELECT DISTINCT`` is parsed but raises :class:`NotImplementedError` during conversion.
* Only equi-joins on a single key column are supported for ``JOIN``.
* ``HAVING``, ``ORDER BY``, and ``LIMIT`` are not yet supported.
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
