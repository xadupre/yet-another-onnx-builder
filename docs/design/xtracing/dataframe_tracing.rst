.. _l-design-dataframe-tracing:

=================
DataFrame Tracing
=================

``yobx`` can export a Python function that operates on a
:class:`~yobx.xtracing.TracedDataFrame` to ONNX via
:func:`~yobx.sql.dataframe_to_onnx`.  Instead of running the function on real
data, the framework passes lightweight *proxy* objects that *record* each
DataFrame operation — filtering, column arithmetic, grouping, joining — as an
AST node.  Once recording is complete the accumulated AST is compiled to ONNX
by the same SQL-to-ONNX backend that powers the plain-SQL converter.

Overview
========

The mechanism is built on three proxy classes and two driver functions:

1. :class:`~yobx.xtracing.TracedDataFrame` — the main proxy.  It exposes a
   pandas-inspired API (``filter``, ``select``, ``assign``, ``groupby``,
   ``join``, ``pivot_table``, …).  Calling any of these methods does *not*
   execute the operation; it appends the corresponding AST node and returns a
   new :class:`~yobx.xtracing.TracedDataFrame`.

2. :class:`~yobx.xtracing.dataframe_trace.TracedSeries` — represents a single
   column expression (a :class:`~yobx.xtracing.parse.ColumnRef` or a derived
   :class:`~yobx.xtracing.parse.BinaryExpr`).  Arithmetic operators (``+``,
   ``-``, ``*``, ``/``) and comparisons (``>``, ``<``, ``==``, …) return new
   :class:`~yobx.xtracing.dataframe_trace.TracedSeries` or
   :class:`~yobx.xtracing.dataframe_trace.TracedCondition` objects.  The
   aggregation methods (``.sum()``, ``.mean()``, ``.min()``, ``.max()``,
   ``.count()``) wrap the expression in an
   :class:`~yobx.xtracing.parse.AggExpr`.

3. :class:`~yobx.xtracing.dataframe_trace.TracedCondition` — a thin wrapper
   around a :class:`~yobx.xtracing.parse.Condition` AST node.  Boolean
   operators ``&`` (AND) and ``|`` (OR) compose conditions.

**Driver functions**

* :func:`~yobx.xtracing.dataframe_trace.trace_dataframe` — low-level driver.
  Builds the proxy frame(s), calls *func*, and returns the recorded
  :class:`~yobx.sql.parse.ParsedQuery` (or a list of them when the function
  returns multiple frames).

* :func:`~yobx.sql.dataframe_to_onnx` — high-level entry point.  Calls
  :func:`~yobx.xtracing.dataframe_trace.trace_dataframe` and then
  :func:`~yobx.sql.sql_convert.parsed_query_to_onnx` to produce a
  self-contained :class:`~yobx.container.ExportArtifact`.

High-level entry point
======================

:func:`~yobx.sql.dataframe_to_onnx` is the recommended way to convert a
DataFrame function to ONNX:

.. code-block:: python

    def dataframe_to_onnx(
        func: Callable,
        input_dtypes: Union[Dict[str, dtype], List[Dict[str, dtype]]],
        target_opset: int = DEFAULT_TARGET_OPSET,
        custom_functions: Optional[Dict[str, Callable]] = None,
        builder_cls: type = GraphBuilder,
        filename: Optional[str] = None,
        verbose: int = 0,
    ) -> ExportArtifact: ...

====================  ================================================================
Parameter             Description
====================  ================================================================
``func``              Python callable that accepts one or more
                      :class:`~yobx.xtracing.TracedDataFrame` objects and returns a
                      :class:`~yobx.xtracing.TracedDataFrame` or a tuple/list of
                      them.
``input_dtypes``      ``{column: dtype}`` dict (single input frame) **or** a list of
                      such dicts (one per input frame).  A
                      :class:`~pandas.DataFrame` can be passed directly; its column
                      names and dtypes are extracted automatically.
``target_opset``      ONNX opset version to target.
``custom_functions``  Optional dict mapping SQL function names to Python
                      callables for use inside ``select``-level expressions.
``builder_cls``       :class:`~yobx.xbuilder.GraphBuilder` subclass to use.
``filename``          Optional path to write the model file.
``verbose``           Verbosity level (0 = silent).
====================  ================================================================

Supported operations
====================

The table below summarises the :class:`~yobx.xtracing.TracedDataFrame`
operations and their SQL/ONNX equivalents.

==================================================  ======================================
Operation                                           SQL equivalent
==================================================  ======================================
``df["col"]``  /  ``df.col``                        Column reference
``df.filter(cond)``  /  ``df[cond]``                ``WHERE``
``df.select([…])``  /  ``df[[…]]``                  ``SELECT``
``df.assign(new_col=expr)``                         ``SELECT …, expr AS new_col``
``df.groupby("key").agg({…})``                      ``GROUP BY … + aggregation``
``df.join(right, left_key, right_key)``             ``JOIN``
``df.pivot_table(values, index, columns)``          ``PIVOT``
``df.pipe(func)``  /  ``df.pipe(func, *args)``      Function composition
``df.copy()``                                       Copy (no-op node)
``series + / - / * / series_or_scalar``             Arithmetic expression
``series > / < / >= / <= / == / != value``          Comparison condition
``cond1 & cond2``  /  ``cond1 | cond2``             ``AND`` / ``OR``
``series.sum()`` / ``.mean()`` / ``.min()``         Aggregation functions
``.max()`` / ``.count()``                           Aggregation functions
``series.alias("name")``                            Column alias
==================================================  ======================================

Walkthrough examples
====================

Filter and select
-----------------

.. runpython::
    :showcode:

    import numpy as np
    from yobx.sql import dataframe_to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    def transform(df):
        filtered = df.filter(df["a"] > 0)
        return filtered.select([(df["a"] + df["b"]).alias("total")])

    dtypes = {"a": np.float32, "b": np.float32}
    artifact = dataframe_to_onnx(transform, dtypes)
    print(pretty_onnx(artifact.proto))

Group-by aggregation
--------------------

The group-by key column must be listed explicitly in the ``agg`` output list
alongside the aggregated values:

.. runpython::
    :showcode:

    import numpy as np
    from yobx.sql import dataframe_to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    def transform(df):
        return df.groupby("key").agg(
            [df["key"].alias("key"), df["val"].sum().alias("total")]
        )

    dtypes = {"key": np.int64, "val": np.float32}
    artifact = dataframe_to_onnx(transform, dtypes)
    print(pretty_onnx(artifact.proto))

Join two frames
---------------

.. runpython::
    :showcode:

    import numpy as np
    from yobx.sql import dataframe_to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    def transform(df1, df2):
        joined = df1.join(df2, left_key="id", right_key="id")
        return joined.select([(df1["a"] + df2["b"]).alias("sum_ab")])

    dtypes1 = {"id": np.int64, "a": np.float32}
    dtypes2 = {"id": np.int64, "b": np.float32}
    artifact = dataframe_to_onnx(transform, [dtypes1, dtypes2])
    print(pretty_onnx(artifact.proto))

Multiple output frames
----------------------

When *func* returns a tuple or list of
:class:`~yobx.xtracing.TracedDataFrame` objects, all outputs are gathered
into a single ONNX graph with multiple output tensors:

.. runpython::
    :showcode:

    import numpy as np
    from yobx.sql import dataframe_to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    def transform(df):
        out1 = df.select([(df["a"] + df["b"]).alias("sum_ab")])
        out2 = df.select([(df["a"] - df["b"]).alias("diff_ab")])
        return out1, out2

    dtypes = {"a": np.float32, "b": np.float32}
    artifact = dataframe_to_onnx(transform, dtypes)
    print(pretty_onnx(artifact.proto))

Assign new columns
------------------

:meth:`~yobx.xtracing.TracedDataFrame.assign` adds computed columns while
keeping all existing ones, similar to ``pandas.DataFrame.assign``:

.. runpython::
    :showcode:

    import numpy as np
    from yobx.sql import dataframe_to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    def transform(df):
        df = df.assign(scaled=(df["a"] * 2.0).alias("scaled"))
        return df.select(["a", "b", "scaled"])

    dtypes = {"a": np.float32, "b": np.float32}
    artifact = dataframe_to_onnx(transform, dtypes)
    print(pretty_onnx(artifact.proto))

Low-level API
=============

:func:`~yobx.xtracing.dataframe_trace.trace_dataframe` is useful when you
want to inspect the recorded AST before compiling to ONNX:

.. runpython::
    :showcode:

    import numpy as np
    from yobx.xtracing.dataframe_trace import trace_dataframe

    def transform(df):
        df = df.filter(df["a"] > 0)
        return df.select([(df["a"] + df["b"]).alias("total")])

    pq = trace_dataframe(transform, {"a": np.float32, "b": np.float32})
    for op in pq.operations:
        print(type(op).__name__, "—", op)

The returned :class:`~yobx.sql.parse.ParsedQuery` can then be passed to
:func:`~yobx.sql.sql_convert.parsed_query_to_onnx` to build the ONNX graph.

Relation to the SQL converter
==============================

DataFrame tracing is a *front-end* for the same SQL-to-ONNX back-end used by
plain SQL strings and Polars ``LazyFrame`` inputs.  The
:class:`~yobx.xtracing.TracedDataFrame` API records operations as the same
:class:`~yobx.sql.parse.ParsedQuery` AST nodes that the SQL parser produces,
so the two paths share identical ONNX code generation.

The unified entry point :func:`yobx.sql.to_onnx` accepts all three flavours
and delegates automatically:

* a **string** → SQL parser → :func:`~yobx.sql.sql_convert.sql_to_onnx`
* a **Polars LazyFrame** → :func:`~yobx.sql.polars_convert.lazyframe_to_onnx`
* a **callable** → :func:`~yobx.sql.dataframe_to_onnx`

.. seealso::

    :ref:`l-design-function-transformer-tracing` — numpy tracing, used to
    convert :class:`~sklearn.preprocessing.FunctionTransformer` to ONNX.
