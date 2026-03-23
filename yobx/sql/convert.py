"""
Unified SQL / polars LazyFrame / DataFrame-function → ONNX dispatcher.

:func:`to_onnx` is the single entry point that accepts a plain SQL query
string, a :class:`polars.LazyFrame`, or a Python callable that operates on a
:class:`~yobx.sql.dataframe_trace.TracedDataFrame`, and returns a
self-contained :class:`~yobx.container.ExportArtifact`.  Internally it
delegates to :func:`~yobx.sql.sql_convert.sql_to_onnx`,
:func:`~yobx.sql.polars_convert.lazyframe_to_onnx`, or
:func:`~yobx.sql.dataframe_trace.dataframe_to_onnx` depending on the type of
the first argument.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import numpy as np

from .. import DEFAULT_TARGET_OPSET
from ..container import ExportArtifact
from ..xbuilder import GraphBuilder
from .dataframe_trace import TracedDataFrame, dataframe_to_onnx
from .polars_convert import lazyframe_to_onnx
from .sql_convert import sql_to_onnx, sql_to_onnx_graph  # noqa: F401 – re-exported


def to_onnx(
    dataframe_or_query: Union[  # type: ignore
        str,
        Callable[["TracedDataFrame"], "TracedDataFrame"],  # type: ignore
        "polars.LazyFrame",  # type: ignore # noqa: F821
    ],
    input_dtypes: Union[
        Dict[str, Union[np.dtype, type, str]], List[Dict[str, Union[np.dtype, type, str]]]
    ],
    right_input_dtypes: Optional[Dict[str, Union[np.dtype, type, str]]] = None,
    target_opset: int = DEFAULT_TARGET_OPSET,
    custom_functions: Optional[Dict[str, Callable]] = None,
    builder_cls: Union[type, Callable] = GraphBuilder,
    filename: Optional[str] = None,
    verbose: int = 0,
) -> ExportArtifact:
    """Convert a SQL string, a DataFrame-tracing function, or a polars LazyFrame to ONNX.

    This is the unified entry point that dispatches to:

    * :func:`sql_to_onnx` — when *dataframe_or_query* is a **string**.
    * :func:`~yobx.sql.dataframe_trace.dataframe_to_onnx` — when
      *dataframe_or_query* is a **callable** (a Python function that accepts a
      :class:`~yobx.sql.dataframe_trace.TracedDataFrame` and returns one).
    * :func:`lazyframe_to_onnx` — for any other value (expected to be a
      :class:`polars.LazyFrame`).

    Each *source* column is represented as a **separate 1-D ONNX input**
    tensor.  The model outputs correspond to the ``SELECT`` expressions (SQL /
    callable) or the ``select`` / ``agg`` step of the LazyFrame plan.

    :param dataframe_or_query: one of:

        * **SQL string** — supported clauses: ``SELECT``, ``FROM``,
          ``[INNER|LEFT|RIGHT|FULL] JOIN … ON``, ``WHERE``, ``GROUP BY``.
          Custom Python functions can be called by name in the ``SELECT`` and
          ``WHERE`` clauses when registered via *custom_functions*.
        * **callable** — a Python function ``(df: TracedDataFrame) ->
          TracedDataFrame`` or a function that accepts *multiple*
          :class:`~yobx.sql.dataframe_trace.TracedDataFrame` arguments.  The
          function is traced to capture all ``filter``, ``select``,
          aggregation, and ``join`` operations it performs, which are then
          compiled to ONNX.
        * **polars.LazyFrame** — the execution plan is extracted via
          :meth:`polars.LazyFrame.explain` and translated into SQL before
          conversion.  See :func:`lazyframe_to_onnx` for details of supported
          operations.

    :param input_dtypes: either a single ``{column: dtype}`` mapping or,
        when *dataframe_or_query* is a callable that accepts *multiple*
        :class:`~yobx.sql.dataframe_trace.TracedDataFrame` arguments, a
        **list** of ``{column: dtype}`` mappings — one per argument.
        For SQL queries this maps *left-table* columns; for a ``LazyFrame``
        it maps the source DataFrame columns referenced in the plan.  Only
        columns that actually appear in the query / plan need to be listed.
    :param right_input_dtypes: if *dataframe_or_query* is a SQL string that
        contains a ``JOIN``, a mapping from *right-table* column name to numpy
        dtype.  Defaults to *input_dtypes* when ``None``.  Ignored when
        *dataframe_or_query* is a callable or a :class:`polars.LazyFrame`.
    :param target_opset: ONNX opset version to target (default:
        :data:`yobx.DEFAULT_TARGET_OPSET`).
    :param custom_functions: an optional mapping from function name (as it
        appears in the SQL string) to a Python callable.  Each callable must
        accept one or more numpy arrays and return a numpy array.  The
        function body is traced with :func:`~yobx.xtracing.trace_numpy_function`
        so that numpy arithmetic is translated into ONNX nodes.  Ignored when
        *dataframe_or_query* is a :class:`polars.LazyFrame`.

        Example::

            import numpy as np
            from yobx.sql import to_onnx

            artifact = to_onnx(
                "SELECT my_sqrt(a) AS r FROM t",
                {"a": np.float32},
                custom_functions={"my_sqrt": np.sqrt},
            )

    :param builder_cls: the graph-builder class (or factory callable) to use.
        Defaults to :class:`~yobx.xbuilder.GraphBuilder`.  Any class that
        implements the :ref:`builder-api` can be supplied here, e.g. a custom
        subclass that adds extra optimisation passes.
    :param filename: if set, the exported ONNX model is saved to this path and
        the :class:`~yobx.container.ExportReport` is written as a companion
        Excel file (same base name with ``.xlsx`` extension).
    :param verbose: verbosity level (0 = silent).
    :return: :class:`~yobx.container.ExportArtifact` wrapping the exported
        ONNX model together with an :class:`~yobx.container.ExportReport`.

    Example — from a SQL string::

        import numpy as np
        from yobx.sql import to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        dtypes = {"a": np.float32, "b": np.float32}
        artifact = to_onnx("SELECT a + b AS total FROM t WHERE a > 0", dtypes)

        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0,  5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        # total == array([5., 9.], dtype=float32)  (rows where a > 0)

    Example — from a DataFrame-tracing callable::

        import numpy as np
        from yobx.sql import to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        def transform(df):
            df = df.filter(df["a"] > 0)
            return df.select([(df["a"] + df["b"]).alias("total")])

        dtypes = {"a": np.float32, "b": np.float32}
        artifact = to_onnx(transform, dtypes)

        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0,  5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        # total == array([5., 9.], dtype=float32)  (rows where a > 0)

    Example — from a polars LazyFrame::

        import numpy as np
        import polars as pl
        from yobx.sql import to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        lf = pl.LazyFrame({"a": [1.0, -2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        lf = lf.filter(pl.col("a") > 0).select(
            [(pl.col("a") + pl.col("b")).alias("total")]
        )
        dtypes = {"a": np.float64, "b": np.float64}
        artifact = to_onnx(lf, dtypes)

        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float64)
        b = np.array([4.0,  5.0, 6.0], dtype=np.float64)
        (total,) = ref.run(None, {"a": a, "b": b})
        # total == array([5., 9.])  (rows where a > 0)

    .. note::

        ``GROUP BY`` produces one output row per unique key combination.
        Supported aggregations: ``SUM``, ``AVG``, ``MIN``, ``MAX``,
        ``COUNT(*)``.  For multi-column ``GROUP BY`` the grouping keys are
        cast to ``float64`` internally, which may lose precision for integers
        larger than 2^53.
    """
    if callable(dataframe_or_query) and not isinstance(dataframe_or_query, str):
        return dataframe_to_onnx(
            dataframe_or_query,  # type: ignore
            input_dtypes,
            target_opset=target_opset,
            custom_functions=custom_functions,
            builder_cls=builder_cls,
            filename=filename,
            verbose=verbose,
        )
    if isinstance(dataframe_or_query, str):
        return sql_to_onnx(
            dataframe_or_query,
            input_dtypes,  # type: ignore
            right_input_dtypes=right_input_dtypes,
            target_opset=target_opset,
            custom_functions=custom_functions,
            builder_cls=builder_cls,
            filename=filename,
            verbose=verbose,
        )
    return lazyframe_to_onnx(
        dataframe_or_query,
        input_dtypes,  # type: ignore
        target_opset=target_opset,
        builder_cls=builder_cls,
        filename=filename,
        verbose=verbose,
    )
