"""
Unified SQL / polars LazyFrame / DataFrame-function → ONNX dispatcher.

:func:`to_onnx` is the single entry point that accepts a plain SQL query
string, a :class:`polars.LazyFrame`, or a Python callable that operates on a
:class:`~yobx.xtracing.dataframe_trace.TracedDataFrame`, and returns a
self-contained :class:`~yobx.container.ExportArtifact`.  Internally it
delegates to :func:`~yobx.sql.sql_convert.sql_to_onnx`,
:func:`~yobx.sql.polars_convert.lazyframe_to_onnx`, or
:func:`dataframe_to_onnx` depending on the type of the first argument.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .. import DEFAULT_TARGET_OPSET
from ..container import ExportArtifact
from ..helpers.onnx_helper import np_dtype_to_tensor_dtype
from ..xbuilder import GraphBuilder
from ..xtracing.dataframe_trace import trace_dataframe
from ..xtracing.parse import JoinOp
from ..xtracing.tracing import trace_numpy_function
from .polars_convert import lazyframe_to_onnx
from .sql_convert import sql_to_onnx, sql_to_onnx_graph  # noqa: F401 – re-exported

#: Name used for the dynamic (batch) first dimension when none is specified.
_BATCH_DIM = "batch"


def dataframe_to_onnx(
    func: Callable,
    input_dtypes: Union[
        Dict[str, Union[np.dtype, type, str]], List[Dict[str, Union[np.dtype, type, str]]]
    ],
    target_opset: int = DEFAULT_TARGET_OPSET,
    custom_functions: Optional[Dict[str, Callable]] = None,
    builder_cls: Union[type, Callable] = GraphBuilder,
    filename: Optional[str] = None,
    verbose: int = 0,
) -> ExportArtifact:
    """Trace *func* and convert the resulting computation to ONNX.

    Combines :func:`~yobx.xtracing.dataframe_trace.trace_dataframe` and
    :func:`~yobx.sql.sql_convert.parsed_query_to_onnx` into a single call.

    :param func: a callable that accepts one or more
        :class:`~yobx.xtracing.dataframe_trace.TracedDataFrame`
        objects and returns a
        :class:`~yobx.xtracing.dataframe_trace.TracedDataFrame`.
    :param input_dtypes: either

        * a single ``{column: dtype}`` mapping — *func* is called with one
          :class:`~yobx.xtracing.dataframe_trace.TracedDataFrame`; or
        * a **list** of ``{column: dtype}`` mappings — *func* is called with
          one :class:`~yobx.xtracing.dataframe_trace.TracedDataFrame` per
          mapping, in order.  When the traced function contains a
          :class:`~yobx.xtracing.parse.JoinOp`, the first mapping is used as
          the *left-table* dtypes and the second as the *right-table* dtypes.
          For functions that simply share columns across multiple frames
          without a join, all mappings are merged into a single input-dtype
          dict.

    :param target_opset: ONNX opset version to target (default:
        :data:`yobx.DEFAULT_TARGET_OPSET`).
    :param custom_functions: optional mapping from function name to Python
        callable.  Functions registered here can be called inside the traced
        body via :class:`~yobx.xtracing.parse.FuncCallExpr` nodes if the
        traced function constructs them directly (advanced usage).
    :param builder_cls: graph-builder class or factory callable.  Defaults to
        :class:`~yobx.xbuilder.GraphBuilder`.
    :param filename: if set, the exported ONNX model is saved to this path and
        the :class:`~yobx.container.ExportReport` is written as a companion
        Excel file (same base name with ``.xlsx`` extension).
    :param verbose: verbosity level (0 = silent).
    :return: :class:`~yobx.container.ExportArtifact` wrapping the exported
        ONNX model together with an :class:`~yobx.container.ExportReport`.

    Example — single dataframe::

        import numpy as np
        from yobx.sql import dataframe_to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        def transform(df):
            df = df.filter(df["a"] > 0)
            return df.select([(df["a"] + df["b"]).alias("total")])

        dtypes = {"a": np.float32, "b": np.float32}
        artifact = dataframe_to_onnx(transform, dtypes)

        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        b = np.array([4.0,  5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        # total == array([5., 9.], dtype=float32)  (rows where a > 0)

    Example — two independent dataframes (no join)::

        import numpy as np
        from yobx.sql import dataframe_to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        def transform(df1, df2):
            return df1.select([(df1["a"] + df2["b"]).alias("total")])

        artifact = dataframe_to_onnx(transform, [{"a": np.float32}, {"b": np.float32}])

        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        (total,) = ref.run(None, {"a": a, "b": b})
        # total == array([5., 7., 9.], dtype=float32)

    Example — two dataframes joined on a key column::

        import numpy as np
        from yobx.sql import dataframe_to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        def transform(df1, df2):
            return df1.join(df2, left_key="cid", right_key="id")

        dtypes1 = {"cid": np.int64, "a": np.float32}
        dtypes2 = {"id": np.int64, "b": np.float32}
        artifact = dataframe_to_onnx(transform, [dtypes1, dtypes2])
    """
    from .sql_convert import parsed_query_to_onnx  # avoid top-level circular import

    pq = trace_dataframe(func, input_dtypes)
    if isinstance(input_dtypes, list):
        has_join = any(isinstance(op, JoinOp) for op in pq.operations)
        if has_join:
            left_dtypes: Dict[str, Union[np.dtype, type, str]] = input_dtypes[0]
            right_dtypes: Optional[Dict[str, Union[np.dtype, type, str]]] = (
                input_dtypes[1] if len(input_dtypes) > 1 else None
            )
        else:
            # Merge all dtype dicts — columns come from a flat table.
            left_dtypes = {}
            for d in input_dtypes:
                left_dtypes.update(d)
            right_dtypes = None
        return parsed_query_to_onnx(
            pq,
            left_dtypes,
            right_input_dtypes=right_dtypes,
            target_opset=target_opset,
            custom_functions=custom_functions,
            builder_cls=builder_cls,
            filename=filename,
            verbose=verbose,
        )
    return parsed_query_to_onnx(
        pq,
        input_dtypes,
        target_opset=target_opset,
        custom_functions=custom_functions,
        builder_cls=builder_cls,
        filename=filename,
        verbose=verbose,
    )


def trace_numpy_to_onnx(
    func: Callable,
    *inputs: np.ndarray,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    target_opset: Union[int, Dict[str, int]] = DEFAULT_TARGET_OPSET,
    batch_dim: str = _BATCH_DIM,
) -> ExportArtifact:
    """
    Trace a numpy function and return the equivalent ONNX model.

    This is the high-level entry point.  Internally it creates a fresh
    :class:`~yobx.xbuilder.GraphBuilder`, registers the graph inputs derived
    from the *inputs* sample arrays, delegates the actual tracing to
    :func:`~yobx.xtracing.tracing.trace_numpy_function`, registers the graph
    outputs, and exports to :class:`onnx.ModelProto`.

    :param func: a Python callable that accepts one or more numpy arrays and
        returns a numpy array (or a tuple/list of arrays).  The function may
        use numpy ufuncs, arithmetic operators, reductions, and shape
        manipulations; see :mod:`yobx.xtracing.numpy_array` for the full list
        of supported operations.
    :param inputs: sample numpy arrays used to determine the element type and
        shape of each input.  Only ``dtype`` and ``shape`` are used; the
        actual values are ignored.
    :param input_names: optional list of tensor names for the ONNX graph
        inputs.  Defaults to ``["X"]`` for a single input or ``["X0", "X1",
        …]`` for multiple inputs.
    :param output_names: optional list of tensor names for the ONNX graph
        outputs.  Defaults to ``["output_0"]``.  For functions that return
        multiple arrays supply the correct number of names, e.g.
        ``["out_a", "out_b"]``.
    :param target_opset: ONNX opset version.  Can be an integer (default
        domain only) or a dictionary mapping domain names to opset versions.
    :param batch_dim: name of the dynamic first dimension (default:
        ``"batch"``).  Change this when the default name conflicts with
        another symbolic dimension in the same graph.
    :return: an :class:`~yobx.container.ExportArtifact` representing the
        traced function.

    Example::

        import numpy as np
        from yobx.sql import trace_numpy_to_onnx

        def my_func(X):
            return np.sqrt(np.abs(X) + 1)

        X = np.random.randn(4, 3).astype(np.float32)
        onx = trace_numpy_to_onnx(my_func, X)
    """
    if not inputs:
        raise ValueError("At least one sample input array must be provided.")

    if isinstance(target_opset, int):
        opsets: Dict[str, int] = {"": target_opset, "ai.onnx.ml": 1}
    else:
        opsets = dict(target_opset)
        if "" not in opsets:
            opsets[""] = DEFAULT_TARGET_OPSET
        if "ai.onnx.ml" not in opsets:
            opsets["ai.onnx.ml"] = 1

    if input_names is None:
        resolved_input_names: List[str] = (
            ["X"] if len(inputs) == 1 else [f"X{i}" for i in range(len(inputs))]
        )
    else:
        resolved_input_names = list(input_names)
        if len(resolved_input_names) != len(inputs):
            raise ValueError(
                f"Length mismatch: {len(inputs)} sample inputs but "
                f"input_names has {len(resolved_input_names)} elements."
            )

    g = GraphBuilder(opsets)  # type: ignore

    for iname, arr in zip(resolved_input_names, inputs):
        itype = np_dtype_to_tensor_dtype(arr.dtype)
        # Make the first (batch) dimension dynamic; keep the rest static.
        shape: Tuple = (batch_dim, *arr.shape[1:])  # type: ignore[assignment]
        g.make_tensor_input(iname, itype, shape)

    if output_names is not None:
        resolved_output_names: List[str] = list(output_names)
        trace_numpy_function(
            g, {}, resolved_output_names, func, resolved_input_names, name="trace"
        )
    else:
        # Auto-detect the number of outputs by passing an empty list and letting
        # trace_numpy_function return whatever the traced function produced.
        # Returns str for single output, tuple[str, ...] for multiple outputs.
        raw_result = trace_numpy_function(g, {}, None, func, resolved_input_names, name="trace")
        resolved_output_names = [raw_result] if isinstance(raw_result, str) else list(raw_result)

    for out_name in resolved_output_names:
        g.make_tensor_output(out_name, indexed=False, allow_untyped_output=True)

    onx = g.to_onnx(return_optimize_report=True)  # type: ignore
    return onx


def to_onnx(
    dataframe_or_query: Union[  # type: ignore
        str,
        Callable[["TracedDataFrame"], "TracedDataFrame"],  # type: ignore # noqa: F821
        "polars.LazyFrame",  # type: ignore # noqa: F821
    ],
    input_dtypes: Union[
        Dict[str, Union[np.dtype, type, str]], List[Dict[str, Union[np.dtype, type, str]]]
    ],
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
