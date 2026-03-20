from typing import Any, Callable, Dict, Tuple, Union
from .. import DEFAULT_TARGET_OPSET
from ..container import ExportArtifact
from ..xbuilder import GraphBuilder
from .polars_convert import lazyframe_to_onnx
from .sql_convert import sql_to_onnx, sql_to_onnx_graph  # noqa: F401 – re-exported

import numpy as np


def to_onnx(
    dataframe_or_query: Union[str, "polars.LazyFrame"],  # noqa: F821, UP037
    input_dtypes: Dict[str, Union[np.dtype, type, str]],
    target_opset: int = DEFAULT_TARGET_OPSET,
    builder_cls: Union[type, Callable] = GraphBuilder,
) -> ExportArtifact:
    """Convert either a SQL query string or a :class:`polars.LazyFrame` to ONNX.

    This is the unified entry point that dispatches to :func:`sql_to_onnx`
    when *dataframe_or_query* is a string, or to :func:`lazyframe_to_onnx`
    when it is a :class:`polars.LazyFrame`.

    :param dataframe_or_query: a SQL query string **or** a
        :class:`polars.LazyFrame`.
    :param input_dtypes: a mapping from *source* column name to numpy dtype.
        For SQL queries this maps left-table columns; for a ``LazyFrame`` it
        maps the source DataFrame columns used in the plan.
    :param target_opset: ONNX opset version to target (default:
        :data:`yobx.DEFAULT_TARGET_OPSET`).
    :param builder_cls: the graph-builder class (or factory callable) to use.
        Defaults to :class:`~yobx.xbuilder.GraphBuilder`.
    :return: :class:`~yobx.container.ExportArtifact` wrapping the exported
        ONNX model together with an :class:`~yobx.container.ExportReport`.

    Example::

        import numpy as np
        import polars as pl
        from yobx.sql import to_onnx

        # From a SQL string:
        artifact = to_onnx(
            "SELECT a + b AS total FROM t",
            {"a": np.float64, "b": np.float64},
        )

        # From a polars LazyFrame:
        lf = pl.LazyFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        lf = lf.select([(pl.col("a") + pl.col("b")).alias("total")])
        artifact = to_onnx(lf, {"a": np.float64, "b": np.float64})
    """
    if isinstance(dataframe_or_query, str):
        return sql_to_onnx(
            dataframe_or_query, input_dtypes, target_opset=target_opset, builder_cls=builder_cls
        )
    return lazyframe_to_onnx(
        dataframe_or_query, input_dtypes, target_opset=target_opset, builder_cls=builder_cls
    )
