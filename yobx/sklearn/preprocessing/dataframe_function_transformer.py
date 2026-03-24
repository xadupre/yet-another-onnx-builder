"""
Converter for :class:`~yobx.sklearn.preprocessing.DataFrameTransformer`.

This module is imported lazily by :func:`yobx.sklearn.register_sklearn_converters`
(via :func:`yobx.sklearn.preprocessing.register`).  It must **not** be imported
at module-level by user code because importing it registers the
:class:`~yobx.sklearn.preprocessing.DataFrameTransformer` converter in
:data:`~yobx.sklearn.register.SKLEARN_CONVERTERS`, which would cause
:func:`~yobx.sklearn.register_sklearn_converters` to short-circuit and skip
registering all other converters.

Users should import :class:`~yobx.sklearn.preprocessing.DataFrameTransformer`
from :mod:`yobx.sklearn.preprocessing` (or :mod:`yobx.sklearn`), not from this
module directly.
"""

from typing import Dict, List, Optional

from ._dataframe_transformer import DataFrameTransformer
from ..register import register_sklearn_converter
from ...typing import GraphBuilderExtendedProtocol


@register_sklearn_converter(DataFrameTransformer)
def sklearn_dataframe_transformer(
    g: GraphBuilderExtendedProtocol,
    sts: Dict,
    outputs: Optional[List[str]],
    estimator: DataFrameTransformer,
    *inputs: str,
    name: str = "df_transformer",
) -> List[str]:
    """
    Convert a :class:`~yobx.sklearn.preprocessing.DataFrameTransformer` into
    ONNX nodes.

    The tracing function :attr:`~DataFrameTransformer.func` is traced via
    :func:`~yobx.sql.trace_dataframe` and the resulting
    :class:`~yobx.sql.parse.ParsedQuery` is compiled into *g* via
    :func:`~yobx.sql.sql_convert.parsed_query_to_onnx_graph` with
    ``_finalize=False`` (the caller manages output registration).

    :param g: graph builder to add nodes to.
    :param sts: shared converter state dict (may contain
        ``"custom_functions"``).
    :param outputs: desired output tensor names; may be ``None`` when the
        :class:`DataFrameTransformer` inherits from
        :class:`~yobx.sklearn.NoKnownOutputMixin`.
    :param estimator: a fitted :class:`DataFrameTransformer`.
    :param inputs: input tensor name(s) already registered in *g* (not used
        directly; the column names from
        :attr:`~DataFrameTransformer.input_dtypes_` match the graph input
        names created by ``register_inputs``).
    :param name: node-name prefix.
    :return: list of output tensor names added to *g*.
    """
    from yobx.sql import trace_dataframe
    from yobx.sql.sql_convert import parsed_query_to_onnx_graph

    if not isinstance(estimator, DataFrameTransformer):
        raise TypeError(
            f"Expected DataFrameTransformer, got {type(estimator)}"
        )

    pq = trace_dataframe(estimator.func, estimator.input_dtypes_)
    out_names = parsed_query_to_onnx_graph(
        g,
        sts,
        list(outputs) if outputs else [],
        pq,
        estimator.input_dtypes_,
        _finalize=False,
    )
    return out_names
