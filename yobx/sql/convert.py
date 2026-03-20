from typing import Any, Callable, Tuple, Union
from .. import DEFAULT_TARGET_OPSET
from ..typing import GraphBuilderExtendedProtocol
from ..container import ExportArtifact
from .polars_convert import lazyframe_to_onnx
from .sql_convert import sql_to_onnx


def to_onnx(
    dataframe_or_query: Union[str, "polars.LazyFrame"],  # noqa: F821
    args: Tuple[Any],
    target_opset: int = DEFAULT_TARGET_OPSET,
    builder_cls: Union[type, Callable] = GraphBuilderExtendedProtocol,
) -> ExportArtifact:
    """Converts aither a `polars.DataFRame` or a SQL query."""
    if isinstance(dataframe_or_query, str):
        return sql_to_onnx(
            dataframe_or_query, args, target_opset=target_opset, builder_cls=builder_cls
        )
    return lazyframe_to_onnx(
        dataframe_or_query, args, target_opset=target_opset, builder_cls=builder_cls
    )
