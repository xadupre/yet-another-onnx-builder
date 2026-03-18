"""
SQL operation converters for the ONNX builder.

Importing this package registers all built-in SQL op converters via
:func:`~yobx.sql.ops.register.register_sql_op_converter`.
"""

from .register import get_sql_op_converter, get_sql_op_converters, register_sql_op_converter

# Import op modules to trigger @register_sql_op_converter decoration.
from . import filter_op, join_op

__all__ = [
    "register_sql_op_converter",
    "get_sql_op_converter",
    "get_sql_op_converters",
    "filter_op",
    "join_op",
]
