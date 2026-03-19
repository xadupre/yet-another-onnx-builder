"""
Registration mechanism for SQL operation converters.

Each SQL operation type (e.g. :class:`~yobx.sql.parse.FilterOp`,
:class:`~yobx.sql.parse.JoinOp`) can have exactly one converter registered via
:func:`register_sql_op_converter`.  Converters follow the same API convention
as other converters in this package::

    def convert_my_op(
        g: GraphBuilder,
        sts: Dict,
        outputs: List[str],
        op: MyOp,
        col_map: Dict[str, str],
        right_col_map: Dict[str, str],
    ) -> Dict[str, str]:
        ...

where

* *g* — the :class:`~yobx.xbuilder.GraphBuilder` to add ONNX nodes to.
* *sts* — shape/type context dict (may be empty; forwarded verbatim).
* *outputs* — the expected output column names (keys of the returned col_map).
* *op* — the SQL operation object being converted.
* *col_map* — mapping from column name to the current ONNX tensor name (left table).
* *right_col_map* — same for the right table (empty dict when not applicable).

The function must return a new ``col_map`` reflecting the state after applying
the operation.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

SQL_OP_CONVERTERS: Dict[type, Callable] = {}


def register_sql_op_converter(cls: type) -> Callable:
    """
    Decorator that registers a converter for a SQL operation class.

    :param cls: the SQL operation class (e.g. ``FilterOp``, ``JoinOp``).
    :return: the decorator.

    Usage::

        @register_sql_op_converter(FilterOp)
        def convert_filter_op(g, sts, outputs, op, col_map, right_col_map):
            ...
    """

    def decorator(fct: Callable) -> Callable:
        global SQL_OP_CONVERTERS
        if cls in SQL_OP_CONVERTERS:
            raise TypeError(f"A SQL op converter is already registered for {cls}.")
        SQL_OP_CONVERTERS[cls] = fct
        return fct

    return decorator


def get_sql_op_converter(cls: type) -> Optional[Callable]:
    """
    Return the converter registered for *cls*, or ``None`` if none is registered.

    :param cls: the SQL operation class to look up.
    :return: the converter callable, or ``None``.
    """
    global SQL_OP_CONVERTERS
    return SQL_OP_CONVERTERS.get(cls)


def get_sql_op_converters() -> Dict[type, Callable]:
    """Return a copy of the full SQL op converter registry."""
    global SQL_OP_CONVERTERS
    return dict(SQL_OP_CONVERTERS)
