"""
Polars helper utilities for the SQL-to-ONNX converter.

This module provides the mapping from polars column dtypes to numpy dtypes,
the :func:`polars_schema_to_input_dtypes` helper, and the
:func:`polars_frame_to_sql` helper used by :func:`~yobx.sql.to_onnx` to
extract column types and reconstruct a SQL query from a
``polars.LazyFrame`` created via :meth:`polars.LazyFrame.sql`.
"""

from __future__ import annotations

import io
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Polars dtype → numpy dtype mapping
# ---------------------------------------------------------------------------

# Mapping from polars DataType class name to numpy dtype.
_POLARS_DTYPE_TO_NP: Dict[str, np.dtype] = {
    "Float32": np.dtype("float32"),
    "Float64": np.dtype("float64"),
    "Int8": np.dtype("int8"),
    "Int16": np.dtype("int16"),
    "Int32": np.dtype("int32"),
    "Int64": np.dtype("int64"),
    "UInt8": np.dtype("uint8"),
    "UInt16": np.dtype("uint16"),
    "UInt32": np.dtype("uint32"),
    "UInt64": np.dtype("uint64"),
    "Boolean": np.dtype("bool"),
    "String": np.dtype("object"),
    "Utf8": np.dtype("object"),
}


def polars_schema_to_input_dtypes(frame) -> Dict[str, np.dtype]:
    """Extract a column-name → numpy-dtype mapping from a polars frame.

    Accepts a ``polars.LazyFrame`` (schema read via :meth:`collect_schema`,
    no data collected) or a ``polars.DataFrame`` (schema read via
    :attr:`schema`).

    :param frame: a ``polars.LazyFrame`` or ``polars.DataFrame``.
    :return: dict mapping each column name to its numpy dtype.
    :raises ImportError: if *polars* is not installed.
    :raises TypeError: if *frame* is neither a LazyFrame nor a DataFrame.
    :raises ValueError: if any column dtype has no supported numpy equivalent.

    Supported polars dtypes: ``Float32``, ``Float64``, ``Int8``, ``Int16``,
    ``Int32``, ``Int64``, ``UInt8``, ``UInt16``, ``UInt32``, ``UInt64``,
    ``Boolean``, ``String``, ``Utf8``.
    """
    import polars as pl  # noqa: F401

    if hasattr(frame, "collect_schema"):
        # polars.LazyFrame — cheap, no execution
        schema = frame.collect_schema()
    elif hasattr(frame, "schema"):
        # polars.DataFrame
        schema = frame.schema
    else:
        raise TypeError(f"Expected a polars.LazyFrame or polars.DataFrame, got {type(frame)!r}")

    result: Dict[str, np.dtype] = {}
    for col_name, dtype in schema.items():
        key = type(dtype).__name__
        if key not in _POLARS_DTYPE_TO_NP:
            raise ValueError(
                f"Column {col_name!r} has polars dtype {dtype!r} which cannot be "
                f"mapped to a numpy dtype supported by sql_to_onnx. "
                f"Supported polars dtypes: {sorted(_POLARS_DTYPE_TO_NP)}"
            )
        result[col_name] = _POLARS_DTYPE_TO_NP[key]
    return result


# ---------------------------------------------------------------------------
# Polars plan → SQL reconstruction
# ---------------------------------------------------------------------------

# Binary operator name → SQL token
_PLAN_BINOP: Dict[str, str] = {
    "Plus": "+",
    "Minus": "-",
    "Multiply": "*",
    "Divide": "/",
    "Gt": ">",
    "Lt": "<",
    "GtEq": ">=",
    "LtEq": "<=",
    "Eq": "=",
    "NotEq": "!=",
    "And": "AND",
    "Or": "OR",
}


def _plan_expr_to_sql(expr) -> str:
    """Convert a single polars plan expression node to a SQL fragment."""
    if isinstance(expr, str):
        if expr == "Len":
            return "COUNT(*)"
        raise ValueError(f"Unsupported string expression in polars plan: {expr!r}")

    if not isinstance(expr, dict):
        raise ValueError(f"Unexpected expression type in polars plan: {type(expr)!r}")

    if "Column" in expr:
        return expr["Column"]

    if "Alias" in expr:
        items = expr["Alias"]
        inner_sql = _plan_expr_to_sql(items[0])
        alias = items[1]
        return f"({inner_sql}) AS {alias}"

    if "BinaryExpr" in expr:
        be = expr["BinaryExpr"]
        left_sql = _plan_expr_to_sql(be["left"])
        right_sql = _plan_expr_to_sql(be["right"])
        op = _PLAN_BINOP.get(be["op"])
        if op is None:
            raise ValueError(f"Unsupported binary operator in polars plan: {be['op']!r}")
        return f"({left_sql}) {op} ({right_sql})"

    if "Literal" in expr:
        lit = expr["Literal"]
        if "Dyn" in lit:
            dyn = lit["Dyn"]
            if "Int" in dyn:
                return str(dyn["Int"])
            if "Float" in dyn:
                return str(dyn["Float"])
            if "String" in dyn:
                s = str(dyn["String"]).replace("'", "''")
                return f"'{s}'"
        raise ValueError(f"Unsupported literal in polars plan: {lit!r}")

    if "Agg" in expr:
        agg = expr["Agg"]
        if "Sum" in agg:
            return f"SUM({_plan_expr_to_sql(agg['Sum'])})"
        if "Mean" in agg:
            return f"AVG({_plan_expr_to_sql(agg['Mean'])})"
        if "Min" in agg:
            return f"MIN({_plan_expr_to_sql(agg['Min']['input'])})"
        if "Max" in agg:
            return f"MAX({_plan_expr_to_sql(agg['Max']['input'])})"
        if "Count" in agg:
            return "COUNT(*)"
        raise ValueError(f"Unsupported aggregation in polars plan: {list(agg.keys())!r}")

    raise ValueError(f"Unsupported expression node in polars plan: {list(expr.keys())!r}")


def _plan_find_source(
    node: dict,
) -> Tuple[Dict[str, str], Optional[dict], Optional[dict], Optional[dict]]:
    """
    Recursively walk the plan tree to find the source DataFrameScan(s) and
    collect intermediate *Filter* and *GroupBy* nodes.

    Returns
    -------
    (schema_fields, filter_predicate, groupby_node, join_node)
        *schema_fields* is the dict ``{col_name: dtype_string}`` from the
        DataFrameScan.  *filter_predicate*, *groupby_node*, and *join_node*
        are either the corresponding plan dicts or ``None``.
    """
    if "IR" in node:
        return _plan_find_source(node["IR"]["dsl"])

    if "DataFrameScan" in node:
        fields = node["DataFrameScan"]["schema"]["fields"]
        return fields, None, None, None

    if "Filter" in node:
        f = node["Filter"]
        fields, pred, gb, join = _plan_find_source(f["input"])
        # Merge multiple filter predicates with AND
        if pred is not None:
            new_pred: dict = {"BinaryExpr": {"left": pred, "op": "And", "right": f["predicate"]}}
        else:
            new_pred = f["predicate"]
        return fields, new_pred, gb, join

    if "GroupBy" in node:
        gb = node["GroupBy"]
        fields, pred, _, join = _plan_find_source(gb["input"])
        return fields, pred, gb, join

    if "Select" in node:
        # Nested SELECT (e.g. the inner projection added for GROUP BY)
        return _plan_find_source(node["Select"]["input"])

    if "Join" in node:
        join = node["Join"]
        # Merge left and right schemas
        left_fields, _, _, _ = _plan_find_source(join["input_left"])
        right_fields, _, _, _ = _plan_find_source(join["input_right"])
        merged = dict(left_fields)
        merged.update(right_fields)
        return merged, None, None, join

    raise ValueError(
        f"Cannot extract SQL from polars plan: unsupported node {list(node.keys())!r}. "
        f"Only plans produced by LazyFrame.sql() on simple SELECT / WHERE / "
        f"GROUP BY / JOIN queries are supported."
    )


def _schema_fields_to_dtypes(fields: Dict[str, str]) -> Dict[str, np.dtype]:
    """Convert a ``{col: dtype_string}`` dict from the plan JSON to numpy dtypes."""
    result: Dict[str, np.dtype] = {}
    for col, dtype_str in fields.items():
        if dtype_str not in _POLARS_DTYPE_TO_NP:
            raise ValueError(
                f"Column {col!r} has dtype {dtype_str!r} in the polars plan which "
                f"has no supported numpy equivalent. "
                f"Supported polars dtypes: {sorted(_POLARS_DTYPE_TO_NP)}"
            )
        result[col] = _POLARS_DTYPE_TO_NP[dtype_str]
    return result


def polars_frame_to_sql(frame, table_name: str = "t") -> Tuple[str, Dict[str, np.dtype]]:
    """Extract a SQL string and input-dtype map from a polars ``LazyFrame``.

    The *frame* must have been produced by calling
    :meth:`polars.LazyFrame.sql` on a source frame.  This function
    serialises the internal logical plan to JSON and reconstructs an
    equivalent SQL query that :func:`~yobx.sql.sql_to_onnx` can compile.

    :param frame: a ``polars.LazyFrame`` created with ``.sql(...)``.
    :param table_name: the table alias to use in the generated ``FROM``
        clause (default ``"t"``).
    :return: ``(query_string, input_dtypes)`` where *query_string* is a SQL
        string understood by :func:`~yobx.sql.sql_to_onnx` and
        *input_dtypes* maps each source column name to its numpy dtype.
    :raises ImportError: if *polars* is not installed.
    :raises TypeError: if *frame* is not a ``polars.LazyFrame``.
    :raises ValueError: if the plan cannot be converted to SQL (e.g. the
        frame was not created via ``.sql()``).
    """
    if not hasattr(frame, "_ldf"):
        raise TypeError(
            f"Expected a polars.LazyFrame, got {type(frame)!r}. "
            f"Call frame.sql(query) to attach a SQL query to the frame first."
        )

    buf = io.StringIO()
    frame._ldf.serialize_json(buf)
    buf.seek(0)
    plan: dict = json.load(buf)

    # Top-level node must be Select (produced by lf.sql(...))
    if "Select" not in plan:
        raise ValueError(
            f"Cannot extract SQL from polars plan: expected a Select node at "
            f"the top level but got {list(plan.keys())!r}. "
            f"Pass the result of frame.sql(query) to to_onnx, or supply the "
            f"query string explicitly as the second argument."
        )

    top = plan["Select"]
    select_exprs: List[dict] = top["expr"]

    # Walk the input to collect source schema + intermediate nodes
    fields, filter_pred, groupby, join_node = _plan_find_source(top["input"])
    input_dtypes = _schema_fields_to_dtypes(fields)

    # Build the SQL query string
    if groupby is not None:
        key_sqls = [_plan_expr_to_sql(k) for k in groupby["keys"]]
        agg_sqls = [_plan_expr_to_sql(a) for a in groupby["aggs"]]
        select_clause = ", ".join(key_sqls + agg_sqls)
        group_clause = ", ".join(key_sqls)
        from_clause = f"FROM {table_name}"
        where_clause = f" WHERE {_plan_expr_to_sql(filter_pred)}" if filter_pred else ""
        query = f"SELECT {select_clause} {from_clause}{where_clause} GROUP BY {group_clause}"
    elif join_node is not None:
        if len(join_node["left_on"]) != 1 or len(join_node["right_on"]) != 1:
            raise ValueError(
                f"polars_frame_to_sql only supports single-column equi-joins; "
                f"got left_on={join_node['left_on']!r}, right_on={join_node['right_on']!r}."
            )
        left_on = _plan_expr_to_sql(join_node["left_on"][0])
        right_on = _plan_expr_to_sql(join_node["right_on"][0])
        select_clause = ", ".join(_plan_expr_to_sql(e) for e in select_exprs)
        join_type = join_node.get("options", {}).get("args", {}).get("how", "Inner")
        _JOIN_KW = {"Inner": "INNER JOIN", "Left": "LEFT JOIN", "Right": "RIGHT JOIN"}
        if join_type not in _JOIN_KW:
            raise ValueError(
                f"polars_frame_to_sql does not support join type {join_type!r}. "
                f"Supported types: {sorted(_JOIN_KW)}."
            )
        join_kw = _JOIN_KW[join_type]
        right_table = f"{table_name}_r"
        from_clause = (
            f"FROM {table_name} {join_kw} {right_table} ON "
            f"{table_name}.{left_on} = {right_table}.{right_on}"
        )
        where_clause = f" WHERE {_plan_expr_to_sql(filter_pred)}" if filter_pred else ""
        query = f"SELECT {select_clause} {from_clause}{where_clause}"
    else:
        select_clause = ", ".join(_plan_expr_to_sql(e) for e in select_exprs)
        where_clause = f" WHERE {_plan_expr_to_sql(filter_pred)}" if filter_pred else ""
        query = f"SELECT {select_clause} FROM {table_name}{where_clause}"

    return query, input_dtypes
