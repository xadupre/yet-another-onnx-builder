"""polars LazyFrame → ONNX converter.

Converts a ``polars.LazyFrame`` execution plan (as returned by
``LazyFrame.explain()``) into a self-contained ONNX model by translating
the polars logical plan into a SQL query that is then processed by
:func:`~yobx.sql.sql_convert.sql_to_onnx`.

Supported polars operations
---------------------------
* Column selection (``select``)
* Row filtering (``filter``)
* Arithmetic binary expressions: ``+``, ``-``, ``*``, ``/``
* Comparison predicates: ``>``, ``<``, ``>=``, ``<=``, ``==``, ``!=``
* Compound predicates: ``&`` (AND), ``|`` (OR)
* Aggregations via methods: ``.sum()``, ``.mean()`` (→ AVG), ``.min()``,
  ``.max()``, ``.count()``
* Column aliases via ``.alias(...)``
* ``group_by`` / ``agg``
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .. import DEFAULT_TARGET_OPSET
from ..container import ExportArtifact
from ..xbuilder import GraphBuilder
from .coverage import not_implemented_error
from .sql_convert import sql_to_onnx

# ---------------------------------------------------------------------------
# Polars → numpy dtype mapping
# ---------------------------------------------------------------------------


def _polars_dtype_to_numpy(dtype: object) -> np.dtype:
    """Map a polars ``DataType`` to a numpy ``dtype``.

    :param dtype: a polars data type (e.g. ``polars.Float32``).
    :return: the equivalent :class:`numpy.dtype`.
    :raises ImportError: if polars is not installed.
    :raises ValueError: if the polars dtype has no numpy equivalent.
    """
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError("polars is required for lazyframe_to_onnx") from exc

    _MAP: List[Tuple[object, np.dtype]] = [
        (pl.Float32, np.dtype("float32")),
        (pl.Float64, np.dtype("float64")),
        (pl.Int8, np.dtype("int8")),
        (pl.Int16, np.dtype("int16")),
        (pl.Int32, np.dtype("int32")),
        (pl.Int64, np.dtype("int64")),
        (pl.UInt8, np.dtype("uint8")),
        (pl.UInt16, np.dtype("uint16")),
        (pl.UInt32, np.dtype("uint32")),
        (pl.UInt64, np.dtype("uint64")),
        (pl.Boolean, np.dtype("bool")),
        (pl.String, np.dtype("object")),
        (pl.Utf8, np.dtype("object")),
    ]
    for pl_dtype, np_dtype in _MAP:
        if dtype == pl_dtype:
            return np_dtype
    raise ValueError(f"Unsupported polars dtype for ONNX conversion: {dtype!r}")


# ---------------------------------------------------------------------------
# Expression string helpers
# ---------------------------------------------------------------------------


def _split_top_level(s: str, sep: str) -> List[str]:
    """Split *s* on *sep* only when bracket depth is zero.

    Respects ``(``, ``)``, ``[``, ``]``, ``{``, ``}`` brackets.

    :param s: string to split.
    :param sep: separator token.
    :return: list of non-empty parts.
    """
    items: List[str] = []
    depth = 0
    start = 0
    sep_len = len(sep)
    i = 0
    while i < len(s):
        c = s[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0 and s[i : i + sep_len] == sep:
            items.append(s[start:i].strip())
            start = i + sep_len
            i += sep_len
            continue
        i += 1
    tail = s[start:].strip()
    if tail:
        items.append(tail)
    return items


def _is_balanced(s: str) -> bool:
    """Return ``True`` if *s* has balanced ``()``, ``[]``, ``{}`` brackets."""
    depth = 0
    for c in s:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _strip_outer_brackets(s: str) -> str:
    """Strip one or more layers of matching outer ``[...]`` or ``(...)``."""
    s = s.strip()
    changed = True
    while changed:
        changed = False
        if len(s) >= 2 and ((s[0] == "[" and s[-1] == "]") or (s[0] == "(" and s[-1] == ")")):
            inner = s[1:-1]
            if _is_balanced(inner):
                s = inner.strip()
                changed = True
    return s


def _polars_expr_to_sql(expr: str) -> str:
    """Convert a single polars expression string to a SQL expression.

    Handles column references, arithmetic, comparison predicates, boolean
    operators, aggregation methods and aliases, as produced by
    :meth:`polars.LazyFrame.explain`.

    :param expr: polars expression string fragment.
    :return: SQL expression string.
    """
    s = expr.strip()

    # ------------------------------------------------------------------
    # 1. Handle .alias("name") suffix (highest priority)
    # ------------------------------------------------------------------
    alias_m = re.search(r'\.alias\("([^"]+)"\)\s*$', s)
    if alias_m:
        inner = s[: alias_m.start()].strip()
        alias_name = alias_m.group(1)
        return f"{_polars_expr_to_sql(inner)} AS {alias_name}"

    # ------------------------------------------------------------------
    # 2. Strip outer [ ] or ( ) wrappers
    # ------------------------------------------------------------------
    s_stripped = _strip_outer_brackets(s)
    if s_stripped != s:
        return _polars_expr_to_sql(s_stripped)

    # ------------------------------------------------------------------
    # 3. col("name").agg_method() — aggregation functions
    # ------------------------------------------------------------------
    agg_m = re.match(r'^col\("([^"]+)"\)\.(sum|mean|min|max|count)\(\)\s*$', s)
    if agg_m:
        col_name = agg_m.group(1)
        func = agg_m.group(2)
        sql_func = {"sum": "SUM", "mean": "AVG", "min": "MIN", "max": "MAX", "count": "COUNT"}[
            func
        ]
        return f"{sql_func}({col_name})"

    # ------------------------------------------------------------------
    # 4. col("name") — bare column reference
    # ------------------------------------------------------------------
    col_m = re.match(r'^col\("([^"]+)"\)\s*$', s)
    if col_m:
        return col_m.group(1)

    # ------------------------------------------------------------------
    # 5. Boolean AND / OR at depth-0 level  (polars uses  & / |)
    # ------------------------------------------------------------------
    # Do NOT wrap sub-conditions in (...) — the SQL parser only handles
    # flat conditions; parenthesised conditions are not supported.
    and_parts = _split_top_level(s, " & ")
    if len(and_parts) > 1:
        return " AND ".join(_polars_expr_to_sql(p) for p in and_parts)

    or_parts = _split_top_level(s, " | ")
    if len(or_parts) > 1:
        return " OR ".join(_polars_expr_to_sql(p) for p in or_parts)

    # ------------------------------------------------------------------
    # 6. Binary arithmetic / comparison operators at depth-0
    #    Polars wraps each operand in parens: (left) OP (right)
    # ------------------------------------------------------------------
    # Comparison operators — do NOT add outer parens around the result
    # because the SQL parser cannot handle parenthesised conditions.
    for polars_op, sql_op in [
        (">=", ">="),
        ("<=", "<="),
        ("!=", "!="),
        ("==", "="),
        (">", ">"),
        ("<", "<"),
    ]:
        parts = _split_top_level(s, f" {polars_op} ")
        if len(parts) == 2:
            left = _polars_expr_to_sql(parts[0])
            right = _polars_expr_to_sql(parts[1])
            return f"{left} {sql_op} {right}"

    # Arithmetic operators — wrap operands in parens for correct precedence.
    for polars_op, sql_op in [("+", "+"), ("-", "-"), ("*", "*"), ("/", "/")]:
        parts = _split_top_level(s, f" {polars_op} ")
        if len(parts) == 2:
            left = _polars_expr_to_sql(parts[0])
            right = _polars_expr_to_sql(parts[1])
            return f"({left}) {sql_op} ({right})"

    # ------------------------------------------------------------------
    # 7. Numeric literal
    # ------------------------------------------------------------------
    if re.match(r"^-?[\d.]+(?:[eE][+-]?\d+)?$", s):
        return s

    # ------------------------------------------------------------------
    # 8. String literal (single or double quoted)
    # ------------------------------------------------------------------
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return f"'{s[1:-1]}'"

    # ------------------------------------------------------------------
    # 9. Bare word (unquoted column name or identifier)
    # ------------------------------------------------------------------
    if re.match(r"^\w+$", s):
        return s

    # Fallback: return as-is
    return s


# ---------------------------------------------------------------------------
# Plan parser
# ---------------------------------------------------------------------------


def _extract_bracketed_list(line: str, keyword: str) -> Optional[str]:
    """Extract the content of the first ``[...]`` block after *keyword*.

    :param line: a single line of the polars plan.
    :param keyword: the keyword prefix to search for (e.g. ``"SELECT "``).
    :return: the content between the outermost ``[`` and ``]``, or ``None``
        if the keyword or bracket were not found.
    """
    idx = line.find(keyword)
    if idx < 0:
        return None
    start = line.find("[", idx + len(keyword))
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(line)):
        if line[i] == "[":
            depth += 1
        elif line[i] == "]":
            depth -= 1
            if depth == 0:
                return line[start + 1 : i]
    return None


def _parse_df_cols(line: str) -> List[str]:
    """Extract the column list from a ``DF ["col1", "col2", ...]`` line.

    :param line: a stripped plan line starting with ``DF``.
    :return: list of column name strings.
    """
    inner = _extract_bracketed_list(line, "DF ")
    if inner is None:
        return []
    cols = []
    for part in inner.split(","):
        name = part.strip().strip('"').strip("'")
        if name:
            cols.append(name)
    return cols


class _PolarsPlan:
    """Parsed representation of a polars execution plan."""

    def __init__(self) -> None:
        self.select_items: List[str] = []
        self.where_condition: Optional[str] = None
        self.group_by_cols: List[str] = []
        self.agg_items: List[str] = []
        self.source_cols: List[str] = []


def _parse_polars_plan(plan: str) -> _PolarsPlan:
    """Parse a polars execution plan string into a :class:`_PolarsPlan`.

    :param plan: string returned by :meth:`polars.LazyFrame.explain`.
    :return: a :class:`_PolarsPlan` instance.
    """
    result = _PolarsPlan()
    lines = plan.strip().split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # SELECT [expr1, expr2, ...]
        if stripped.startswith("SELECT ["):
            inner = _extract_bracketed_list(stripped, "SELECT ")
            if inner is not None:
                items_raw = _split_top_level(inner, ", ")
                result.select_items = [_polars_expr_to_sql(it) for it in items_raw]
            i += 1
            continue

        # FILTER [condition]
        if stripped.startswith("FILTER ["):
            inner = _extract_bracketed_list(stripped, "FILTER ")
            if inner is not None:
                result.where_condition = _polars_expr_to_sql(inner)
            i += 1
            continue

        # AGGREGATE[...] header
        if stripped.startswith("AGGREGATE"):
            # Next line has:  [agg_exprs] BY [group_cols]
            i += 1
            if i < len(lines):
                agg_line = lines[i].strip()
                # Extract agg exprs: first [...]
                agg_inner = _extract_bracketed_list(agg_line, "")
                if agg_inner is not None:
                    agg_items_raw = _split_top_level(agg_inner, ", ")
                    result.agg_items = [_polars_expr_to_sql(it) for it in agg_items_raw]
                # Extract group-by cols: BY [...]
                by_inner = _extract_bracketed_list(agg_line, "BY ")
                if by_inner is not None:
                    gb_items_raw = _split_top_level(by_inner, ", ")
                    result.group_by_cols = [_polars_expr_to_sql(it) for it in gb_items_raw]
            i += 1
            continue

        # DF ["col1", "col2", ...] — source data frame
        if stripped.startswith("DF ["):
            cols = _parse_df_cols(stripped)
            if cols:
                result.source_cols = cols
            i += 1
            continue

        # Detect unsupported operations — raise early rather than silently
        # dropping the plan node (coverage data drives the error message).
        _upper = stripped.upper()
        if re.match(r"^(?:INNER\s+|LEFT\s+|RIGHT\s+|FULL\s+)?JOIN\b", _upper):
            raise not_implemented_error("polars", "lf.join")
        if _upper.startswith("SORT"):
            raise not_implemented_error("polars", "lf.sort")
        if _upper.startswith("UNIQUE"):
            raise not_implemented_error("polars", "lf.unique")
        if _upper.startswith(("SLICE", "LIMIT")):
            raise not_implemented_error("polars", "lf.limit")

        # FROM / END ... / other structural lines — skip
        i += 1

    return result


# ---------------------------------------------------------------------------
# SQL builder
# ---------------------------------------------------------------------------


def _plan_to_sql(plan: _PolarsPlan) -> str:
    """Build a SQL query string from a parsed polars plan.

    :param plan: parsed polars plan.
    :return: SQL query string suitable for :func:`~yobx.sql.convert.sql_to_onnx`.
    """
    # Determine SELECT clause
    if plan.agg_items:
        # AGGREGATE: combine agg_items (and group_by_cols if not already present)
        select_parts = list(plan.agg_items)
        if plan.select_items:
            # If there is also a SELECT wrapping the agg, prefer the outer SELECT
            select_parts = list(plan.select_items)
    elif plan.select_items:
        select_parts = list(plan.select_items)
    else:
        # No SELECT: pass through all source columns
        select_parts = list(plan.source_cols) if plan.source_cols else ["*"]

    select_clause = ", ".join(select_parts)
    sql = f"SELECT {select_clause} FROM t"

    if plan.where_condition:
        sql += f" WHERE {plan.where_condition}"

    if plan.group_by_cols:
        sql += " GROUP BY " + ", ".join(plan.group_by_cols)

    return sql


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lazyframe_to_onnx(
    lf: "polars.LazyFrame",  # type: ignore # noqa: F821
    input_dtypes: Dict[str, Union[np.dtype, type, str]],
    target_opset: int = DEFAULT_TARGET_OPSET,
    builder_cls: Union[type, Callable] = GraphBuilder,
    filename: Optional[str] = None,
    verbose: int = 0,
    large_model: bool = False,
    external_threshold: int = 1024,
    return_optimize_report: bool = False,
) -> ExportArtifact:
    """Convert a :class:`polars.LazyFrame` into a self-contained ONNX model.

    The function extracts the logical execution plan from the ``LazyFrame``
    via :meth:`polars.LazyFrame.explain`, translates it into a SQL query
    understood by :func:`~yobx.sql.convert.sql_to_onnx`, and returns an
    :class:`~yobx.container.ExportArtifact` containing the ONNX model.

    Each *source* column of the plan is represented as a separate 1-D ONNX
    input tensor.  The ONNX model outputs correspond to the columns or
    expressions in the ``select`` (or ``agg``) step of the plan.

    Supported ``LazyFrame`` operations
    -----------------------------------
    * ``select`` — column pass-through and arithmetic expressions
    * ``filter`` — row filtering with comparison and boolean predicates
    * ``group_by`` + ``agg`` — aggregations (``sum``, ``mean``, ``min``,
      ``max``, ``count``)

    :param lf: a :class:`polars.LazyFrame`.  The execution plan returned by
        ``lf.explain()`` is parsed and converted.
    :param input_dtypes: a mapping from **source** column name to numpy dtype
        (e.g. ``{"a": np.float32, "b": np.float64}``).  Only the columns
        that actually appear in the plan need to be listed.
    :param target_opset: ONNX opset version to target (default:
        :data:`yobx.DEFAULT_TARGET_OPSET`).
    :param builder_cls: the graph-builder class (or factory callable) to use.
        Defaults to :class:`~yobx.xbuilder.GraphBuilder`.
    :param filename: if set, the exported ONNX model is saved to this path and
        the :class:`~yobx.container.ExportReport` is written as a companion
        Excel file (same base name with ``.xlsx`` extension).
    :param verbose: verbosity level (0 = silent).
    :param large_model: if True the returned :class:`~yobx.container.ExportArtifact`
        has its :attr:`~yobx.container.ExportArtifact.container` attribute set to
        an :class:`~yobx.container.ExtendedModelContainer`
    :param external_threshold: if ``large_model`` is True, every tensor whose
        element count exceeds this threshold is stored as external data
    :param return_optimize_report: if True, the returned
        :class:`~yobx.container.ExportArtifact` has its
        :attr:`~yobx.container.ExportArtifact.report` attribute populated with
        per-pattern optimization statistics
    :return: :class:`~yobx.container.ExportArtifact` wrapping the exported
        ONNX model together with an :class:`~yobx.container.ExportReport`.

    Example::

        import numpy as np
        import polars as pl
        from yobx.sql import lazyframe_to_onnx
        from yobx.reference import ExtendedReferenceEvaluator

        lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        lf = lf.filter(pl.col("a") > 0).select(
            [(pl.col("a") + pl.col("b")).alias("total")]
        )

        dtypes = {"a": np.float64, "b": np.float64}
        artifact = lazyframe_to_onnx(lf, dtypes)

        ref = ExtendedReferenceEvaluator(artifact)
        a = np.array([1.0, -2.0, 3.0], dtype=np.float64)
        b = np.array([4.0,  5.0, 6.0], dtype=np.float64)
        (total,) = ref.run(None, {"a": a, "b": b})
        # total contains rows where a > 0: [5.0, 9.0]

    .. note::

        ``GROUP BY`` aggregations are computed over the **whole filtered
        dataset** (same limitation as :func:`~yobx.sql.convert.sql_to_onnx`).
        True SQL group-by semantics (one output row per unique key) would
        require an ONNX ``Loop`` or custom kernel and are not yet supported.
    """
    plan_str = lf.explain()
    parsed = _parse_polars_plan(plan_str)
    query = _plan_to_sql(parsed)
    return sql_to_onnx(
        query,
        input_dtypes,
        target_opset=target_opset,
        builder_cls=builder_cls,
        filename=filename,
        verbose=verbose,
        large_model=large_model,
        external_threshold=external_threshold,
        return_optimize_report=return_optimize_report,
    )
