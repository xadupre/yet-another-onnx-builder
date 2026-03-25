"""
SQL / DataFrame / Polars coverage data.

Provides :func:`get_sql_coverage` which returns structured coverage information
for all three input paths (SQL strings, DataFrame tracer, Polars LazyFrame) as
lists of ``(construct, status, notes)`` tuples.  The coverage page in the
documentation calls this function to build its tables programmatically.
"""

from __future__ import annotations

from typing import List, Tuple

# Status symbols
_SUPPORTED = "✔ supported"
_PARTIAL = "⚠ partial"
_UNSUPPORTED = "✘ not supported"

# (construct, status, notes)
CoverageRow = Tuple[str, str, str]

SQL_COVERAGE: List[CoverageRow] = [
    ("``SELECT col``", _SUPPORTED, "column pass-through via ``Identity``"),
    (
        "``SELECT expr AS alias``",
        _SUPPORTED,
        "arithmetic: ``Add``, ``Sub``, ``Mul``, ``Div``",
    ),
    (
        "``SELECT AGG(col)``",
        _SUPPORTED,
        "``SUM``, ``AVG``, ``MIN``, ``MAX``, ``COUNT``",
    ),
    (
        "``WHERE condition``",
        _SUPPORTED,
        "comparisons + ``AND`` / ``OR``",
    ),
    (
        "``GROUP BY cols``",
        _PARTIAL,
        "whole-dataset aggregation only; no per-group rows",
    ),
    (
        "``[INNER|LEFT|RIGHT|FULL] JOIN … ON col = col``",
        _SUPPORTED,
        "equi-join on a single key column",
    ),
    (
        "``SELECT DISTINCT``",
        _UNSUPPORTED,
        "parsed but raises ``NotImplementedError``",
    ),
    ("``HAVING``", _UNSUPPORTED, "not yet implemented"),
    ("``ORDER BY``", _UNSUPPORTED, "not yet implemented"),
    ("``LIMIT``", _UNSUPPORTED, "not yet implemented"),
    ("Subqueries", _UNSUPPORTED, "not yet implemented"),
    (
        "String equality (``WHERE col = 'val'``)",
        _UNSUPPORTED,
        "requires special ONNX string handling",
    ),
]

DATAFRAME_COVERAGE: List[CoverageRow] = [
    ("``df[\"col\"]``", _SUPPORTED, "column access"),
    ("``df.filter(condition)``", _SUPPORTED, "maps to ``WHERE``"),
    ("``df.select([series, …])``", _SUPPORTED, "maps to ``SELECT``"),
    ("``df.assign(name=series)``", _SUPPORTED, "maps to ``SELECT … AS name``"),
    ("``df.groupby(cols)``", _PARTIAL, "whole-dataset aggregation only"),
    (
        "Series arithmetic (``+``, ``-``, ``*``, ``/``)",
        _SUPPORTED,
        "``Add``, ``Sub``, ``Mul``, ``Div``",
    ),
    (
        "Series comparisons (``>``, ``<``, ``>=``, ``<=``, ``==``, ``!=``)",
        _SUPPORTED,
        "``Greater``, ``Less``, ``Equal``, …",
    ),
    (
        "``.sum()`` / ``.mean()`` / ``.min()`` / ``.max()`` / ``.count()``",
        _SUPPORTED,
        "``ReduceSum``, ``ReduceMean``, ``ReduceMin``, ``ReduceMax``",
    ),
    ("``cond1 & cond2`` / ``cond1 | cond2``", _SUPPORTED, "``And``, ``Or``"),
    ("``series.alias(\"name\")``", _SUPPORTED, "output rename"),
    (
        "Conditional branches (``if``/``else``)",
        _UNSUPPORTED,
        "tracing captures one execution path only",
    ),
]

POLARS_COVERAGE: List[CoverageRow] = [
    (
        "``lf.select([…])``",
        _SUPPORTED,
        "column selection and arithmetic expressions",
    ),
    (
        "``lf.filter(condition)``",
        _SUPPORTED,
        "comparison and boolean predicates",
    ),
    (
        "``lf.group_by(cols).agg([…])``",
        _PARTIAL,
        "whole-dataset aggregation only",
    ),
    (
        "Arithmetic (``+``, ``-``, ``*``, ``/``)",
        _SUPPORTED,
        "inlined into ``SELECT`` expressions",
    ),
    (
        "Comparisons (``>``, ``<``, ``>=``, ``<=``, ``==``, ``!=``)",
        _SUPPORTED,
        "``WHERE`` predicates",
    ),
    (
        "``&`` / ``|`` compound predicates",
        _SUPPORTED,
        "``AND`` / ``OR`` in ``WHERE``",
    ),
    ("``.alias(\"name\")``", _SUPPORTED, "output rename"),
    (
        "Aggregation methods (``.sum()``, ``.mean()``, ``.min()``, ``.max()``, ``.count()``)",
        _SUPPORTED,
        "``ReduceSum``, ``ReduceMean``, ``ReduceMin``, ``ReduceMax``",
    ),
    ("``lf.join(…)``", _UNSUPPORTED, "not yet implemented"),
    ("``lf.sort(…)``", _UNSUPPORTED, "not yet implemented"),
    ("``lf.limit(…)`` / ``lf.head(…)``", _UNSUPPORTED, "not yet implemented"),
    ("``lf.unique(…)``", _UNSUPPORTED, "not yet implemented"),
]


def _rst_table(rows: List[CoverageRow], col1_header: str) -> str:
    """Render *rows* as a RST list-table string.

    :param rows: list of ``(construct, status, notes)`` tuples.
    :param col1_header: header text for the first column.
    :return: RST source string for a ``list-table`` directive.
    """
    lines = [
        ".. list-table::",
        "    :header-rows: 1",
        "    :widths: 35 15 50",
        "",
        f"    * - {col1_header}",
        "      - Status",
        "      - Notes",
    ]
    for construct, status, notes in rows:
        lines.append(f"    * - {construct}")
        lines.append(f"      - {status}")
        lines.append(f"      - {notes}")
    lines.append("")
    return "\n".join(lines)


def get_sql_coverage(section: str = "sql") -> str:
    """Return a RST ``list-table`` for one of the three coverage sections.

    :param section: one of ``"sql"``, ``"dataframe"``, or ``"polars"``.
    :return: RST source string ready to be printed inside a
        ``.. runpython::`` block with ``:rst:`` enabled.
    :raises ValueError: if *section* is not one of the recognised values.
    """
    if section == "sql":
        return _rst_table(SQL_COVERAGE, "SQL construct")
    if section == "dataframe":
        return _rst_table(DATAFRAME_COVERAGE, "Operation")
    if section == "polars":
        return _rst_table(POLARS_COVERAGE, "Polars operation")
    raise ValueError(f"Unknown section {section!r}; expected 'sql', 'dataframe', or 'polars'.")
