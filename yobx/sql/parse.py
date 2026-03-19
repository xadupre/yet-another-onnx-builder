"""
SQL parser — translates a SQL string into a structured list of
:class:`SqlOperation` objects.

The parser handles a small but useful subset of SQL:

* ``SELECT`` — column selection and simple arithmetic expressions
  (``+``, ``-``, ``*``, ``/``).
* ``WHERE`` / filter — comparison predicates
  (``=``, ``<``, ``>``, ``<=``, ``>=``, ``<>``/``!=``) combined with
  ``AND`` / ``OR``.
* ``GROUP BY`` — column grouping (aggregations ``SUM``, ``COUNT``, ``AVG``,
  ``MIN``, ``MAX`` in the ``SELECT`` list are recognised).
* ``JOIN`` — ``[INNER] JOIN … ON col1 = col2`` linking two input tables.
* Subqueries in the ``FROM`` clause:
  ``SELECT … FROM (SELECT … FROM table) [AS alias]``.

All names are case-insensitive; they are normalised to lower-case by
:func:`parse_sql`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Small expression dataclasses used inside operations
# ---------------------------------------------------------------------------


@dataclass
class ColumnRef:
    """A bare column reference, optionally qualified: ``table.column``."""

    column: str
    table: Optional[str] = None

    def __str__(self) -> str:
        if self.table:
            return f"{self.table}.{self.column}"
        return self.column


@dataclass
class Literal:
    """A scalar literal value (number or quoted string)."""

    value: object  # int | float | str

    def __str__(self) -> str:
        if isinstance(self.value, str):
            return f"'{self.value}'"
        return str(self.value)


@dataclass
class BinaryExpr:
    """A binary expression: ``left op right``."""

    left: object  # ColumnRef | Literal | BinaryExpr | AggExpr
    op: str  # '+' | '-' | '*' | '/' | '=' | '<' | '>' | '<=' | '>=' | '<>'
    right: object

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class AggExpr:
    """An aggregation expression: ``SUM(col)``, ``COUNT(*)``, etc."""

    func: str  # 'sum' | 'count' | 'avg' | 'min' | 'max'
    arg: object  # ColumnRef | Literal | '*' (string)

    def __str__(self) -> str:
        return f"{self.func.upper()}({self.arg})"


@dataclass
class FuncCallExpr:
    """A call to a user-defined (custom) function: ``func_name(arg1, arg2, …)``."""

    func: str  # function name as it appears in the SQL string (lower-cased)
    args: List[object]  # positional arguments (ColumnRef | Literal | BinaryExpr | AggExpr)

    def __str__(self) -> str:
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.func}({args_str})"


@dataclass
class SelectItem:
    """One item in the SELECT list: an expression with an optional alias."""

    expr: object  # ColumnRef | Literal | BinaryExpr | AggExpr
    alias: Optional[str] = None

    def output_name(self) -> str:
        """Return the alias, or derive a name from the expression."""
        if self.alias:
            return self.alias
        if isinstance(self.expr, ColumnRef):
            return self.expr.column
        if isinstance(self.expr, AggExpr):
            arg = self.expr.arg
            col = arg.column if isinstance(arg, ColumnRef) else str(arg)
            return f"{self.expr.func}_{col}"
        return str(self.expr)


@dataclass
class Condition:
    """A WHERE predicate, either a leaf comparison or a compound AND / OR."""

    left: object  # ColumnRef | Literal | Condition
    op: str  # '=' | '<' | '>' | '<=' | '>=' | '<>' | 'and' | 'or'
    right: object

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


# ---------------------------------------------------------------------------
# SqlOperation hierarchy
# ---------------------------------------------------------------------------


@dataclass
class SqlOperation:
    """Base class for all SQL operations produced by :func:`parse_sql`."""


@dataclass
class SelectOp(SqlOperation):
    """
    Represents the ``SELECT`` clause.

    :param items: the list of :class:`SelectItem` objects to compute.
    :param distinct: ``True`` when the query contains ``SELECT DISTINCT``.
    """

    items: List[SelectItem] = field(default_factory=list)
    distinct: bool = False


@dataclass
class FilterOp(SqlOperation):
    """
    Represents a ``WHERE`` clause.

    :param condition: the parsed predicate tree.
    """

    condition: Condition = field(
        default_factory=lambda: Condition(Literal(True), "=", Literal(True))
    )


@dataclass
class GroupByOp(SqlOperation):
    """
    Represents a ``GROUP BY`` clause.

    :param columns: the column names to group by.
    """

    columns: List[str] = field(default_factory=list)


@dataclass
class JoinOp(SqlOperation):
    """
    Represents a ``JOIN`` clause.

    :param right_table: the name of the right-hand table being joined.
    :param left_key: column name from the left table used in the equi-join.
    :param right_key: column name from the right table used in the equi-join.
    :param join_type: ``'inner'`` (default), ``'left'``, ``'right'``,
        or ``'full'``.
    """

    right_table: str = ""
    left_key: str = ""
    right_key: str = ""
    join_type: str = "inner"


@dataclass
class ParsedQuery:
    """
    The result of :func:`parse_sql`.

    :param operations: ordered list of :class:`SqlOperation` objects derived
        from the SQL string.  The order reflects the logical execution
        sequence: ``JoinOp`` (if any) → ``FilterOp`` (if any) → ``GroupByOp``
        (if any) → ``SelectOp``.
    :param from_table: the primary (left) table name from the ``FROM`` clause,
        or the alias of the subquery when ``subquery`` is set.
    :param columns: all column names referenced in the query, in the order
        they appear (deduped).
    :param subquery: when the ``FROM`` clause contains a sub-select
        (``FROM (SELECT …)``), this holds the parsed inner query; otherwise
        ``None``.
    """

    operations: List[SqlOperation] = field(default_factory=list)
    from_table: str = ""
    columns: List[str] = field(default_factory=list)
    subquery: Optional["ParsedQuery"] = None


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

_KEYWORDS = {
    "select",
    "distinct",
    "from",
    "where",
    "group",
    "by",
    "join",
    "inner",
    "left",
    "right",
    "full",
    "outer",
    "on",
    "and",
    "or",
    "not",
    "as",
    "order",
    "having",
    "limit",
    "sum",
    "count",
    "avg",
    "min",
    "max",
}

_COMPARISON_OPS = {"=", "<", ">", "<=", ">=", "<>", "!="}
_ARITH_OPS = {"+", "-", "*", "/"}

# Regex for a single token
_TOKEN_RE = re.compile(
    r"""
    (?P<num>-?\d+(?:\.\d+)?)            # number literal
    |(?P<str>'[^']*'|"[^"]*")          # string literal
    |(?P<op><>|!=|<=|>=|[=<>()+\-*/,]) # operators and punctuation
    |(?P<id>[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)  # identifier (may contain dot)
    |\s+                                # whitespace (skip)
    """,
    re.VERBOSE,
)


def _tokenize(sql: str) -> List[Tuple[str, str]]:
    """
    Return a list of ``(kind, value)`` pairs from *sql*.

    *kind* is one of ``"num"``, ``"str"``, ``"op"``, ``"id"``.
    """
    tokens: List[Tuple[str, str]] = []
    for m in _TOKEN_RE.finditer(sql):
        if m.lastgroup is None:
            continue
        kind = m.lastgroup
        val = m.group(kind)
        if kind == "id":
            val = val.lower()
        elif kind == "str":
            val = val[1:-1]  # strip quotes
            kind = "str"
        tokens.append((kind, val))
    return tokens


# ---------------------------------------------------------------------------
# Recursive descent parser
# ---------------------------------------------------------------------------


class _Parser:
    """Simple recursive-descent parser for a subset of SQL."""

    def __init__(self, tokens: List[Tuple[str, str]]):
        self._tokens = tokens
        self._pos = 0

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _peek(self) -> Optional[Tuple[str, str]]:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _peek_value(self) -> Optional[str]:
        tok = self._peek()
        return tok[1] if tok else None

    def _consume(self) -> Tuple[str, str]:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, value: str) -> None:
        tok = self._consume()
        if tok[1] != value:
            raise ValueError(f"Expected {value!r} but got {tok[1]!r}")

    def _maybe(self, value: str) -> bool:
        if self._peek_value() == value:
            self._consume()
            return True
        return False

    # ------------------------------------------------------------------
    # Expression parsers
    # ------------------------------------------------------------------

    def _parse_primary(self) -> object:
        """Parse a primary expression: literal, column ref, aggregation, or function call."""
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of input in expression")

        kind, val = tok

        # Aggregation function call
        if kind == "id" and val in ("sum", "count", "avg", "min", "max"):
            self._consume()
            self._expect("(")
            if self._peek_value() == "*":
                self._consume()
                arg: object = "*"
            else:
                arg = self._parse_expr()
            self._expect(")")
            return AggExpr(func=val, arg=arg)

        # User-defined function call: any identifier immediately followed by '('
        if kind == "id" and val not in _KEYWORDS:
            # Look ahead one token to check for '('
            next_pos = self._pos + 1
            if next_pos < len(self._tokens) and self._tokens[next_pos] == ("op", "("):
                self._consume()  # consume function name
                self._consume()  # consume '('
                args: List[object] = []
                if self._peek_value() != ")":
                    args.append(self._parse_expr())
                    while self._maybe(","):
                        args.append(self._parse_expr())
                self._expect(")")
                return FuncCallExpr(func=val, args=args)
            # Plain identifier — fall through to ColumnRef below
            self._consume()
            parts = val.split(".")
            if len(parts) == 2:
                return ColumnRef(column=parts[1], table=parts[0])
            return ColumnRef(column=parts[0])

        # Parenthesised expression
        if kind == "op" and val == "(":
            self._consume()
            expr = self._parse_expr()
            self._expect(")")
            return expr

        # Numeric literal
        if kind == "num":
            self._consume()
            return Literal(float(val) if "." in val else int(val))

        # String literal
        if kind == "str":
            self._consume()
            return Literal(val)

        raise ValueError(f"Unexpected token {tok!r} in expression")

    def _parse_multiplicative(self) -> object:
        left = self._parse_primary()
        while self._peek_value() in ("*", "/"):
            op = self._consume()[1]
            right = self._parse_primary()
            left = BinaryExpr(left=left, op=op, right=right)
        return left

    def _parse_additive(self) -> object:
        left = self._parse_multiplicative()
        while self._peek_value() in ("+", "-"):
            op = self._consume()[1]
            right = self._parse_multiplicative()
            left = BinaryExpr(left=left, op=op, right=right)
        return left

    def _parse_expr(self) -> object:
        """Parse an arithmetic expression."""
        return self._parse_additive()

    def _parse_comparison(self) -> Condition:
        """Parse a single comparison predicate."""
        left = self._parse_expr()
        op_tok = self._peek()
        if op_tok and op_tok[1] in _COMPARISON_OPS:
            self._consume()
            op = op_tok[1]
            if op == "!=":
                op = "<>"
            right = self._parse_expr()
            return Condition(left=left, op=op, right=right)
        # bare expression treated as truth-y
        return Condition(left=left, op="=", right=Literal(True))

    def _parse_and(self) -> Condition:
        left = self._parse_comparison()
        while self._peek_value() == "and":
            self._consume()
            right = self._parse_comparison()
            left = Condition(left=left, op="and", right=right)
        return left

    def _parse_condition(self) -> Condition:
        left = self._parse_and()
        while self._peek_value() == "or":
            self._consume()
            right = self._parse_and()
            left = Condition(left=left, op="or", right=right)
        return left

    # ------------------------------------------------------------------
    # Clause parsers
    # ------------------------------------------------------------------

    def _parse_select_list(self) -> Tuple[bool, List[SelectItem]]:
        distinct = self._maybe("distinct")
        items: List[SelectItem] = []
        while True:
            expr = self._parse_expr()
            alias: Optional[str] = None
            if self._peek_value() == "as":
                self._consume()
                _, alias = self._consume()
            elif (
                self._peek() is not None
                and self._peek()[0] == "id"  # type: ignore
                and self._peek()[1] not in _KEYWORDS  # type: ignore
                and self._peek()[1] not in (",",)  # type: ignore
            ):
                _, alias = self._consume()
            items.append(SelectItem(expr=expr, alias=alias))
            if not self._maybe(","):
                break
        return distinct, items

    def _parse_from(self) -> Tuple[str, Optional["ParsedQuery"]]:
        """Parse the FROM clause.

        Returns a ``(table_name, subquery)`` pair.  When the FROM clause
        contains a parenthesised sub-select the first element is the alias
        (may be an empty string) and the second is the parsed inner query.
        For a plain table name the second element is ``None``.
        """
        self._expect("from")
        # Subquery: FROM (SELECT ...)
        if self._peek() is not None and self._peek() == ("op", "("):
            self._consume()  # consume '('
            sub_pq = self._parse_query()
            self._expect(")")
            # Optional alias: [AS] alias_name
            alias = ""
            if self._maybe("as"):
                _, alias = self._consume()
            elif (
                self._peek() is not None
                and self._peek()[0] == "id"  # type: ignore[index]
                and self._peek()[1] not in _KEYWORDS  # type: ignore[index]
            ):
                _, alias = self._consume()
            return alias, sub_pq
        # Plain table name
        _, table = self._consume()
        return table, None

    def _parse_join(self, join_type: str = "inner") -> Optional[JoinOp]:
        """Parse ``[INNER|LEFT|RIGHT|FULL] JOIN … ON …``."""
        _, right_table = self._consume()
        self._expect("on")
        left_expr = self._parse_expr()
        self._expect("=")
        right_expr = self._parse_expr()

        # Resolve key names
        if isinstance(left_expr, ColumnRef):
            left_key = left_expr.column
        else:
            left_key = str(left_expr)

        if isinstance(right_expr, ColumnRef):
            right_key = right_expr.column
        else:
            right_key = str(right_expr)

        return JoinOp(
            right_table=right_table, left_key=left_key, right_key=right_key, join_type=join_type
        )

    def _parse_query(self) -> "ParsedQuery":
        """Parse a full ``SELECT … FROM … [JOIN] [WHERE] [GROUP BY]`` statement.

        This method is called recursively for subqueries.
        """
        self._expect("select")
        distinct, select_items = self._parse_select_list()

        from_table, subquery = self._parse_from()

        operations: List[SqlOperation] = []

        # JOIN
        join_type = "inner"
        while self._peek_value() in ("inner", "left", "right", "full", "join"):
            val = self._peek_value()
            if val in ("inner", "left", "right", "full"):
                self._consume()
                join_type = val
                if self._peek_value() == "outer":
                    self._consume()
            if self._peek_value() == "join":
                self._consume()
                join_op = self._parse_join(join_type=join_type)
                if join_op:
                    operations.append(join_op)
                join_type = "inner"

        # WHERE
        if self._peek_value() == "where":
            self._consume()
            cond = self._parse_condition()
            operations.append(FilterOp(condition=cond))

        # GROUP BY
        if self._peek_value() == "group":
            self._consume()
            self._expect("by")
            group_cols: List[str] = []
            while True:
                expr = self._parse_expr()
                if isinstance(expr, ColumnRef):
                    group_cols.append(expr.column)
                else:
                    group_cols.append(str(expr))
                if not self._maybe(","):
                    break
            operations.append(GroupByOp(columns=group_cols))

        # SELECT always goes last in the execution order
        operations.append(SelectOp(items=select_items, distinct=distinct))

        # Collect all referenced column names
        columns = _collect_columns(operations)

        return ParsedQuery(
            operations=operations, from_table=from_table, columns=columns, subquery=subquery
        )

    def parse(self) -> "ParsedQuery":
        """Parse the token stream and return a :class:`ParsedQuery`."""
        return self._parse_query()


# ---------------------------------------------------------------------------
# Column collector helper
# ---------------------------------------------------------------------------


def _collect_columns(operations: List[SqlOperation]) -> List[str]:
    """Walk the operation tree and collect every distinct column name."""
    seen: List[str] = []

    def _visit(node: object) -> None:
        if isinstance(node, ColumnRef):
            if node.column not in seen:
                seen.append(node.column)
        elif isinstance(node, BinaryExpr):
            _visit(node.left)
            _visit(node.right)
        elif isinstance(node, AggExpr):
            if isinstance(node.arg, ColumnRef):
                _visit(node.arg)
        elif isinstance(node, FuncCallExpr):
            for arg in node.args:
                _visit(arg)
        elif isinstance(node, Condition):
            _visit(node.left)
            _visit(node.right)
        elif isinstance(node, SelectItem):
            _visit(node.expr)
        elif isinstance(node, SelectOp):
            for item in node.items:
                _visit(item)
        elif isinstance(node, FilterOp):
            _visit(node.condition)
        elif isinstance(node, GroupByOp):
            for col in node.columns:
                if col not in seen:
                    seen.append(col)
        elif isinstance(node, JoinOp):
            for col in (node.left_key, node.right_key):
                if col not in seen:
                    seen.append(col)

    for op in operations:
        _visit(op)

    return seen


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_sql(query: str) -> ParsedQuery:
    """
    Parse a SQL *query* string and return a :class:`ParsedQuery`.

    The parser handles:

    * ``SELECT [DISTINCT] expr [AS alias], …``
    * ``FROM table``
    * ``FROM (SELECT …) [AS alias]`` — subquery in the ``FROM`` clause
    * ``[INNER|LEFT|RIGHT|FULL [OUTER]] JOIN table ON col = col``
    * ``WHERE condition [AND|OR condition] …``
    * ``GROUP BY col, …``

    Column names in the returned operations are normalised to *lower-case*.

    :param query: the SQL query string to parse.
    :return: a :class:`ParsedQuery` with an ``operations`` list and a
        ``columns`` list of all referenced column names.

    .. runpython::
        :showcode:

        from yobx.sql.parse import parse_sql

        pq = parse_sql("SELECT a, b FROM t WHERE a > 0")
        for op in pq.operations:
            print(type(op).__name__, op)
    """
    tokens = _tokenize(query)
    parser = _Parser(tokens)
    return parser.parse()
