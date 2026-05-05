from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Union
import itertools
import torch
from ...xexpressions import evaluate_expression, simplify_expression

# ---------------------------------------------------------------------------
# Registry of conditions known to be True (populated by _handle_check during
# tracing via torch._check interception).  Symbolic TracingBool values whose
# expression matches an entry resolve to True in __bool__.
# ---------------------------------------------------------------------------

# Counter used to generate unique names for unnamed dynamic-dimension hints
# such as ``torch.export.Dim.DYNAMIC`` or ``torch.export.Dim.AUTO``.
_unnamed_dim_counter: itertools.count = itertools.count()


def _dim_spec_to_name(dim: Any) -> str:
    """Converts a dynamic-shape dimension specification to a string name.

    Accepts plain strings, named :class:`torch.export.Dim` instances, and
    unnamed :class:`torch.export.Dim` sentinel values (``Dim.DYNAMIC``,
    ``Dim.AUTO``).

    :param dim: A ``str``, a named ``torch.export.Dim`` object, or an unnamed
        ``_DimHint`` value.

    Returns:
        A string suitable for use as a :class:`TracingInt` symbolic name.
    """
    if isinstance(dim, str):
        return dim
    # Named torch.export.Dim objects expose their user-given name via __name__.
    name = getattr(dim, "__name__", None)
    if isinstance(name, str):
        return name
    # Unnamed dynamic hints (torch.export.Dim.DYNAMIC / AUTO) – treat each
    # occurrence as an independent symbolic dimension.
    return f"_dyn_{next(_unnamed_dim_counter)}"


_known_true_conditions: Set[str] = set()


def register_condition(cond: "TracingBool") -> None:
    """Registers *cond* as a condition known to be True during tracing.

    Only symbolic (string-valued) :class:`TracingBool` instances are
    registered; concrete booleans are ignored.

    :param cond: A :class:`TracingBool` that has been asserted via
        :func:`torch._check` and is therefore known to hold.
    """
    if isinstance(cond, TracingBool) and isinstance(cond.value, str):
        _known_true_conditions.add(cond.value)


def clear_conditions() -> None:
    """Clears the registry of known-True conditions.

    Called at the start of each :meth:`GraphTracer.trace` invocation so that
    constraints do not leak between independent traces.
    """
    _known_true_conditions.clear()


# Mapping from comparison-operator string to the corresponding callable.
# Defined once at module level to avoid recreating the dict on every _cmp call.
_COMPARISON_OPS: Dict[str, Callable[[int, int], bool]] = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "!=": lambda a, b: a != b,
}


def _negate_condition(cond: str) -> Optional[str]:
    """Returns the logical negation of a simple equality/inequality condition.

    Only handles the ``==`` ↔ ``!=`` relationship (sufficient for the common
    ``if x.numel() == 0:`` / ``torch._check(x.numel() != 0)`` pairing).
    Returns ``None`` for all other patterns.  Handles both the ``"(E==0)"``
    and ``"E==0"`` forms since :func:`simplify_expression` strips outer parens.

    :param cond: A condition string such as ``"E==0"`` or ``"E!=0"``.

    Returns:
        The negated string, or ``None`` if negation cannot be determined.
    """
    if "!=" in cond:
        return cond.replace("!=", "==", 1)
    if "==" in cond:
        # Guard against matching the `=` inside `<=` or `>=`
        idx = cond.index("==")
        if idx > 0 and cond[idx - 1] in "<>":
            return None
        return cond.replace("==", "!=", 1)
    return None


def _is_positivity_condition(cond: str) -> bool:
    """Returns True if *cond* is a simple positivity constraint.

    Recognises ``"expr>0"`` and ``"expr>=N"`` (N ≥ 1) patterns — the same
    forms that :func:`_can_prove_expr_nonzero` uses to build its variable
    context.  These conditions arise when comparing a symbolic dimension or
    expression against zero, e.g. ``x.shape[0] > 0``.

    Uses :func:`~yobx.xexpressions.evaluate_expression` to evaluate the
    right-hand side so that constant expressions are handled uniformly.

    :param cond: A condition string such as ``"d0>0"`` or ``"_dyn_1>=1"``.

    Returns:
        ``True`` when the condition asserts that some expression is strictly
        positive, ``False`` otherwise.
    """
    if ">=" in cond:
        parts = cond.split(">=", 1)
        try:
            rhs = evaluate_expression(parts[1].strip(), {})
            return rhs >= 1
        except (NameError, TypeError, SyntaxError, ValueError):
            return False
    if ">" in cond:
        parts = cond.split(">", 1)
        try:
            rhs = evaluate_expression(parts[1].strip(), {})
            return rhs == 0
        except (NameError, TypeError, SyntaxError, ValueError):
            return False
    return False


def _can_prove_expr_nonzero_from_neq_conditions(expr: str) -> bool:
    """Returns True if *expr* is provably non-zero from registered ``!=0`` conditions.

    Checks whether *expr* appears as a multiplicative factor in any registered
    condition of the form ``"P!=0"``.  If ``"P!=0"`` is known-true and *expr*
    is a factor of P, then *expr* must also be non-zero.

    This handles the :class:`~yobx.torch.testing._model_eval_cases.ControlFlowNumelZero5`
    pattern::

        torch._check(x.numel() != 0)   # registers "10*d0*d2!=0"
        if x.shape[0] != 0: ...         # needs to prove "d0!=0"

    :param expr: A simple symbolic expression such as ``"_dyn_0"``.

    Returns:
        ``True`` if *expr* is a multiplicative factor of a known-nonzero
        product, ``False`` otherwise.
    """
    expr_stripped = expr.strip()
    for cond in _known_true_conditions:
        if "!=" not in cond:
            continue
        idx = cond.index("!=")
        lhs = cond[:idx].strip()
        rhs = cond[idx + 2 :].strip()
        if rhs != "0":
            continue
        # Remove outer parentheses if present.
        if lhs.startswith("(") and lhs.endswith(")"):
            lhs = lhs[1:-1]
        # Check whether expr_stripped appears as a multiplicative factor.
        # Numel expressions are always pure products (no additions), so
        # splitting on "*" reliably extracts individual factors.
        factors = [f.strip() for f in lhs.split("*")]
        if expr_stripped in factors:
            return True
    return False


def _can_prove_expr_nonzero(expr: str) -> bool:
    """Returns True if *expr* is provably non-zero given the known conditions.

    Builds a context mapping each variable known to be strictly positive
    (i.e. ``"var>0"`` or ``"var>=1"`` appears in
    :data:`_known_true_conditions`) to ``1`` — the smallest positive integer
    value — then delegates to :func:`~yobx.xexpressions.evaluate_expression`.
    If the expression evaluates to a non-zero integer value, the expression is
    provably non-zero.  This covers patterns such as ``10*_dyn_0*_dyn_1`` when
    ``torch._check(x.shape[0] > 0)`` and ``torch._check(x.shape[2] > 0)``
    have previously been called.

    If any variable in *expr* is absent from the context (i.e. its sign is
    unknown), :func:`~yobx.xexpressions.evaluate_expression` raises
    :exc:`NameError` and the function conservatively returns ``False``.

    :param expr: A symbolic expression string such as ``"10*_dyn_0*_dyn_1"``.

    Returns:
        ``True`` if the expression evaluates to a non-zero integer given the
        minimal positive assignment for each constrained variable.
    """
    # Build context: map each variable known to be strictly positive to 1.
    ctx: Dict[str, int] = {}
    for cond in _known_true_conditions:
        # "var>=N" where N >= 1  →  var is at least 1, map to 1.
        if ">=" in cond:
            parts = cond.split(">=", 1)
            var = parts[0].strip()
            try:
                if int(parts[1].strip()) >= 1:
                    ctx[var] = 1
            except ValueError:
                pass
        # "var>N" where N == 0  →  var is at least 1, map to 1.
        elif ">" in cond:
            parts = cond.split(">", 1)
            var = parts[0].strip()
            try:
                if int(parts[1].strip()) == 0:
                    ctx[var] = 1
            except ValueError:
                pass
    if not ctx:
        return False
    try:
        val = evaluate_expression(expr, ctx)
        return val != 0
    except (NameError, TypeError, SyntaxError, ValueError):
        return False


class TracingBool:
    """
    Represents a boolean comparison that may involve symbolic :class:`TracingInt`
    dimensions and therefore cannot always be evaluated at trace time.

    :param value: Either a concrete :class:`bool` or a :class:`str` containing
        a symbolic expression such as ``"(batch==4)"``.

    - ``TracingBool(True)`` / ``TracingBool(False)`` — concrete result.
    - ``TracingBool("(n==4)")`` — symbolic; cannot be used as a Python bool.

    Symbolic :class:`TracingBool` instances can be resolved to a concrete
    ``True`` when the condition has been registered via
    :func:`register_condition` (typically by the :func:`torch._check`
    interception in :class:`~yobx.torch.new_tracing.tracer.GraphTracer`).
    The logical negation of a registered condition resolves to ``False``
    (e.g. if ``(E!=0)`` is known-true, then ``(E==0)`` resolves to ``False``).

    When this instance is produced by a comparison on a
    :class:`TracingInt` that carries an FX node (e.g. ``numel() > 0``
    during graph tracing), the corresponding FX comparison node is stored
    in :attr:`_node`.  :meth:`~yobx.torch.new_tracing.tracer.GraphTracer._handle_cond`
    uses this to pass a proper bool tensor node as the ``torch.cond`` predicate
    instead of the raw :class:`TracingBool` object.
    """

    def __init__(self, value: Union[bool, str]):
        self.value = value
        # Optional FX node that produces this boolean value as a 0-dim bool
        # tensor.  Set by :meth:`TracingInt._cmp` when the left-hand operand
        # carries an FX node (see :attr:`TracingInt._node`).
        self._node: Any = None

    def __repr__(self) -> str:
        return f"TracingBool({self.value!r})"

    def __str__(self) -> str:
        return str(self.value)

    def __bool__(self) -> bool:
        if isinstance(self.value, bool):
            return self.value
        if self.value in _known_true_conditions:
            return True
        # If the logical negation of this condition is known to be True,
        # this condition must be False.  This handles the common pattern
        # ``torch._check(x.numel() != 0)`` followed by ``if x.numel() == 0:``.
        neg = _negate_condition(self.value)
        if neg is not None and neg in _known_true_conditions:
            return False
        # Handle ``E == 0`` where E is a product of factors that are each
        # individually known to be strictly positive (e.g.
        # ``torch._check(x.shape[0] > 0)`` and
        # ``torch._check(x.shape[2] > 0)`` registered before
        # ``if x.numel() == 0:``).  If every factor of E is provably > 0
        # then E > 0 and the condition ``E == 0`` is False.
        if "==" in self.value:
            idx = self.value.index("==")
            if idx > 0 and self.value[idx - 1] not in "<>":
                lhs = self.value[:idx].strip()
                rhs_str = self.value[idx + 2 :].strip()
                if rhs_str == "0" and _can_prove_expr_nonzero(lhs):
                    return False
        # Handle ``E != 0`` where E is a factor in a registered ``P!=0``
        # product condition.  This covers the ControlFlowNumelZero5 pattern:
        #
        #     torch._check(x.numel() != 0)   # registers "10*d0*d2!=0"
        #     if x.shape[0] != 0 and x.shape[2] != 0:  # needs "d0!=0", "d2!=0"
        #
        # If the registered "P!=0" has E as one of its multiplicative factors,
        # then E != 0 must hold.
        if "!=" in self.value:
            idx = self.value.index("!=")
            lhs = self.value[:idx].strip()
            rhs_str = self.value[idx + 2 :].strip()
            if rhs_str == "0" and _can_prove_expr_nonzero_from_neq_conditions(lhs):
                return True
        # Handle simple positivity conditions (e.g. ``"var > 0"`` or ``"var >= 1"``)
        # that arise during Python ``and``/``or`` short-circuit evaluation when
        # such a condition is part of a compound expression passed to
        # ``torch._check``.  For example::
        #
        #     torch._check(x.shape[0] > 0 and x.shape[2] > 0)
        #
        # Python evaluates ``bool(TracingBool("var > 0"))`` for the left operand
        # before evaluating the right operand.  Since ``torch._check`` asserts
        # that the whole expression is True, each conjunct is also True.  We
        # self-register the condition and return ``True`` so that the right
        # operand is evaluated and can be registered by :meth:`_handle_check`.
        # A symbolic :class:`TracingBool` can only be produced by comparing a
        # :class:`TracingInt` (which only exists during tracing), so this
        # self-registration is always safe for positivity conditions.
        if _is_positivity_condition(self.value):
            _known_true_conditions.add(self.value)
            return True
        raise ValueError(
            f"TracingBool({self.value!r}) cannot be converted to a Python bool; "
            "the result depends on a symbolic/dynamic dimension. "
            "Use torch._check(condition) to assert the condition holds."
        )

    def __eq__(self, other: Any) -> bool:  # noqa: PYI032
        if isinstance(other, TracingBool):
            return self.value == other.value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)


class TracingInt:
    """
    Represents a single symbolic or concrete integer (e.g. a tensor dimension).

    :param value: Either a concrete :class:`int` (concrete dimension) or a
        :class:`str` (symbolic/dynamic dimension name such as ``"batch"``).

    Examples::

        # Concrete dimension
        d = TracingInt(4)
        print(int(d))       # 4
        print(d + 2)        # TracingInt(6)

        # Symbolic/dynamic dimension
        s = TracingInt("batch")
        print(str(s))       # batch
        print(s + 2)        # TracingInt('(batch+2)')
        print(s == 4)       # TracingBool('(batch==4)')

    When a :class:`TracingInt` is produced from :meth:`TracingTensor.numel`
    during graph tracing, three extra attributes are set so that subsequent
    comparisons (e.g. ``numel > 0``) can emit proper FX nodes:

    * :attr:`_node` — the FX :class:`~torch.fx.Node` that produces this
      integer value as a scalar ``int64`` tensor.
    * :attr:`_tracer` — the :class:`~yobx.torch.new_tracing.tracer.GraphTracer`
      that owns the FX graph.
    * :attr:`_device` — the device string (e.g. ``"cpu"``) of the source
      tensor, used when creating wrapper :class:`TracingTensor` objects for
      FX node metadata.

    When a :class:`TracingInt` comes from a placeholder tensor's shape (set by
    :meth:`~yobx.torch.new_tracing.tracer.GraphTracer.placeholder`), two
    additional attributes allow lazy emission of ``aten.sym_size.int`` nodes:

    * :attr:`_source_node` — the FX placeholder :class:`~torch.fx.Node` of
      the tensor this dimension belongs to.
    * :attr:`_source_dim` — the integer dimension index within that tensor.
    """

    def __init__(self, value: Union[int, str]):
        assert isinstance(value, (int, str)), f"Unexpected type {type(value)} for value"
        self.value = value
        # Optional FX node that produces this integer as a scalar int64 tensor.
        # Set by TracingTensor.numel() when a tracer is active.
        self._node: Any = None
        # Tracer that owns the FX graph referenced by _node.
        self._tracer: Any = None
        # Device of the source tensor (used when creating FX node metadata).
        self._device: Any = None
        # FX node of the placeholder tensor this dimension belongs to.
        # Set by GraphTracer.placeholder() for input tensor shape dims.
        self._source_node: Any = None
        # Dimension index within the source tensor's shape.
        self._source_dim: Optional[int] = None

    @property
    def is_static(self):
        return isinstance(self.value, int)

    def __repr__(self) -> str:
        return f"TracingInt({self.value!r})"

    def __str__(self) -> str:
        return str(self.value)

    def __int__(self) -> int:
        if isinstance(self.value, int):
            return self.value
        raise ValueError(
            f"TracingInt({self.value!r}) has no concrete integer value; "
            "pass a concrete int or check .value"
        )

    def __index__(self) -> int:
        """Enable use as a sequence index (calls :meth:`__int__`)."""
        return int(self)

    def __eq__(self, other: Any) -> Union[bool, TracingBool]:  # type: ignore # noqa: PYI032
        """
        Return a plain :class:`bool` when both sides are concrete; return a
        :class:`TracingBool` when at least one side is symbolic.
        """
        if isinstance(other, TracingInt):
            if isinstance(self.value, int) and isinstance(other.value, int):
                return self.value == other.value
            simp = simplify_expression(f"({self.value}=={other.value})")
            if isinstance(simp, str):
                return TracingBool(simp)
            return bool(simp)
        if isinstance(other, int):
            if isinstance(self.value, int):
                return self.value == other
            simp = simplify_expression(f"({self.value}=={other})")
            if isinstance(simp, str):
                return TracingBool(simp)
            return bool(simp)
        if isinstance(other, str):
            if isinstance(self.value, str) and self.value == other:
                return True
            simp = simplify_expression(f"({self.value}=={other})")
            if isinstance(simp, str):
                return TracingBool(simp)
            return bool(simp)
        raise NotImplementedError(f"Unable to check equality for type {type(other)}.")

    def __hash__(self) -> int:
        return hash(self.value)

    # ------------------------------------------------------------------
    # Arithmetic helpers so TracingInt can participate in shape
    # calculations while keeping symbolic expressions intact.
    # ------------------------------------------------------------------

    def _sym(self) -> str:
        """String representation of *self* for building expressions."""
        return str(self.value)

    def __add__(self, other: Union[int, "TracingInt"]) -> "TracingInt":
        if isinstance(other, TracingInt):
            if isinstance(self.value, int) and isinstance(other.value, int):
                return TracingInt(self.value + other.value)
            return TracingInt(simplify_expression(f"({self._sym()})+({other._sym()})"))
        if isinstance(other, int):
            if isinstance(self.value, int):
                return TracingInt(self.value + other)
            return TracingInt(simplify_expression(f"({self._sym()})+({other})"))
        return NotImplemented

    def __radd__(self, other: int) -> "TracingInt":
        if isinstance(other, int):
            if isinstance(self.value, int):
                return TracingInt(other + self.value)
            return TracingInt(simplify_expression(f"({other})+({self._sym()}"))
        return NotImplemented

    def __sub__(self, other: Union[int, "TracingInt"]) -> "TracingInt":
        if isinstance(other, TracingInt):
            if isinstance(self.value, int) and isinstance(other.value, int):
                return TracingInt(self.value - other.value)
            return TracingInt(simplify_expression(f"({self._sym()})-({other._sym()})"))
        if isinstance(other, int):
            if isinstance(self.value, int):
                return TracingInt(self.value - other)
            return TracingInt(simplify_expression(f"({self._sym()})-({other})"))
        return NotImplemented

    def __rsub__(self, other: int) -> "TracingInt":
        if isinstance(other, int):
            if isinstance(self.value, int):
                return TracingInt(other - self.value)
            return TracingInt(simplify_expression(f"({other})-({self._sym()})"))
        return NotImplemented

    def __mul__(self, other: Union[int, "TracingInt"]) -> "TracingInt":
        if isinstance(other, TracingInt):
            if isinstance(self.value, int) and isinstance(other.value, int):
                return TracingInt(self.value * other.value)
            return TracingInt(simplify_expression(f"({self._sym()})*({other._sym()})"))
        if isinstance(other, int):
            if isinstance(self.value, int):
                return TracingInt(self.value * other)
            return TracingInt(simplify_expression(f"({self._sym()})*({other})"))
        return NotImplemented

    def __rmul__(self, other: int) -> "TracingInt":
        if isinstance(other, int):
            if isinstance(self.value, int):
                return TracingInt(other * self.value)
            return TracingInt(simplify_expression(f"({other}*{self._sym()})"))
        return NotImplemented

    def __floordiv__(self, other: Union[int, "TracingInt"]) -> "TracingInt":
        if isinstance(other, TracingInt):
            if isinstance(self.value, int) and isinstance(other.value, int):
                return TracingInt(self.value // other.value)
            return TracingInt(simplify_expression(f"({self._sym()})//({other._sym()})"))
        if isinstance(other, int):
            if isinstance(self.value, int):
                return TracingInt(self.value // other)
            return TracingInt(simplify_expression(f"({self._sym()})//({other})"))
        return NotImplemented

    def __neg__(self) -> "TracingInt":
        if isinstance(self.value, int):
            return TracingInt(-self.value)
        return TracingInt(simplify_expression(f"-({self._sym()})"))

    # ------------------------------------------------------------------
    # Comparison operators — return a plain bool for concrete values and
    # a TracingBool for symbolic ones.  These are required so that model
    # code such as ``if tensor.shape[0] > 0:`` or assertions via
    # ``torch._check(tensor.shape[0] >= 1)`` work correctly during
    # GraphTracer-based tracing.
    # ------------------------------------------------------------------

    # Maps comparison operator strings to the corresponding aten scalar
    # comparison ops.  Used in _cmp to emit FX nodes when self._node is set.
    _ATEN_SCALAR_CMP_OPS: Dict[str, Any] = {}

    def _cmp(self, op: str, other: Union[int, "TracingInt"]) -> Union[bool, "TracingBool"]:
        """Applies comparison *op* to *self* and *other*.

        Returns a concrete :class:`bool` when both operands are concrete, and
        a :class:`TracingBool` with a symbolic expression string otherwise.
        Symbolic expressions are normalised via :func:`simplify_expression` so
        that the produced strings are consistent with those from :meth:`__eq__`,
        enabling the negation lookup in :meth:`TracingBool.__bool__`.

        When :attr:`_node` and :attr:`_tracer` are set (i.e. this integer was
        produced by :meth:`~yobx.torch.new_tracing.tensor.TracingTensor.numel`
        during graph tracing), a corresponding FX comparison node is emitted
        and stored on the returned :class:`TracingBool` as :attr:`TracingBool._node`.
        :meth:`~yobx.torch.new_tracing.tracer.GraphTracer._handle_cond` then
        uses this node as the ``torch.cond`` predicate rather than the raw
        :class:`TracingBool`.

        :param op: Comparison operator string: ``">"``, ``">="``, ``"<"``, ``"<="``, or ``"!="``.
        :param other: The right-hand-side operand.
        :return: :class:`bool` or :class:`TracingBool`.
        """
        if isinstance(other, TracingInt):
            if isinstance(self.value, int) and isinstance(other.value, int):
                return bool(_COMPARISON_OPS[op](self.value, other.value))
            expr = simplify_expression(f"({self._sym()}{op}{other._sym()})")
            if isinstance(expr, str):
                return TracingBool(expr)
            return TracingBool(bool(expr))
        if isinstance(other, int):
            if isinstance(self.value, int):
                return bool(_COMPARISON_OPS[op](self.value, other))
            expr = simplify_expression(f"({self._sym()}{op}{other})")
            result: Union[bool, TracingBool]
            if isinstance(expr, str):
                result = TracingBool(expr)
            else:
                result = TracingBool(bool(expr))
            # If self carries an FX node (e.g. produced by numel()), emit a
            # comparison FX node and store it on the TracingBool so that
            # _handle_cond can use it as the torch.cond predicate.
            if (
                isinstance(result, TracingBool)
                and self._node is not None
                and self._tracer is not None
            ):
                result._node = self._emit_cmp_node(op, other)
            return result
        return NotImplemented

    def _emit_cmp_node(self, op: str, other: int) -> Any:
        """Emits an FX comparison node for ``self <op> other``.

        Uses the FX node stored in :attr:`_node` (which produces the integer
        value of *self* as a scalar ``int64`` tensor) and emits an ATen
        scalar-comparison op (e.g. ``aten.gt.Scalar``) that produces a 0-dim
        ``bool`` tensor.

        A local import of :class:`~yobx.torch.new_tracing.tensor.TracingTensor`
        is used to avoid a circular import at module level (``tensor.py``
        imports from ``shape.py``).  This is safe because both modules are
        fully initialised by the time any tracing operation calls this method.

        :param op: One of ``">"``, ``">="``, ``"<"``, ``"<="``, ``"!="``.
        :param other: The right-hand scalar integer value.
        :return: The newly created FX :class:`~torch.fx.Node`, or ``None`` if
            the required ATen op is unavailable.
        """
        import traceback

        if not TracingInt._ATEN_SCALAR_CMP_OPS:
            TracingInt._ATEN_SCALAR_CMP_OPS = {
                ">": torch.ops.aten.gt.Scalar,
                ">=": torch.ops.aten.ge.Scalar,
                "<": torch.ops.aten.lt.Scalar,
                "<=": torch.ops.aten.le.Scalar,
                "!=": torch.ops.aten.ne.Scalar,
            }
        aten_op = TracingInt._ATEN_SCALAR_CMP_OPS.get(op)
        if aten_op is None:
            return None
        cmp_node = self._tracer.graph.call_function(aten_op, args=(self._node, other), kwargs={})
        # Set meta["val"] to a 0-dim bool TracingTensor so that the
        # interpreter can determine the output type.  Use a local import to
        # avoid a circular dependency between shape.py and tensor.py.
        from .tensor import TracingTensor

        device = self._device or "cpu"
        cmp_tt = TracingTensor(
            TracingShape(()), dtype=torch.bool, device=device, tracer=self._tracer
        )
        cmp_tt._node = cmp_node
        cmp_node.meta["val"] = cmp_tt
        cmp_node.meta["stack_trace"] = "".join(traceback.format_stack())
        return cmp_node

    def __gt__(self, other: Union[int, "TracingInt"]) -> Union[bool, "TracingBool"]:
        """Returns ``True`` / ``False`` or :class:`TracingBool` for ``self > other``."""
        return self._cmp(">", other)

    def __ge__(self, other: Union[int, "TracingInt"]) -> Union[bool, "TracingBool"]:
        """Returns ``True`` / ``False`` or :class:`TracingBool` for ``self >= other``."""
        return self._cmp(">=", other)

    def __lt__(self, other: Union[int, "TracingInt"]) -> Union[bool, "TracingBool"]:
        """Returns ``True`` / ``False`` or :class:`TracingBool` for ``self < other``."""
        return self._cmp("<", other)

    def __le__(self, other: Union[int, "TracingInt"]) -> Union[bool, "TracingBool"]:
        """Returns ``True`` / ``False`` or :class:`TracingBool` for ``self <= other``."""
        return self._cmp("<=", other)

    def __ne__(self, other: Any) -> Union[bool, "TracingBool"]:  # type: ignore[override]  # noqa: PYI032
        """Returns ``True`` / ``False`` or :class:`TracingBool` for ``self != other``."""
        if isinstance(other, (int, TracingInt)):
            return self._cmp("!=", other)
        return NotImplemented


# Keep TracingDimension as a backward-compatible alias.
TracingDimension = TracingInt


class TracingShape:
    """
    Represents the shape of a :class:`TracingTensor` as an ordered collection
    of :class:`TracingInt` or :class:`int` dimension values.

    Unlike :class:`torch.Size`, individual dimensions may be symbolic
    (:class:`TracingInt` with a ``str`` value).

    :param dims: Iterable of :class:`TracingInt` or :class:`int` values.

    Example::

        shape = TracingShape([TracingInt(4), 16])
        print(shape)              # TracingShape([TracingInt(4), 16])
        print(shape.numel())      # 64
        print(shape.is_concrete)  # True

        sym_shape = TracingShape([TracingInt("n"), 8])
        print(sym_shape.is_concrete)  # False
    """

    def __init__(self, dims: "Sequence[Union[TracingInt, int]]") -> None:
        assert all(
            isinstance(d, (int, TracingInt)) for d in dims
        ), f"Unexpected type in dims {[type(d) for d in dims]}"
        self.dims: Tuple[Union["TracingInt", int], ...] = tuple(dims)

    def __repr__(self) -> str:
        return f"TracingShape({list(self.dims)!r})"

    def __str__(self) -> str:
        items = ", ".join(str(d) for d in self.dims)
        return f"TracingShape([{items}])"

    def __len__(self) -> int:
        return len(self.dims)

    def __iter__(self):
        return iter(self.dims)

    def __getitem__(self, idx):
        return self.dims[idx]

    def __eq__(self, other: Any) -> bool:  # noqa: PYI032
        if isinstance(other, TracingShape):
            return self.dims == other.dims
        if isinstance(other, tuple):
            if len(self.dims) != len(other):
                return False
            return all(d == i for d, i in zip(self.dims, other))
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.dims)

    @property
    def is_concrete(self) -> bool:
        """Return ``True`` if every dimension has a concrete integer value."""
        return all(
            isinstance(d, int) or (isinstance(d, TracingInt) and isinstance(d.value, int))
            for d in self.dims
        )

    def numel(self) -> int:
        """
        Returns the total number of elements (product of all dimensions).

        :raises ValueError: If any dimension is purely symbolic (no concrete
            integer value).
        """
        result = 1
        for d in self.dims:
            if isinstance(d, TracingInt):
                if not isinstance(d.value, int):
                    raise ValueError(
                        f"Cannot compute numel: dimension {d.value!r} has no concrete value"
                    )
                result *= d.value
            else:
                result *= int(d)
        return result

    def to_torch_size(self) -> torch.Size:
        """
        Converts to :class:`torch.Size` (requires all dimensions to be concrete).

        :raises ValueError: If any dimension is purely symbolic.
        """
        if not self.is_concrete:
            raise ValueError(
                "Cannot convert TracingShape with purely symbolic dims to torch.Size; "
                "ensure all TracingInt objects have a concrete (int) value"
            )
        return torch.Size(tuple(int(d) for d in self.dims))

    @classmethod
    def from_existing_shape(
        cls, shape: Tuple[int, ...], dynamic_shapes: Optional[Dict[int, Any]] = None
    ) -> "TracingShape":
        """
        Build a :class:`TracingShape` from a concrete shape tuple, optionally
        making selected dimensions symbolic.

        :param shape: The concrete shape (e.g. from ``tensor.shape``).
        :param dynamic_shapes: An optional mapping from *dimension index* to
            a symbolic name.  Each value may be a plain ``str``, a named
            :class:`torch.export.Dim` instance (e.g. ``torch.export.Dim("batch")``),
            or an unnamed ``torch.export.Dim`` sentinel such as
            ``torch.export.Dim.DYNAMIC`` or ``torch.export.Dim.AUTO``.
            For every key ``d`` the integer ``shape[d]`` is replaced by the
            resolved symbolic name in the resulting :class:`TracingShape`.
            When ``None`` or empty, all dimensions remain concrete integers.
        :return: A :class:`TracingShape` whose ``dims`` are ``int`` values for
            static dimensions and :class:`TracingInt` values for dynamic ones.
        """
        if not dynamic_shapes:
            return TracingShape(tuple(int(i) for i in shape))
        new_shape = [int(i) for i in shape]
        for d, dim_spec in dynamic_shapes.items():
            new_shape[d] = TracingInt(_dim_spec_to_name(dim_spec))  # type: ignore
        return TracingShape(tuple(new_shape))  # type: ignore
