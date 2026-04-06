from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Union
from ...xexpressions import simplify_expression
import torch

# ---------------------------------------------------------------------------
# Registry of conditions known to be True (populated by _handle_check during
# tracing via torch._check interception).  Symbolic TracingBool values whose
# expression matches an entry resolve to True in __bool__.
# ---------------------------------------------------------------------------

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
    """

    def __init__(self, value: Union[bool, str]):
        self.value = value

    def __repr__(self) -> str:
        return f"TracingBool({self.value!r})"

    def __str__(self) -> str:
        return str(self.value)

    def __bool__(self) -> bool:
        if isinstance(self.value, bool):
            return self.value
        if self.value in _known_true_conditions:
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
    """

    def __init__(self, value: Union[int, str]):
        assert isinstance(value, (int, str)), f"Unexpected type {type(value)} for value"
        self.value = value

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

    def _cmp(self, op: str, other: Union[int, "TracingInt"]) -> Union[bool, "TracingBool"]:
        """Applies comparison *op* to *self* and *other*.

        Returns a concrete :class:`bool` when both operands are concrete, and
        a :class:`TracingBool` with a symbolic expression string otherwise.

        :param op: Comparison operator string: ``">"``, ``">="``, ``"<"``, ``"<="``, or ``"!="``.
        :param other: The right-hand-side operand.
        :return: :class:`bool` or :class:`TracingBool`.
        """
        if isinstance(other, TracingInt):
            if isinstance(self.value, int) and isinstance(other.value, int):
                return bool(_COMPARISON_OPS[op](self.value, other.value))
            return TracingBool(f"({self._sym()}{op}{other._sym()})")
        if isinstance(other, int):
            if isinstance(self.value, int):
                return bool(_COMPARISON_OPS[op](self.value, other))
            return TracingBool(f"({self._sym()}{op}{other})")
        return NotImplemented

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
        Builds a :class:`TracingShape` from a concrete shape tuple, optionally
        making selected dimensions symbolic.

        :param shape: The concrete shape (e.g. from ``tensor.shape``).
        :param dynamic_shapes: An optional mapping from *dimension index* to
            either a :class:`str` (symbolic name) or a
            :class:`torch.export.Dim` object (whose ``__name__`` is used as
            the symbolic name).  For every key ``d`` the integer ``shape[d]``
            is replaced by a :class:`TracingInt` in the resulting
            :class:`TracingShape`.  When ``None`` or empty, all dimensions
            remain concrete integers.
        :return: A :class:`TracingShape` whose ``dims`` are :class:`int` values
            for static dimensions and :class:`TracingInt` for dynamic ones.
        """
        if not dynamic_shapes:
            return TracingShape(tuple(int(i) for i in shape))
        new_shape = [int(i) for i in shape]
        for d, dim_spec in dynamic_shapes.items():
            if isinstance(dim_spec, (int, str)):
                name = dim_spec
            elif hasattr(dim_spec, "__name__"):
                # torch.export.Dim objects expose their name via ``__name__``
                name = dim_spec.__name__
            else:
                raise ValueError(
                    f"Unexpected type {type(dim_spec)} for dynamic shape specification "
                    f"at dimension {d}: {dim_spec!r}"
                )
            new_shape[d] = TracingInt(name)  # type: ignore
        return TracingShape(tuple(new_shape))  # type: ignore
