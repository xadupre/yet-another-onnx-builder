from typing import Any, Dict, Optional, Sequence, Tuple, Union
from ...xexpressions import simplify_expression
import torch


class TracingBool:
    """
    Represents a boolean comparison that may involve symbolic :class:`TracingInt`
    dimensions and therefore cannot always be evaluated at trace time.

    :param value: Either a concrete :class:`bool` or a :class:`str` containing
        a symbolic expression such as ``"(batch==4)"``.

    - ``TracingBool(True)`` / ``TracingBool(False)`` — concrete result.
    - ``TracingBool("(n==4)")`` — symbolic; cannot be used as a Python bool.
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
        raise ValueError(
            f"TracingBool({self.value!r}) cannot be converted to a Python bool; "
            "the result depends on a symbolic/dynamic dimension"
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
        self.value = value

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
        cls, shape: Tuple[int, ...], dynamic_shapes: Optional[Dict[int, str]] = None
    ) -> "TracingShape":
        """
        Build a :class:`TracingShape` from a concrete shape tuple, optionally
        making selected dimensions symbolic.

        :param shape: The concrete shape (e.g. from ``tensor.shape``).
        :param dynamic_shapes: An optional mapping from *dimension index* to
            *symbolic name*.  For every key ``d`` the integer ``shape[d]`` is
            replaced by the string ``dynamic_shapes[d]`` in the resulting
            :class:`TracingShape`.  When ``None`` or empty, all dimensions
            remain concrete integers.
        :return: A :class:`TracingShape` whose ``dims`` are ``int`` values for
            static dimensions and ``str`` values for dynamic ones.
        """
        if not dynamic_shapes:
            return TracingShape(tuple(int(i) for i in shape))
        new_shape = [int(i) for i in shape]
        for d, name in dynamic_shapes.items():
            new_shape[d] = name
        return TracingShape(tuple(new_shape))  # type: ignore
