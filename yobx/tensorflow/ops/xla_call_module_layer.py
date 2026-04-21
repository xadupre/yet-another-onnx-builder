"""Layer class for the XlaCallModule converter.

:class:`XlaLayer` represents a single StableHLO operation parsed from a
StableHLO MLIR module.  It replaces the plain ``dict`` that was previously
used so that layer fields are typed, documented, and accessible via attribute
syntax as well as the legacy ``layer["key"]`` dict-style syntax.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union


class XlaLayer:
    """Represents a single StableHLO operation layer.

    All layers carry the four core fields *id*, *op*, *operands*, *shape*,
    and *loc*.  Op-specific optional fields default to empty / ``None`` and
    are populated only for the ops that require them.

    Dict-style access (``layer["op"]``, ``layer.get("axes", [])``) is
    supported for backward compatibility with code that was written against
    the plain-dict API.

    :param id: SSA result identifier (e.g. ``"%0"``).
    :param op: Operation name (e.g. ``"sine"``, ``"Input"``, ``"return"``).
    :param operands: List (or tuple) of input SSA ids.
    :param shape: Result tensor type string (e.g. ``"tensor<3x4xf32>"``).
    :param loc: Source location from ``loc(...)`` annotations.
    :param dense_content: Raw dense value string for ``constant`` ops.
    :param dims: Broadcast dimension list for ``broadcast_in_dim`` /
        ``dynamic_broadcast_in_dim`` ops.
    :param axes: Reduction axis list for ``reduce_max`` / ``reduce_sum`` ops.
    :param func: Callee name for ``call`` ops.
    :param lhs_contracting: LHS contracting dimensions for ``dot_general``.
    :param rhs_contracting: RHS contracting dimensions for ``dot_general``.
    """

    __slots__ = (  # noqa: RUF023 – logical ordering: core fields first, then optional
        "id",
        "op",
        "operands",
        "shape",
        "loc",
        "dense_content",
        "dims",
        "axes",
        "func",
        "lhs_contracting",
        "rhs_contracting",
    )

    def __init__(
        self,
        id: str,
        op: str,
        operands: Union[List[str], Sequence[str]],
        shape: str = "",
        loc: str = "",
        *,
        dense_content: str = "",
        dims: Optional[List[int]] = None,
        axes: Optional[List[int]] = None,
        func: str = "",
        lhs_contracting: Optional[List[int]] = None,
        rhs_contracting: Optional[List[int]] = None,
    ) -> None:
        self.id: str = id
        self.op: str = op
        self.operands: Union[List[str], Sequence[str]] = operands
        self.shape: str = shape
        self.loc: str = loc
        self.dense_content: str = dense_content
        self.dims: List[int] = dims if dims is not None else []
        self.axes: List[int] = axes if axes is not None else []
        self.func: str = func
        self.lhs_contracting: List[int] = lhs_contracting if lhs_contracting is not None else []
        self.rhs_contracting: List[int] = rhs_contracting if rhs_contracting is not None else []

    # ------------------------------------------------------------------
    # Dict-compatible interface (backward compatibility)
    # ------------------------------------------------------------------

    def __getitem__(self, key: str):
        """Returns the attribute named *key*, raising :exc:`KeyError` if absent."""
        try:
            return getattr(self, key)
        except AttributeError:
            # Intentionally converts AttributeError to KeyError to match dict semantics.
            raise KeyError(key) from None

    def __setitem__(self, key: str, value) -> None:
        """Sets the attribute named *key*."""
        try:
            setattr(self, key, value)
        except AttributeError:
            # Intentionally converts AttributeError to KeyError to match dict semantics.
            raise KeyError(key) from None

    def __contains__(self, key: str) -> bool:
        """Returns ``True`` when *key* is a known field name with a truthy value."""
        return key in self.__slots__ and bool(getattr(self, key, None))

    def get(self, key: str, default=None):
        """Returns the attribute named *key*, or *default* when not present."""
        return getattr(self, key, default)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Returns a concise string representation."""
        parts = [f"op={self.op!r}", f"id={self.id!r}"]
        if self.operands:
            parts.append(f"operands={list(self.operands)!r}")
        if self.shape:
            parts.append(f"shape={self.shape!r}")
        if self.dense_content:
            parts.append(f"dense_content={self.dense_content!r}")
        if self.dims:
            parts.append(f"dims={self.dims!r}")
        if self.axes:
            parts.append(f"axes={self.axes!r}")
        if self.func:
            parts.append(f"func={self.func!r}")
        if self.lhs_contracting:
            parts.append(f"lhs_contracting={self.lhs_contracting!r}")
        if self.rhs_contracting:
            parts.append(f"rhs_contracting={self.rhs_contracting!r}")
        return f"XlaLayer({', '.join(parts)})"
