from typing import Any, Optional, Union

from .wrap_dim import WrapDim


class WrapSym:
    """Wraps a symbolic int (a dimension for example)."""

    def __init__(self, sym: Union["torch.SymInt", "torch.SymFloat", "WrapSym"]):  # noqa: F821
        if isinstance(sym, WrapDim):
            self.sym = sym.name
        else:
            assert isinstance(sym, str) or hasattr(
                sym, "node"
            ), f"Missing attribute node for type {type(sym)}"
            self.sym = sym  # type: ignore

    def __repr__(self) -> str:
        return f"WrapSym({self._dynamic_to_str(self.sym)})"

    @property
    def name(self):
        """Returns the name as a string."""
        return self._dynamic_to_str(self.sym)

    def _dynamic_to_str(self, obj: Any) -> Optional[str]:
        if isinstance(obj, str):
            return obj
        import torch

        if isinstance(obj, torch.export.dynamic_shapes._DerivedDim):
            return obj.__name__
        if isinstance(obj, torch.export.dynamic_shapes._Dim):
            return obj.__name__
        if isinstance(obj, (torch.SymInt, torch.SymFloat)):
            if isinstance(obj.node, str):
                return obj.node
            i = obj.node._expr
            if "sympy" in str(type(i)):
                return str(i).replace(" ", "")
            return None
        raise AssertionError(f"Unexpected type {type(obj)} to convert into string")
