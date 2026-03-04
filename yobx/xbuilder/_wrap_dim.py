from __future__ import annotations
class WrapDim:
    """Wraps a string considered as a ``torch.export.Dim``."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"WrapDim({self.name!r})"

    @property
    def name_as_string(self):
        """Returns the name as a string."""
        if isinstance(self.name, str):
            return self.name
        if self.name.__class__.__name__ == "Dim":
            # It should be torch.export.dynamic_shapes.Dim
            return self.name.__name__
        raise AssertionError(
            f"Unable to return the dimension as a string type is "
            f"{type(self.name)}, name={self.name!r}"
        )
