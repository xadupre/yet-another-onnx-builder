from typing import Optional


class InitializerInfo:
    """
    Tracks the location where the initializer was created.

    :param name: initializer name
    :param source: information
    :param same_as: same as an existing initializers
    """

    def __init__(self, name: str, source: str, same_as: Optional[str] = None):
        self.name = name
        self.source = source
        self.same_as = same_as

    def __repr__(self) -> str:
        if self.same_as:
            return (
                f"InitializerInfo({self.name!r}, source={self.source!r}, "
                f"same_as={self.same_as!r})"
            )
        return f"InitializerInfo({self.name!r}, source={self.source!r})"

    def add_source(self, source: str):
        """Adds other sources."""
        self.source += f"##{source}"
