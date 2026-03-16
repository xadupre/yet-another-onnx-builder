import re


class PvVersion:
    """Simple version of packaging.version.Version."""

    def to_int(self, i: str) -> int:
        if i[0] == "0" and len(i) != 1:
            raise ValueError(f"{self!r} is not a valid version")
        return int(i)

    def __init__(self, version: str):
        self.version = version
        self.t_version = tuple(
            self.to_int(i)
            for i in re.split(r"[.+]", version)
            if not i.startswith(("dev", "rc", "post", "cpu", "cu"))
        )

    def __repr__(self) -> str:
        "usual"
        return f"Version({self.version!r})"

    def __eq__(self, other) -> bool:
        """=="""
        return self.version == other.version

    def __ge__(self, other) -> bool:
        """Lesser than."""
        assert isinstance(other, PvVersion), f"Unexpected type {type(other)}"
        return self.t_version >= other.t_version

    def __gt__(self, other) -> bool:
        """Lesser than."""
        assert isinstance(other, PvVersion), f"Unexpected type {type(other)}"
        return self.t_version > other.t_version

    def __le__(self, other) -> bool:
        """Lesser than."""
        assert isinstance(other, PvVersion), f"Unexpected type {type(other)}"
        return self.t_version <= other.t_version

    def __lt__(self, other) -> bool:
        """Lesser than."""
        assert isinstance(other, PvVersion), f"Unexpected type {type(other)}"
        return self.t_version < other.t_version
