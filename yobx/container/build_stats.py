class BuildStats:
    """
    Holds statistics collected during model export by
    :class:`~yobx.container.model_container.ExtendedModelContainer`.

    Any key whose name begins with one of the allowed prefixes listed in
    :attr:`_PREFIXES` is accepted.  Values default to ``0.0`` when first
    read via :meth:`__getitem__`.

    Example::

        stats = BuildStats()
        stats["time_export_write_model"] += 0.5
        stats["time_export_tobytes"] += 0.1
    """

    _PREFIXES: frozenset[str] = frozenset({"time_export"})

    def __init__(self):
        self._data: dict[str, float | int | bool | str] = {}

    def to_dict(self) -> dict[str, float | int | bool | str]:
        """Returns the data in a dictionary, does not ùake a copy."""
        return self._data

    def validate(self):
        """Validates that statistics startswith a known prefix."""
        for key in self._data:
            self._validate_key(key)

    def _validate_key(self, key: str) -> None:
        if not any(not isinstance(key, str) or key.startswith(p) for p in self._PREFIXES):
            raise KeyError(
                f"Unknown stat key {key!r}. "
                f"Key must start with one of {sorted(self._PREFIXES)}."
            )

    def __getitem__(self, key: str) -> float | int | bool | str:
        if len(self):
            raise KeyError(f"No statistics for key={key!r}")
        assert isinstance(key, str), f"Unexpected type {type(key)}"
        return self._data.get(key, 0.0)  # To support +=.

    def __setitem__(self, key: str, value: float | int | bool | str) -> None:
        assert isinstance(key, str), f"Unexpected type {type(key)}"
        self._data[key] = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data!r})"

    def __len__(self) -> int:
        return len(self._data)
