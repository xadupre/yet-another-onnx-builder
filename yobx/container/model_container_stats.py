class ModelContainerStats:
    """
    Holds statistics collected during model export by
    :class:`~yobx.container.model_container.ExtendedModelContainer`.

    Any key whose name begins with one of the allowed prefixes listed in
    :attr:`_PREFIXES` is accepted.  Values default to ``0.0`` when first
    read via :meth:`__getitem__`.

    Example::

        stats = ModelContainerStats()
        stats["time_export_write_model"] += 0.5
        stats["time_export_tobytes"] += 0.1
    """

    _PREFIXES: frozenset[str] = frozenset({"time_export"})

    def __init__(self):
        self._data: dict[str, float] = {}

    def _validate_key(self, key: str) -> None:
        if not any(key.startswith(p) for p in self._PREFIXES):
            raise KeyError(
                f"Unknown stat key {key!r}. "
                f"Key must start with one of {sorted(self._PREFIXES)}."
            )

    def __getitem__(self, key: str) -> float:
        self._validate_key(key)
        return self._data.get(key, 0.0)

    def __setitem__(self, key: str, value: float) -> None:
        self._validate_key(key)
        self._data[key] = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data!r})"
