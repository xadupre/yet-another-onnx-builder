from typing import Set, Union
from sklearn.base import BaseEstimator
from ..typing import ConvertOptionsProtocol


class ConvertOptions(ConvertOptionsProtocol):
    """
    Tunes the way every piece of a model is exported.

    :param decision_leaf: (:class:`bool`) — when ``True``, an extra output
        tensor is appended containing the leaf node index (int64) for each
        input sample.  Shapes follow the same convention as ``decision_path``.
    :param decision_path: (:class:`bool`) — when ``True``, an extra output
        tensor is appended containing the binary decision path string(s) for
        each input sample.  For single trees the shape is ``(N, 1)``; for
        ensembles ``(N, n_estimators)``.
    """

    OPTIONS = ["decision_leaf", "decision_path"]

    def __init__(
        self,
        decision_leaf: Union[bool, Set[str]] = False,
        decision_path: Union[bool, Set[str]] = False,
    ):
        self.decision_leaf = decision_leaf
        self.decision_path = decision_path

    def __repr__(self):
        rows = []
        for name in self.OPTIONS:
            rows.append(f"    {name}={getattr(self, name)},")
        text = "\n".join(rows)
        return f"{self.__class__.__name__}(\n{text}\n)"

    def has(self, option_name: str, piece: BaseEstimator) -> bool:  # type: ignore[bad-override]
        """Tells of options `option_name` applies on estimator `piece`."""
        assert hasattr(
            self, option_name
        ), f"Missing option {option_name!r}. Allowed {self.OPTIONS}."
        value = getattr(self, option_name)
        if not value:
            return False
        if value is True:
            return True
        raise NotImplementedError(
            f"Not implemented with {option_name!r} is not a boolean but {value!r}."
        )
