from typing import Callable, Set, Sequence, Union
from sklearn.base import BaseEstimator
from ..typing import ConvertOptionsProtocol


class ConvertOptions(ConvertOptionsProtocol):
    """
    Tunes the way every piece of a model is exported.

    Pass an instance of this class to :func:`yobx.sklearn.to_onnx` via the
    ``convert_options`` keyword argument to request **extra outputs** from
    tree and ensemble estimators.

    :param decision_leaf: when ``True``, an extra ``int64`` output tensor is
        appended containing the zero-based leaf node index reached by each
        input sample.  The shape is ``(N, 1)`` for single trees and
        ``(N, n_estimators)`` for ensembles. The option triggers for every
        estimator which implements `decision_path` method.
    :param decision_path: when ``True``, an extra ``object`` (string) output
        tensor is appended containing the binary root-to-leaf path for each
        input sample.  Each value is a byte-string whose *i*-th character is
        ``'1'`` if node *i* was visited and ``'0'`` otherwise.  The shape is
        ``(N, 1)`` for single trees and ``(N, n_estimators)`` for ensembles.
        The option triggers for every estimator which implements
        `decision_path` method.

    **Class attributes**

    .. attribute:: OPTIONS

        :type: list[str]

        Sorted list of all recognised option names.  Currently
        ``["decision_leaf", "decision_path"]``.

    **Example**::

        import numpy as np
        from sklearn.tree import DecisionTreeClassifier
        from yobx.sklearn import ConvertOptions, to_onnx

        X = np.random.randn(20, 4).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        clf = DecisionTreeClassifier(max_depth=3).fit(X, y)

        # Export with both extra outputs enabled
        opts = ConvertOptions(decision_leaf=True, decision_path=True)
        artifact = to_onnx(clf, (X,), convert_options=opts)
        # The model now produces four outputs:
        #   label (int64), probabilities (float32),
        #   decision_path (object/string), decision_leaf (int64)

    .. seealso::

        :ref:`l-plot-sklearn-convert-options` — worked examples for single
        trees and ensemble models.
    """

    OPTIONS = {
        "decision_leaf": lambda est: hasattr(est, "decision_path"),
        "decision_path": lambda est: hasattr(est, "decision_path"),
    }

    def __init__(
        self,
        decision_leaf: Union[bool, Set[Union[str, type, int, Callable]]] = False,
        decision_path: Union[bool, Set[Union[str, type, int, Callable]]] = False,
    ):
        self.decision_leaf = decision_leaf
        self.decision_path = decision_path

    def available_options(self) -> Sequence[str]:
        """Returns the list of available options."""
        return list(self.OPTIONS)

    def __repr__(self):
        rows = []
        for name in self.OPTIONS:
            rows.append(f"    {name}={getattr(self, name)},")
        text = "\n".join(rows)
        return f"{self.__class__.__name__}(\n{text}\n)"

    def has(self, option_name: str, piece: BaseEstimator) -> bool:  # type: ignore[bad-override]
        """Return ``True`` if option *option_name* is active for estimator *piece*.

        :param option_name: name of the option to query.  Must be one of the
            strings listed in :attr:`OPTIONS`.  An :exc:`AssertionError` is
            raised when an unknown name is passed.
        :param piece: the fitted :class:`~sklearn.base.BaseEstimator` for which
            the option is being queried.  When the attribute value is a
            :class:`set`, the set may contain types, integer object ids,
            class-name strings, or callables.  A callable element is called
            with *piece* as its sole argument and the option is considered
            active if any callable returns ``True``.  A string element is
            compared against ``type(piece).__name__``.
        :return: ``True`` when the option is enabled globally (the attribute
            value is ``True``), ``False`` when it is disabled (``False`` or any
            falsy value).
        :raises AssertionError: if *option_name* is not a member of
            :attr:`OPTIONS`.
        """
        assert hasattr(
            self, option_name
        ), f"Missing option {option_name!r}. Allowed {sorted(self.OPTIONS)}."
        value = getattr(self, option_name)
        if not value:
            return False
        if value is True:
            return self.OPTIONS[option_name](piece)
        if isinstance(value, set):
            for item in value:
                if callable(item) and not isinstance(item, type) and item(piece):
                    return True
            return (
                type(piece) in value
                or id(piece) in value
                or type(piece).__name__ in value
            )
        raise NotImplementedError(
            f"Not implemented with {option_name!r} is not a boolean but {value!r}."
        )
