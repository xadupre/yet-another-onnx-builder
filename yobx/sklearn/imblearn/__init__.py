from typing import List, Tuple


def register():
    try:
        import imblearn  # noqa: F401

        from . import balanced_random_forest  # noqa: F401
        from . import easy_ensemble  # noqa: F401
        from . import balanced_bagging  # noqa: F401
    except ImportError:
        # No imbalanced-learn installed.
        pass


def all_estimators() -> List[Tuple[str, type]]:
    """Returns all estimators in :epkg:`imbalanced-learn`."""
    import inspect

    try:
        import imblearn.ensemble as imblearn_ensemble
        from sklearn.base import ClassifierMixin
    except ImportError:
        return []

    estimators = []
    for name, obj in inspect.getmembers(imblearn_ensemble, inspect.isclass):
        if issubclass(obj, ClassifierMixin) and obj.__module__.startswith("imblearn"):
            estimators.append((name, obj))
    return estimators
