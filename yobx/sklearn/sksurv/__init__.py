from typing import List, Tuple


def register():
    try:
        import sksurv  # noqa: F401

        from . import linear_model  # noqa: F401
    except ImportError:
        # No scikit-survival installed.
        pass


def all_estimators() -> List[Tuple[str, type]]:
    """Returns all estimators in :epkg:`scikit-survival`."""
    import inspect

    try:
        import sksurv.linear_model as sksurv_linear_model
        from sklearn.base import RegressorMixin
    except ImportError:
        return []

    estimators = []
    for name, obj in inspect.getmembers(sksurv_linear_model, inspect.isclass):
        if issubclass(obj, RegressorMixin) and obj.__module__.startswith("sksurv"):
            estimators.append((name, obj))
    return estimators
