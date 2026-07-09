from typing import List, Tuple


def register():
    try:
        import sksurv  # noqa: F401
    except ImportError:
        # No scikit-survival installed.
        return

    try:  # noqa: SIM105
        from . import ensemble  # noqa: F401
    except ImportError:
        # sksurv.ensemble may be incompatible with the installed sklearn version.
        pass

    try:  # noqa: SIM105
        from . import linear_model  # noqa: F401
    except ImportError:
        pass


def all_estimators() -> List[Tuple[str, type]]:
    """Returns all estimators in :epkg:`scikit-survival`."""
    import importlib
    import inspect

    try:
        from sklearn.base import BaseEstimator
    except ImportError:
        return []

    modules = []
    for mod_name in (
        "sksurv.linear_model",
        "sksurv.ensemble",
        "sksurv.svm",
        "sksurv.tree",
        "sksurv.meta",
    ):
        try:  # noqa: SIM105
            modules.append(importlib.import_module(mod_name))
        except ImportError:  # noqa: PERF203
            pass

    estimators = {}
    for module in modules:
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseEstimator) and obj.__module__.startswith("sksurv"):
                estimators[name] = obj
    return list(estimators.items())
