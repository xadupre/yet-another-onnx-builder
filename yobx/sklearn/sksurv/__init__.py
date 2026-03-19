from typing import List, Tuple


def register():
    try:
        import sksurv  # noqa: F401

        from . import ensemble  # noqa: F401
        from . import linear_model  # noqa: F401
    except ImportError:
        # No scikit-survival installed.
        pass


def all_estimators() -> List[Tuple[str, type]]:
    """Returns all estimators in :epkg:`scikit-survival`."""
    import inspect

    try:
        import sksurv.ensemble as sksurv_ensemble
        import sksurv.linear_model as sksurv_linear_model
        import sksurv.meta as sksurv_meta
        import sksurv.svm as sksurv_svm
        import sksurv.tree as sksurv_tree
        from sklearn.base import BaseEstimator
    except ImportError:
        return []

    estimators = {}
    for module in (sksurv_linear_model, sksurv_ensemble, sksurv_svm, sksurv_tree, sksurv_meta):
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseEstimator) and obj.__module__.startswith("sksurv"):
                estimators[name] = obj
    return list(estimators.items())
