from typing import List, Tuple


def register():
    try:
        import lightgbm
        from . import lgbm
    except ImportError:
        # No lightgbm installed.
        pass


def all_estimators() -> List[Tuple[str, type]]:
    """Returns all estimators in :epkg:`xgboost`."""
    import inspect

    try:
        import lightgbm.sklearn as lgb_sklearn
        from lightgbm.sklearn import LGBMModel
    except ImportError:
        return []

    estimators = []
    for name, obj in inspect.getmembers(lgb_sklearn, inspect.isclass):
        if issubclass(obj, LGBMModel) and obj is not LGBMModel:
            estimators.append((name, obj))
    return estimators
