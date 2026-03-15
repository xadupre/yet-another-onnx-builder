from typing import List, Tuple


def register():
    try:
        import xgboost
        from . import xgb
    except ImportError:
        # No xgboost installed.
        pass


def all_estimators() -> List[Tuple[str, type]]:
    """Returns all estimators in :epkg:`xgboost`."""
    import inspect

    try:
        import xgboost.sklearn as xgb_sklearn
        from xgboost.sklearn import XGBModel
    except ImportError:
        return []

    estimators = []
    for name, obj in inspect.getmembers(xgb_sklearn, inspect.isclass):
        if issubclass(obj, XGBModel) and obj is not XGBModel:
            estimators.append((name, obj))
    return estimators
