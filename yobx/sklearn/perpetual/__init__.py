from typing import List, Tuple


def register():
    try:
        import perpetual  # noqa: F401
        from . import perpetual as perpetual_converter  # noqa: F401
    except ImportError:
        # No perpetual installed.
        pass


def all_estimators() -> List[Tuple[str, type]]:
    """Returns supported estimators in :epkg:`perpetual`."""
    try:
        from perpetual import PerpetualClassifier, PerpetualRegressor
    except ImportError:
        return []

    return [
        ("PerpetualClassifier", PerpetualClassifier),
        ("PerpetualRegressor", PerpetualRegressor),
    ]
