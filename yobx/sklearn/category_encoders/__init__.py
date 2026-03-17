from typing import List, Tuple


def register():
    try:
        import category_encoders  # noqa: F401

        from . import one_hot_encoder, ordinal_encoder, polynomial_encoder, quantile_encoder, woe_encoder
    except ImportError:
        # No category_encoders installed.
        pass


def all_estimators() -> List[Tuple[str, type]]:
    """Returns all estimators in :epkg:`category_encoders`."""
    import inspect

    try:
        import category_encoders as ce
        from category_encoders.utils import BaseEncoder
    except ImportError:
        return []

    encoders = []
    for name, obj in inspect.getmembers(ce):
        if inspect.isclass(obj) and issubclass(obj, BaseEncoder) and obj is not BaseEncoder:
            encoders.append((name, obj))
    return encoders
