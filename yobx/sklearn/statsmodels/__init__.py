from typing import List, Tuple


def register():
    try:
        import statsmodels  # noqa: F401

        from . import glm  # noqa: F401
    except ImportError:
        # No statsmodels installed.
        pass


def all_estimators() -> List[Tuple[str, type]]:
    """Returns all wrapper estimators in :epkg:`statsmodels` supported by this package."""
    try:
        from .glm import StatsmodelsGLMWrapper
    except ImportError:
        return []

    return [("StatsmodelsGLMWrapper", StatsmodelsGLMWrapper)]
