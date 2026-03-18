def register():
    from . import grid_search  # noqa: F401
    from . import halving_search  # noqa: F401
    from . import random_search  # noqa: F401

    try:  # noqa: SIM105
        from . import tuned_threshold_classifier_cv  # noqa: F401
    except ImportError:
        # This was introduced in scikit-learn==1.5
        pass
