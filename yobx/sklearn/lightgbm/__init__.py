def register():
    try:
        import lightgbm
        from . import lgbm
    except ImportError:
        # No lightgbm installed.
        pass
