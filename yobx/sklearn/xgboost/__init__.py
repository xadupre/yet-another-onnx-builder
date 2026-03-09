def register():
    try:
        import xgboost
        from . import xgb
    except ImportError:
        # No xgboost installed.
        pass
