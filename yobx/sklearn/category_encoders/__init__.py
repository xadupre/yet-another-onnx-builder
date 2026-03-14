def register():
    try:
        import category_encoders  # noqa: F401

        from . import quantile_encoder  # noqa: F401
    except ImportError:
        # No category_encoders installed.
        pass
