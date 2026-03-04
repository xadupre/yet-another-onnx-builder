from .convert import to_onnx, convert_standard_scaler, convert_logistic_regression, convert_pipeline


def register_sklearn_converters():
    """Registers all converters implemented in this package."""
    from .register import SKLEARN_CONVERTERS

    if SKLEARN_CONVERTERS:
        # already done
        return
    from .linear_model import register as register_linear_model
    from .pipeline import register as register_pipeline
    from .preprocessing import register as register_preprocessing

    register_linear_model()
    register_pipeline()
    register_preprocessing()
