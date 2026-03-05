from .convert import to_onnx


def register_sklearn_converters():
    """Registers all converters implemented in this package."""
    from .register import SKLEARN_CONVERTERS

    if SKLEARN_CONVERTERS:
        # already done
        return
    from .linear_model import register as register_linear_model
    from .neural_network import register as register_neural_network
    from .pipeline import register as register_pipeline
    from .preprocessing import register as register_preprocessing
    from .tree import register as register_tree

    register_linear_model()
    register_neural_network()
    register_pipeline()
    register_preprocessing()
    register_tree()
