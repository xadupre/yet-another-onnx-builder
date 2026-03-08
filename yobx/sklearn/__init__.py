from .convert import to_onnx


def register_sklearn_converters():
    """Registers all converters implemented in this package."""
    from .register import SKLEARN_CONVERTERS

    if SKLEARN_CONVERTERS:
        # already done
        return
    from .compose import register as register_compose
    from .decomposition import register as register_decomposition
    from .ensemble import register as register_ensemble
    from .linear_model import register as register_linear_model
    from .multiclass import register as register_multiclass
    from .neural_network import register as register_neural_network
    from .pipeline import register as register_pipeline
    from .preprocessing import register as register_preprocessing
    from .tree import register as register_tree

    register_compose()
    register_decomposition()
    register_ensemble()
    register_linear_model()
    register_multiclass()
    register_neural_network()
    register_pipeline()
    register_preprocessing()
    register_tree()
