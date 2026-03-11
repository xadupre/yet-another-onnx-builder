from .convert import to_onnx


def register_sklearn_converters():
    """Registers all converters implemented in this package."""
    from .register import SKLEARN_CONVERTERS

    if SKLEARN_CONVERTERS:
        # already done
        return
    from .cluster import register as register_cluster
    from .compose import register as register_compose
    from .decomposition import register as register_decomposition
    from .discriminant_analysis import register as register_discriminant_analysis
    from .ensemble import register as register_ensemble
    from .linear_model import register as register_linear_model
    from .multiclass import register as register_multiclass
    from .neighbors import register as register_neighbors
    from .neural_network import register as register_neural_network
    from .pipeline import register as register_pipeline
    from .preprocessing import register as register_preprocessing
    from .tree import register as register_tree
    from .lightgbm import register as register_lightgbm
    from .svm import register as register_svm
    from .xgboost import register as register_xgboost

    register_cluster()
    register_compose()
    register_decomposition()
    register_discriminant_analysis()
    register_ensemble()
    register_linear_model()
    register_lightgbm()
    register_multiclass()
    register_neighbors()
    register_neural_network()
    register_pipeline()
    register_preprocessing()
    register_svm()
    register_tree()
    register_xgboost()
