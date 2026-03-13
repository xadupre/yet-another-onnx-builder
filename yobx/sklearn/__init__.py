from .convert import to_onnx

__all__ = ["to_onnx"]


def register_sklearn_converters():
    """Registers all converters implemented in this package."""
    from .register import SKLEARN_CONVERTERS

    if SKLEARN_CONVERTERS:
        # already done
        return
    from .calibration import register as register_calibration
    from .cluster import register as register_cluster
    from .compose import register as register_compose
    from .decomposition import register as register_decomposition
    from .discriminant_analysis import register as register_discriminant_analysis
    from .dummy import register as register_dummy
    from .ensemble import register as register_ensemble
    from .gaussian_process import register as register_gaussian_process
    from .lightgbm import register as register_lightgbm
    from .linear_model import register as register_linear_model
    from .manifold import register as register_manifold
    from .mixture import register as register_mixture
    from .multiclass import register as register_multiclass
    from .multioutput import register as register_multioutput
    from .naive_bayes import register as register_naive_bayes
    from .neighbors import register as register_neighbors
    from .neural_network import register as register_neural_network
    from .pipeline import register as register_pipeline
    from .preprocessing import register as register_preprocessing
    from .svm import register as register_svm
    from .tree import register as register_tree
    from .xgboost import register as register_xgboost

    register_calibration()
    register_cluster()
    register_compose()
    register_decomposition()
    register_discriminant_analysis()
    register_dummy()
    register_ensemble()
    register_gaussian_process()
    register_linear_model()
    register_lightgbm()
    register_manifold()
    register_mixture()
    register_multiclass()
    register_multioutput()
    register_naive_bayes()
    register_neighbors()
    register_neural_network()
    register_pipeline()
    register_preprocessing()
    register_svm()
    register_tree()
    register_xgboost()
