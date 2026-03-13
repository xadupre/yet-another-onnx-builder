from .convert import to_onnx

__all__ = ["NumericalDiscrepancyWarning", "register_sklearn_converters", "to_onnx"]


def has_sklearn(version: str = ""):
    """Tells if scikit-learn is available and more recent than the given version."""
    try:
        import sklearn
    except ImportError:
        return False
    if not hasattr(sklearn, "__version__"):
        return False
    if not version:
        return True
    import packaging.version as pv

    return pv.Version(version) <= pv.Version(sklearn.__version__)


class NumericalDiscrepancyWarning(UserWarning):
    """
    The exported model has discrepancies for a particular version
    of scikit-learn while running unit tests.
    scikit-learn>=1.8 is more strict about types and does not implicitly switch
    to float64 in many models relying on a matrix multiplication or a
    normalization.
    """


def register_sklearn_converters():
    """Registers all converters implemented in this package."""
    from .register import SKLEARN_CONVERTERS

    if SKLEARN_CONVERTERS:
        # already done
        return
    from .calibration import register as register_calibration
    from .cluster import register as register_cluster
    from .compose import register as register_compose
    from .covariance import register as register_covariance
    from .decomposition import register as register_decomposition
    from .discriminant_analysis import register as register_discriminant_analysis
    from .dummy import register as register_dummy
    from .ensemble import register as register_ensemble
    from .gaussian_process import register as register_gaussian_process
    from .lightgbm import register as register_lightgbm
    from .linear_model import register as register_linear_model
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
    register_covariance()
    register_decomposition()
    register_discriminant_analysis()
    register_dummy()
    register_ensemble()
    register_gaussian_process()
    register_linear_model()
    register_lightgbm()
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
