from ..pv_version import PvVersion
from .convert import to_onnx
from .convert_options import ConvertOptions
from .skl2onnx_converter import wrap_skl2onnx_converter
from .sklearn_helper import NoKnownOutputMixin

__all__ = [
    "ConvertOptions",
    "NoKnownOutputMixin",
    "NumericalDiscrepancyWarning",
    "register_sklearn_converters",
    "to_onnx",
    "wrap_skl2onnx_converter",
]


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
    return PvVersion(version) <= PvVersion(sklearn.__version__)


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
    from .category_encoders import register as register_category_encoders
    from .cluster import register as register_cluster
    from .compose import register as register_compose
    from .covariance import register as register_covariance
    from .cross_decomposition import register as register_cross_decomposition
    from .decomposition import register as register_decomposition
    from .discriminant_analysis import register as register_discriminant_analysis
    from .dummy import register as register_dummy
    from .ensemble import register as register_ensemble
    from .feature_extraction import register as register_feature_extraction
    from .feature_selection import register as register_feature_selection
    from .gaussian_process import register as register_gaussian_process
    from .imblearn import register as register_imblearn
    from .impute import register as register_impute
    from .isotonic import register as register_isotonic
    from .kernel_approximation import register as register_kernel_approximation
    from .kernel_ridge import register as register_kernel_ridge
    from .lightgbm import register as register_lightgbm
    from .linear_model import register as register_linear_model
    from .manifold import register as register_manifold
    from .mixture import register as register_mixture
    from .model_selection import register as register_model_selection
    from .multiclass import register as register_multiclass
    from .multioutput import register as register_multioutput
    from .naive_bayes import register as register_naive_bayes
    from .neighbors import register as register_neighbors
    from .neural_network import register as register_neural_network
    from .pipeline import register as register_pipeline
    from .preprocessing import register as register_preprocessing
    from .sksurv import register as register_sksurv
    from .statsmodels import register as register_statsmodels
    from .svm import register as register_svm
    from .tree import register as register_tree
    from .xgboost import register as register_xgboost

    register_calibration()
    register_category_encoders()
    register_cluster()
    register_compose()
    register_covariance()
    register_cross_decomposition()
    register_decomposition()
    register_discriminant_analysis()
    register_dummy()
    register_ensemble()
    register_feature_extraction()
    register_feature_selection()
    register_gaussian_process()
    register_imblearn()
    register_impute()
    register_isotonic()
    register_kernel_approximation()
    register_kernel_ridge()
    register_linear_model()
    register_lightgbm()
    register_manifold()
    register_mixture()
    register_model_selection()
    register_multiclass()
    register_multioutput()
    register_naive_bayes()
    register_neighbors()
    register_neural_network()
    register_pipeline()
    register_preprocessing()
    register_sksurv()
    register_statsmodels()
    register_svm()
    register_tree()
    register_xgboost()
