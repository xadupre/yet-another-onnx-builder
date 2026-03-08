def register():
    """Register XGBoost converters if :epkg:`xgboost` is installed."""
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError:
        return  # xgboost not installed; skip silently

    from ..register import SKLEARN_CONVERTERS
    from .xgb import sklearn_xgb_classifier, sklearn_xgb_regressor

    if XGBClassifier not in SKLEARN_CONVERTERS:
        SKLEARN_CONVERTERS[XGBClassifier] = sklearn_xgb_classifier
    if XGBRegressor not in SKLEARN_CONVERTERS:
        SKLEARN_CONVERTERS[XGBRegressor] = sklearn_xgb_regressor
