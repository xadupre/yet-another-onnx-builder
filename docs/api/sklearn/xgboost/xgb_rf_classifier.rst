XGBRFClassifier
===============

The :class:`xgboost.XGBRFClassifier` converter is implemented in the same
module as :class:`xgboost.XGBClassifier` and shares the same ONNX graph
construction logic.  Because :class:`~xgboost.XGBRFClassifier` inherits
from :class:`~xgboost.XGBClassifier` and exposes an identical booster
interface, the converter handles both classes transparently.

See :func:`yobx.sklearn.xgboost.xgb.sklearn_xgb_classifier` for the full
converter documentation.

.. autofunction:: yobx.sklearn.xgboost.xgb.sklearn_xgb_classifier
