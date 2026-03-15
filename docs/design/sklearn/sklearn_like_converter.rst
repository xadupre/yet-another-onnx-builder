.. _l-design-sklearn-like-converters:

========================================
External Libraries Based on scikit-learn
========================================

:func:`yobx.sklearn.to_onnx` converts fitted estimators to ONNX
from :epkg:`xgboost`, :epkg:`lightgbm`, :epkg:`category_encoders`,
and :epkg:`imbalanced-learn`
using the same registry-based architecture as the other
:mod:`yobx.sklearn` converters.

+--------------------------------+-------------------------------------------+
| Package                        | Module                                    |
+================================+===========================================+
| :epkg:`category_encoders`      | :mod:`yobx.sklearn.category_encoders`     |
+--------------------------------+-------------------------------------------+
| :epkg:`xgboost`                | :mod:`yobx.sklearn.xgboost`               |
+--------------------------------+-------------------------------------------+
| :epkg:`lightgbm`               | :mod:`yobx.sklearn.lightgbm`              |
+--------------------------------+-------------------------------------------+
| :epkg:`imbalanced-learn`       | :mod:`yobx.sklearn.imblearn`              |
+--------------------------------+-------------------------------------------+

Comparison: XGBoost vs LightGBM
================================

Both :epkg:`XGBoost` and :epkg:`LightGBM` are gradient-boosted tree
libraries — their fitted models consist of an ensemble of binary decision
trees.  The converters map these trees to ONNX ``TreeEnsemble*`` operators
and share the same high-level structure, differing only in how split
conditions are expressed and how the raw margin is post-processed.
The table below summarises the key differences between the converters
for xgboost and lightgbm.

==================================  ================================  ==============================
Property                            XGBoost                           LightGBM
==================================  ================================  ==============================
Tree dump method                    ``get_dump(dump_format='json')``  ``booster_.dump_model()``
Split direction (numerical)         ``x < threshold`` (BRANCH_LT)     ``x ≤ threshold`` (BRANCH_LEQ)
Categorical splits                  Not supported by converter        Expanded to BRANCH_EQ chains
Base-score bias                     Added from ``base_score`` cfg     None (baked into leaf values)
Regression transform                Identity / Sigmoid / Exp          Identity / Exp
Multi-class targets per round       ``n_classes`` trees               ``n_classes`` trees
Binary classifier raw outputs       1 tree per round (sigmoid)        1 tree per round (sigmoid)
==================================  ================================  ==============================

imbalanced-learn
================

:epkg:`imbalanced-learn` extends :epkg:`scikit-learn` with resampling
techniques for imbalanced datasets.  Resampling only happens at training
time, so ONNX inference pipelines skip any step that exposes
``fit_resample`` and only convert the remaining transformers and the
final estimator.

Two converters are registered in :mod:`yobx.sklearn.imblearn`:

* **imblearn Pipeline** — wraps ``imblearn.pipeline.Pipeline``.  At
  conversion time the steps that expose ``fit_resample`` are filtered
  out; the remaining steps are forwarded to their own registered
  converters exactly as a standard ``sklearn.pipeline.Pipeline`` would
  be.

* **EasyEnsembleClassifier** — wraps
  ``imblearn.ensemble.EasyEnsembleClassifier``, a
  ``BaggingClassifier`` subclass whose sub-estimators are imblearn
  ``Pipeline`` instances (resampler + classifier).  The converter
  iterates ``estimators_`` / ``estimators_features_``, applies the
  imblearn Pipeline converter to each, averages the predicted
  probabilities (soft vote), and emits ``(label, proba)`` — the same
  logic used by the ``BaggingClassifier`` converter.

See :mod:`yobx.sklearn.imblearn` for the full API reference.
