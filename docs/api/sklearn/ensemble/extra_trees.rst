yobx.sklearn.ensemble — Extra Trees
====================================

:func:`~yobx.sklearn.to_onnx` converts
:class:`sklearn.ensemble.ExtraTreesClassifier` and
:class:`sklearn.ensemble.ExtraTreesRegressor` to ONNX using the same
``TreeEnsemble*`` operators as the Random Forest converters (the two
families share an identical internal ``tree_`` structure).

Both encodings are supported, selected automatically from the active
``ai.onnx.ml`` opset:

* **Opset ≤ 4 (legacy)** — ``TreeEnsembleClassifier`` /
  ``TreeEnsembleRegressor``
* **Opset ≥ 5 (modern)** — unified ``TreeEnsemble``

Classifier
----------

.. autofunction:: yobx.sklearn.ensemble.random_forest.sklearn_extra_trees_classifier

Regressor
---------

.. autofunction:: yobx.sklearn.ensemble.random_forest.sklearn_extra_trees_regressor
