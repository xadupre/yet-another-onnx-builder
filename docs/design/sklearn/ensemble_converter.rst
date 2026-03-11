.. _l-design-ensemble-converters:

============================================================
Sklearn Tree-Ensemble Converters (Random Forest, Extra Trees)
============================================================

:func:`yobx.sklearn.to_onnx` converts fitted
:class:`sklearn.ensemble.RandomForestClassifier`,
:class:`sklearn.ensemble.RandomForestRegressor`,
:class:`sklearn.ensemble.ExtraTreesClassifier`, and
:class:`sklearn.ensemble.ExtraTreesRegressor` estimators to ONNX using the
same registry-based architecture as the other :mod:`yobx.sklearn` converters.

Implementation modules:

* :mod:`yobx.sklearn.ensemble.random_forest` — all four converters

Common ONNX encoding
====================

All four converters support two ``ai.onnx.ml`` encodings, selected
automatically based on the active opset:

* **Opset ≤ 4 (legacy)** — ``TreeEnsembleRegressor`` /
  ``TreeEnsembleClassifier`` with flat ``nodes_*`` / ``target_*``
  (regressor) or ``class_*`` (classifier) attribute arrays.
* **Opset ≥ 5 (modern)** — unified ``TreeEnsemble`` operator with
  ``nodes_splits`` / ``leaf_weights`` tensor attributes.

Both encodings support ``float32`` and ``float64`` inputs; the regressor
paths insert an explicit ``Cast`` node when the input is ``float64`` because
the ``TreeEnsembleRegressor`` and ``TreeEnsemble`` operators always produce
``float32`` leaf scores.

Tree structure
==============

scikit-learn stores each fitted base estimator in
``estimator.estimators_`` as a :class:`sklearn.tree.DecisionTreeClassifier`
or :class:`sklearn.tree.DecisionTreeRegressor`.  Each base estimator exposes
a ``tree_`` attribute (a Cython ``Tree`` object) with the following flat
arrays:

====================  ========================================================
Attribute             Description
====================  ========================================================
``feature``           Feature index used at each internal node (``-2`` = leaf)
``threshold``         Split threshold (``-2.0`` = leaf)
``children_left``     Index of left child (``-1`` = leaf)
``children_right``    Index of right child (``-1`` = leaf)
``value``             Leaf/node values, shape ``(n_nodes, n_outputs, max_n_classes)``
====================  ========================================================

Internal nodes use *less-than-or-equal* splits (``x ≤ threshold`` goes
left), matching ONNX's ``BRANCH_LEQ`` mode.

Classifier converters
=====================

RandomForestClassifier
----------------------

:func:`~yobx.sklearn.ensemble.random_forest.sklearn_random_forest_classifier`
converts :class:`sklearn.ensemble.RandomForestClassifier`.

All ``n_estimators`` trees are packed into a **single ONNX node**.
Leaf weights are divided by ``n_estimators`` at export time so that the
``SUM`` aggregate (opset 5) or ``post_transform="NONE"`` (legacy) yields
the averaged class-probability vector.

**Legacy path** (``ai.onnx.ml`` opset ≤ 4):

.. code-block:: text

    X  ──TreeEnsembleClassifier(all trees, post_transform=NONE)──►  (label, probabilities)

The ``classlabels_int64s`` (or ``classlabels_strings``) attribute is
populated from ``estimator.classes_`` so integer and string class labels are
both supported.

**Modern path** (``ai.onnx.ml`` opset ≥ 5):

.. code-block:: text

    X  ──TreeEnsemble(n_classes virtual trees per estimator, aggregate=SUM)
                                │
                           scores [N, n_classes]
                                │
                       ┌────────┴────────────────────────┐
                    Identity                          ArgMax → Cast → Gather(classes_)
                       │                                  │
               probabilities [N, n_classes]           label [N]

ExtraTreesClassifier
--------------------

:func:`~yobx.sklearn.ensemble.random_forest.sklearn_extra_trees_classifier`
converts :class:`sklearn.ensemble.ExtraTreesClassifier`.

Extra Trees are identical to Random Forests from the ONNX-encoding
perspective — the ``tree_`` structure of every base estimator is the same.
This converter therefore delegates to the same attribute-extraction helpers
and the same ``_sklearn_random_forest_classifier_v5`` helper as the Random
Forest classifier; only the registered estimator class differs.

Regressor converters
====================

RandomForestRegressor
---------------------

:func:`~yobx.sklearn.ensemble.random_forest.sklearn_random_forest_regressor`
converts :class:`sklearn.ensemble.RandomForestRegressor`.

**Legacy path** (``ai.onnx.ml`` opset ≤ 4):

.. code-block:: text

    X  ──TreeEnsembleRegressor(all trees, aggregate=AVERAGE)──►  [N, 1] float32
                                │
                              Cast (to input dtype)
                                │
                           predictions [N, 1]

**Modern path** (``ai.onnx.ml`` opset ≥ 5):

.. code-block:: text

    X  ──TreeEnsemble(one virtual tree per estimator, aggregate=SUM)──►  predictions [N, 1]

Leaf weights are pre-divided by ``n_estimators`` so ``SUM`` aggregation is
equivalent to averaging.

ExtraTreesRegressor
-------------------

:func:`~yobx.sklearn.ensemble.random_forest.sklearn_extra_trees_regressor`
converts :class:`sklearn.ensemble.ExtraTreesRegressor`.

Identical ONNX encoding to the Random Forest regressor; only the registered
estimator class differs.

Attribute extraction helpers
============================

Both the legacy and v5 encodings share reusable helpers that convert the
``tree_`` object arrays from all base estimators into flat attribute lists
suitable for ONNX node creation:

:func:`~yobx.sklearn.ensemble.random_forest._extract_forest_attributes_legacy`
    Produces the flat ``nodes_*``, ``class_*`` / ``target_*`` arrays for
    ``TreeEnsembleClassifier`` / ``TreeEnsembleRegressor`` (opset ≤ 4).
    Each tree is assigned a unique ``tree_id``.

:func:`~yobx.sklearn.ensemble.random_forest._extract_forest_attributes_v5`
    Produces the ``nodes_*``, ``leaf_targetids``, and ``leaf_weights``
    arrays for the unified ``TreeEnsemble`` operator (opset 5).

    For classifiers ``n_classes`` *virtual trees* are created per
    estimator; virtual tree ``est * n_classes + cls`` stores only the
    probability for class ``cls``.  With ``aggregate_function=SUM`` and
    ``n_targets=n_classes`` the ``[N, n_classes]`` output is the averaged
    class-probability matrix.

Quick start
===========

.. code-block:: python

    import numpy as np
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Classifier — returns (label, probabilities)
    clf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    onx_clf = to_onnx(clf, (X,))

    # Regressor — returns predictions
    y_r = X[:, 0] * 2 + X[:, 1]
    reg = ExtraTreesRegressor(n_estimators=10, random_state=0).fit(X, y_r)
    onx_reg = to_onnx(reg, (X,))

    # Modern opset-5 path
    onx_v5 = to_onnx(clf, (X,), target_opset={"": 20, "ai.onnx.ml": 5})

For a worked end-to-end example including pipeline usage and opset-5 output
verification see :ref:`l-plot-sklearn-extra-trees`.

Supported input dtypes and opsets
==================================

Both converters handle ``float32`` and ``float64`` inputs and the active
``ai.onnx.ml`` opset:

* ``float32`` + opset ≤ 4 → ``TreeEnsembleRegressor`` / ``TreeEnsembleClassifier``
* ``float64`` + opset ≤ 4 → legacy operators + ``Cast`` node for regressor output
* ``float32`` or ``float64`` + opset 5 → unified ``TreeEnsemble``

Specifying the target opset:

.. code-block:: python

    onx = to_onnx(clf, (X.astype(np.float64),),
                  target_opset={"": 21, "ai.onnx.ml": 5})

Differences: Random Forest vs Extra Trees
==========================================

From the converter's perspective these two ensemble families are
**identical** — both expose ``estimators_``, ``classes_``, and
``n_estimators`` with the same semantics, and every base estimator has an
identical ``tree_`` structure.  The only difference is the registered class:

==========================================  ===================================
Estimator class                             Converter function
==========================================  ===================================
:class:`~sklearn.ensemble.RandomForestClassifier`  :func:`~yobx.sklearn.ensemble.random_forest.sklearn_random_forest_classifier`
:class:`~sklearn.ensemble.RandomForestRegressor`   :func:`~yobx.sklearn.ensemble.random_forest.sklearn_random_forest_regressor`
:class:`~sklearn.ensemble.ExtraTreesClassifier`    :func:`~yobx.sklearn.ensemble.random_forest.sklearn_extra_trees_classifier`
:class:`~sklearn.ensemble.ExtraTreesRegressor`     :func:`~yobx.sklearn.ensemble.random_forest.sklearn_extra_trees_regressor`
==========================================  ===================================
