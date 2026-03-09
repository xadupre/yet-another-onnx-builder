.. _l-design-lightgbm-converter:

==================
LightGBM Converter
==================

:func:`yobx.sklearn.to_onnx` converts fitted
:class:`lightgbm.LGBMRegressor` and :class:`lightgbm.LGBMClassifier`
estimators to ONNX using the same registry-based architecture as the other
:mod:`yobx.sklearn` converters.

The implementation lives in :mod:`yobx.sklearn.lightgbm.lgbm` and is
registered automatically when
:func:`~yobx.sklearn.register_sklearn_converters` is called.

Overview
========

The fitted model's internal boosted-tree structure is extracted via
``booster_.dump_model()`` and encoded into an ONNX tree-ensemble operator.
Two encodings are supported depending on the active ``ai.onnx.ml`` opset:

* **Opset ≤ 4 (legacy)** — ``TreeEnsembleRegressor`` /
  ``TreeEnsembleClassifier`` with flat ``nodes_*`` / ``target_*`` (regressor)
  or ``class_*`` (classifier) attribute arrays.
* **Opset ≥ 5 (modern)** — unified ``TreeEnsemble`` operator with separate
  ``nodes_splits`` / ``leaf_weights`` tensor attributes.

Both encodings are supported for ``float32`` and ``float64`` inputs.

Tree structure
==============

LightGBM trees use a *binary, left-deep* structure where internal nodes
contain a *split condition* and two child pointers:

.. code-block:: text

    internal node
    ├── split_feature  (feature index)
    ├── threshold      (split value or category set)
    ├── decision_type  ('<=' for numerical, '==' for categorical)
    ├── left_child     (taken when condition is TRUE)
    └── right_child    (taken when condition is FALSE)

Numerical splits
----------------

Numerical splits use ``decision_type == '<='`` with a floating-point
threshold.  The ONNX ``BRANCH_LEQ`` mode matches LightGBM's exact semantics
(*go left when* ``x ≤ threshold``).

Categorical splits
------------------

Categorical splits use ``decision_type == '=='`` with a threshold string
such as ``'0||1||2'``, meaning *go left if the feature value is 0, 1, or 2*.
Because ONNX only supports single-value ``BRANCH_EQ`` (``x == v``), the
converter calls :func:`~yobx.sklearn.lightgbm.lgbm._expand_categorical_splits`
to replace each multi-value categorical node with a **chain of single-value
BRANCH_EQ nodes**:

.. code-block:: text

    feature IN {0, 1, 2}
      → BRANCH_EQ(feature==0): true→left_subtree, false→
          BRANCH_EQ(feature==1): true→left_subtree, false→
              BRANCH_EQ(feature==2): true→left_subtree, false→right_subtree

Shared ``left_subtree`` references across the chain are handled by
:func:`~yobx.sklearn.lightgbm.lgbm._flatten_lgbm_tree`, which uses a
memoised depth-first traversal keyed on Python object identity, ensuring
each unique subtree is assigned exactly one flat node ID.

LGBMRegressor
=============

:func:`~yobx.sklearn.lightgbm.lgbm.sklearn_lgbm_regressor` converts
:class:`lightgbm.LGBMRegressor`.

The raw sum of tree outputs (margin) is post-processed depending on the
model's objective:

=======================================  =======================
Objective                                Output transform
=======================================  =======================
``regression``, ``regression_l1``,       Identity (no transform)
``huber``, ``quantile``, ``mape``, …
``poisson``, ``tweedie``                 ``Exp(margin)``
=======================================  =======================

Unsupported objectives raise :class:`NotImplementedError`.

.. code-block:: python

    import numpy as np
    from lightgbm import LGBMRegressor
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4)).astype(np.float32)
    y = X[:, 0] + 2 * X[:, 1]

    reg = LGBMRegressor(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    onx = to_onnx(reg, (X,))

Graph structure (regression, identity objective):

.. code-block:: text

    X  ──TreeEnsemble(Σ trees)──►  margin
                                       │
                                  Cast(→itype)
                                       │
                                  [Exp]  (only for poisson/tweedie)
                                       │
                                  Cast  ──►  predictions  [N, 1]

LGBMClassifier
==============

:func:`~yobx.sklearn.lightgbm.lgbm.sklearn_lgbm_classifier` converts
:class:`lightgbm.LGBMClassifier`.

**Binary classification** (``n_classes_ == 2``):

.. code-block:: text

    X  ──TreeEnsemble(1 target)──►  raw_score  [N, 1]
                                        │
                                    Sigmoid  ──►  p1  [N, 1]
                                        │
                               ┌────────┴────────┐
                            p1  [N,1]        Sub(1, p1)  ──►  p0  [N, 1]
                               └────────┬────────┘
                                     Concat  ──►  probabilities  [N, 2]
                                        │
                                     ArgMax ──Cast──Gather(classes_)  ──►  label

**Multi-class classification** (``n_classes_ > 2``):

.. code-block:: text

    X  ──TreeEnsemble(n_classes targets)──►  raw_scores  [N, n_classes]
                                                │
                                           Softmax  ──►  probabilities  [N, n_classes]
                                                │
                                           ArgMax ──Cast──Gather(classes_)  ──►  label

.. code-block:: python

    import numpy as np
    from lightgbm import LGBMClassifier
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = LGBMClassifier(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    label_onx, proba_onx = to_onnx(clf, (X,))

Supported input dtypes and opsets
==================================

The converter respects the input dtype (``float32`` or ``float64``) and
the active ``ai.onnx.ml`` opset:

* ``float32`` input with ``ai.onnx.ml`` opset 3 → ``TreeEnsembleRegressor``
  / ``TreeEnsembleClassifier`` (float32 weights)
* ``float64`` input with ``ai.onnx.ml`` opset 3 → legacy operators with
  float64 routing via an explicit ``Cast`` node
* ``float32`` or ``float64`` with ``ai.onnx.ml`` opset 5 → unified
  ``TreeEnsemble`` with matching weight dtype

Specifying the target opset:

.. code-block:: python

    onx = to_onnx(reg, (X.astype(np.float64),),
                  target_opset={"": 21, "ai.onnx.ml": 3})

Categorical features
====================

Integer-coded categorical features are supported.  Pass
``categorical_feature`` to LightGBM's ``fit()`` to mark which columns are
categorical, then convert as usual:

.. code-block:: python

    import numpy as np
    from lightgbm import LGBMRegressor
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X_num = rng.standard_normal((200, 3)).astype(np.float32)
    cat = rng.integers(0, 5, size=200).astype(np.float32)
    X = np.column_stack([X_num, cat])
    y = X_num[:, 0] + cat * 0.5

    reg = LGBMRegressor(n_estimators=10, random_state=0).fit(
        X, y, categorical_feature=[3]
    )
    onx = to_onnx(reg, (X,))

.. note::

    The input to the ONNX model must use the same integer encoding for
    categorical features as was used during training.  The converter does
    **not** perform any label-encoding or category mapping — it relies on
    the integer codes already being in the LightGBM-expected range.

Pipeline embedding
==================

:class:`lightgbm.LGBMRegressor` and :class:`lightgbm.LGBMClassifier` can
be used as the final step in a :class:`sklearn.pipeline.Pipeline`:

.. code-block:: python

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMClassifier
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LGBMClassifier(n_estimators=5, random_state=0)),
    ]).fit(X, y)

    label_onx, proba_onx = to_onnx(pipe, (X,))
