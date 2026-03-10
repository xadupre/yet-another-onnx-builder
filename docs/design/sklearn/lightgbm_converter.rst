.. _l-design-gbm-converters:

================================================
Gradient Boosting Converters (XGBoost, LightGBM)
================================================

:func:`yobx.sklearn.to_onnx` converts fitted
:class:`xgboost.XGBRegressor`, :class:`xgboost.XGBClassifier`,
:class:`lightgbm.LGBMRegressor`, and :class:`lightgbm.LGBMClassifier`
estimators to ONNX using the same registry-based architecture as the other
:mod:`yobx.sklearn` converters.

Both :epkg:`XGBoost` and :epkg:`LightGBM` are gradient-boosted tree
libraries — their fitted models consist of an ensemble of binary decision
trees.  The converters map these trees to ONNX ``TreeEnsemble*`` operators
and share the same high-level structure, differing only in how split
conditions are expressed and how the raw margin is post-processed.

Implementations:

* :mod:`yobx.sklearn.xgboost.xgb` — XGBoost converters
* :mod:`yobx.sklearn.lightgbm.lgbm` — LightGBM converters

Common ONNX encoding
====================

Both converters support two ``ai.onnx.ml`` encodings, selected
automatically based on the active opset:

* **Opset ≤ 4 (legacy)** — ``TreeEnsembleRegressor`` /
  ``TreeEnsembleClassifier`` with flat ``nodes_*`` / ``target_*`` (regressor)
  or ``class_*`` (classifier) attribute arrays.
* **Opset ≥ 5 (modern)** — unified ``TreeEnsemble`` operator with separate
  ``nodes_splits`` / ``leaf_weights`` tensor attributes.

Both encodings support ``float32`` and ``float64`` inputs.

XGBoost
=======

The implementation lives in :mod:`yobx.sklearn.xgboost.xgb` and is
registered automatically when
:func:`~yobx.sklearn.register_sklearn_converters` is called.

Tree structure
--------------

XGBoost trees are extracted via ``booster.get_dump(dump_format='json')``.
Internal nodes use *less-than* splits:

.. code-block:: text

    internal node
    ├── nodeid
    ├── split         (feature name, e.g. "f0")
    ├── split_condition  (float threshold)
    ├── yes           (taken when x < threshold)
    └── no            (taken when x >= threshold)

The ONNX ``BRANCH_LT`` mode (mode 1) matches XGBoost's exact semantics
(*go to yes-child when* ``x < threshold``).

XGBRegressor
------------

:func:`~yobx.sklearn.xgboost.xgb.sklearn_xgb_regressor` converts
:class:`xgboost.XGBRegressor`.

The ``base_score`` bias is read from ``booster.save_config()`` and added to
the raw margin before the objective-dependent output transform:

===========================================================  ==========================
Objective                                                    Output transform
===========================================================  ==========================
``reg:squarederror``, ``reg:absoluteerror``, …               Identity + base_score bias
``reg:logistic``                                             ``Sigmoid(margin)``
``count:poisson``, ``reg:gamma``, ``reg:tweedie``,           ``Exp(margin)``
``survival:cox``
===========================================================  ==========================

Unsupported objectives raise :class:`NotImplementedError`.

XGBClassifier
-------------

:func:`~yobx.sklearn.xgboost.xgb.sklearn_xgb_classifier` converts
:class:`xgboost.XGBClassifier`.

**Binary classification** (``n_classes == 2``):

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
                                     ArgMax ──Cast──Gather(classes)  ──►  label

**Multi-class classification** (``n_classes > 2``):

.. code-block:: text

    X  ──TreeEnsemble(n_classes targets)──►  raw_scores  [N, n_classes]
                                                │
                                           Softmax  ──►  probabilities  [N, n_classes]
                                                │
                                           ArgMax ──Cast──Gather(classes)  ──►  label

.. runpython::
    :showcode:

    import numpy as np
    from xgboost import XGBClassifier
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = XGBClassifier(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    onx = to_onnx(clf, (X,))
    print(pretty_onnx(onx))

LightGBM
========

The implementation lives in :mod:`yobx.sklearn.lightgbm.lgbm` and is
registered automatically when
:func:`~yobx.sklearn.register_sklearn_converters` is called.

Tree structure
--------------

LightGBM trees are extracted via ``booster_.dump_model()``.  Internal
nodes use *less-than-or-equal* splits and, for categorical features,
set-membership tests:

.. code-block:: text

    internal node
    ├── split_feature  (feature index)
    ├── threshold      (float value or category set like '0||1||2')
    ├── decision_type  ('<=' for numerical, '==' for categorical)
    ├── left_child     (taken when condition is TRUE)
    └── right_child    (taken when condition is FALSE)

**Numerical splits** use ``BRANCH_LEQ`` (mode 0): *go left when*
``x ≤ threshold``.

**Categorical splits** use ``decision_type == '=='`` with a threshold
string such as ``'0||1||2'``, meaning *go left if the feature value is 0,
1, or 2*.  Because ONNX only supports single-value ``BRANCH_EQ``
(``x == v``), the converter expands each multi-value node into a **chain
of single-value BRANCH_EQ nodes**:

.. code-block:: text

    feature IN {0, 1, 2}
      → BRANCH_EQ(feature==0): true→left_subtree, false→
          BRANCH_EQ(feature==1): true→left_subtree, false→
              BRANCH_EQ(feature==2): true→left_subtree, false→right_subtree

Shared ``left_subtree`` references across the chain are handled by a
memoised depth-first traversal, ensuring each unique subtree is assigned
exactly one flat node ID.

LGBMRegressor
-------------

:func:`~yobx.sklearn.lightgbm.lgbm.sklearn_lgbm_regressor` converts
:class:`lightgbm.LGBMRegressor`.

Unlike XGBoost, LightGBM leaf values are already shrinkage-scaled — there
is no separate ``base_score`` bias.  The raw sum of tree outputs equals the
final margin directly:

=======================================  =======================
Objective                                Output transform
=======================================  =======================
``regression``, ``regression_l1``,       Identity (no transform)
``huber``, ``quantile``, ``mape``, …
``poisson``, ``tweedie``                 ``Exp(margin)``
=======================================  =======================

Unsupported objectives raise :class:`NotImplementedError`.

LGBMClassifier
--------------

:func:`~yobx.sklearn.lightgbm.lgbm.sklearn_lgbm_classifier` converts
:class:`lightgbm.LGBMClassifier`.

The graph structure is identical to the XGBClassifier graphs above (binary
→ sigmoid + concat; multi-class → softmax), with ``LGBMClassifier``'s
``classes_`` used for the final label gather.

.. code-block:: python

    import numpy as np
    from lightgbm import LGBMRegressor
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4)).astype(np.float32)
    y = X[:, 0] + 2 * X[:, 1]

    reg = LGBMRegressor(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
    onx = to_onnx(reg, (X,))

Categorical features (LightGBM only)
-------------------------------------

Integer-coded categorical features are supported for LightGBM.  Pass
``categorical_feature`` to ``fit()`` to mark which columns are categorical,
then convert as usual:

.. runpython:: python
    :showcode:

    import numpy as np
    from lightgbm import LGBMRegressor
    from yobx.sklearn import to_onnx
    from yobx.helprs.onnx_helper import pretty_onnx

    rng = np.random.default_rng(0)
    X_num = rng.standard_normal((200, 3)).astype(np.float32)
    cat = rng.integers(0, 5, size=200).astype(np.float32)
    X = np.column_stack([X_num, cat])
    y = X_num[:, 0] + cat * 0.5

    reg = LGBMRegressor(n_estimators=10, random_state=0).fit(
        X, y, categorical_feature=[3]
    )
    onx = to_onnx(reg, (X,))
    print(pretty_onnx(onx))

.. note::

    XGBoost models with categorical features must encode them as regular
    numeric columns (e.g. using ``OrdinalEncoder`` or one-hot encoding) before
    training when using this converter.  The converter reads standard numeric
    XGBoost splits only.  LightGBM models support integer-coded categoricals
    natively via the ``categorical_feature`` argument.

Comparison: XGBoost vs LightGBM
================================

The table below summarises the key differences between the two converters:

==================================  ================================  ==============================
Property                            XGBoost                          LightGBM
==================================  ================================  ==============================
Tree dump method                    ``get_dump(dump_format='json')``  ``booster_.dump_model()``
Split direction (numerical)         ``x < threshold`` (BRANCH_LT)     ``x ≤ threshold`` (BRANCH_LEQ)
Categorical splits                  Not supported by converter        Expanded to BRANCH_EQ chains
Base-score bias                     Added from ``base_score`` cfg     None (baked into leaf values)
Regression transform                Identity / Sigmoid / Exp          Identity / Exp
Multi-class targets per round       ``n_classes`` trees               ``n_classes`` trees
Binary classifier raw outputs       1 tree per round (sigmoid)        1 tree per round (sigmoid)
==================================  ===============================   ==============================

Supported input dtypes and opsets
==================================

Both converters respect the input dtype and the active ``ai.onnx.ml`` opset:

* ``float32`` input with ``ai.onnx.ml`` opset 3 → ``TreeEnsembleRegressor``
  / ``TreeEnsembleClassifier`` (float32 weights)
* ``float64`` input with ``ai.onnx.ml`` opset 3 → legacy operators with
  float64 routing via an explicit ``Cast`` node
* ``float32`` or ``float64`` with ``ai.onnx.ml`` opset 5 → unified
  ``TreeEnsemble`` with matching weight dtype

Specifying the target opset:

.. code-block:: python

    onx = to_onnx(reg, (X.astype(np.float64),),
                  target_opset={"": 21, "ai.onnx.ml": 5})

Pipeline embedding
==================

Both :class:`xgboost.XGBRegressor` / :class:`xgboost.XGBClassifier` and
:class:`lightgbm.LGBMRegressor` / :class:`lightgbm.LGBMClassifier` can be
used as the final step in a :class:`sklearn.pipeline.Pipeline`:

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMClassifier
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LGBMClassifier(n_estimators=5, random_state=0)),
    ]).fit(X, y)

    onx = to_onnx(pipe, (X,))
    print(pretty_onnx(onx))

