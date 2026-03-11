.. _l-design-sklearn-converter:

=================
Sklearn Converter
=================

:func:`yobx.sklearn.to_onnx` converts a fitted :epkg:`scikit-learn`
estimator into an :class:`onnx.ModelProto`.  The conversion is
powered by :class:`yobx.xbuilder.GraphBuilder` and follows a
**registry-based** design: each estimator class maps to a dedicated
converter function that emits the required ONNX nodes.

High-level workflow
===================

.. code-block:: text

    fitted estimator
          │
          ▼
      to_onnx()          ← builds GraphBuilder, looks up converter
          │
          ▼
    converter function   ← adds ONNX nodes via GraphBuilder.op.*
          │
          ▼
      GraphBuilder.to_onnx()   ← validates and returns ModelProto

1. :func:`to_onnx <yobx.sklearn.to_onnx>` accepts the fitted estimator,
   representative dummy inputs (used to infer dtype and shape), and
   optional ``input_names`` / ``dynamic_shapes``.
2. It calls :func:`register_sklearn_converters
   <yobx.sklearn.register_sklearn_converters>` (idempotent) to populate
   the global registry on first use.
3. It constructs a :class:`GraphBuilder <yobx.xbuilder.GraphBuilder>` and
   declares one graph input per dummy array via
   :meth:`make_tensor_input <yobx.xbuilder.GraphBuilder.make_tensor_input>`.
4. It looks up the converter for ``type(estimator)`` and calls it.
5. Each graph output is declared with
   :meth:`make_tensor_output <yobx.xbuilder.GraphBuilder.make_tensor_output>`.
6. :meth:`GraphBuilder.to_onnx <yobx.xbuilder.GraphBuilder.to_onnx>`
   finalises and returns the model.

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from yobx.sklearn import to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4)).astype(np.float32)

    scaler = StandardScaler().fit(X)
    model = to_onnx(scaler, (X,))
    print(pretty_onnx(model))


Converter registry
==================

The registry is a plain module-level dictionary
``SKLEARN_CONVERTERS: Dict[type, Callable]`` defined in
:mod:`yobx.sklearn.register`.

Registering a converter
-----------------------

Use the :func:`register_sklearn_converter
<yobx.sklearn.register.register_sklearn_converter>` decorator.
Pass a single class or a tuple of classes as the first argument:

.. code-block:: python

    from yobx.sklearn.register import register_sklearn_converter
    from yobx.typing import GraphBuilderExtendedProtocol
    from yobx.xbuilder import GraphBuilder

    @register_sklearn_converter(MyEstimator)
    def convert_my_estimator(
        g: GraphBuilderExtendedProtocol,
        sts: dict,
        outputs: list[str],
        estimator: MyEstimator,
        X: str,
        name: str = "my_estimator",
    ) -> str:
        ...

The decorator raises :class:`TypeError` if a converter is already
registered for the same class, preventing accidental double-registration.

Looking up a converter
----------------------

:func:`get_sklearn_converter <yobx.sklearn.register.get_sklearn_converter>`
takes a class and returns the registered callable, raising
:class:`ValueError` if none is found.

Converter function signature
============================

Every converter follows the same contract:

``(g, sts, outputs, estimator, *input_names, name) → output_name(s)``

=============  =====================================================
Parameter      Description
=============  =====================================================
``g``          :class:`GraphBuilder <yobx.xbuilder.GraphBuilder>`
               — call ``g.op.<OpType>(…)`` to emit ONNX nodes.
``sts``        ``Dict`` of shape/type metadata provided by
               :epkg:`scikit-learn` (empty ``{}`` in the default
               path; reserved for future shape propagation).
``outputs``    ``List[str]`` of pre-allocated output tensor names
               that the converter **must** write to.
``estimator``  The fitted :epkg:`scikit-learn` object.
``*inputs``    One positional ``str`` argument per graph input
               (the tensor name in the graph).
``name``       String prefix used when generating unique node names
               via ``g.op``.
=============  =====================================================

The function must return the output tensor name (``str``) for
single-output estimators, or a tuple of names for multi-output ones
(e.g. classifiers that produce both a label and probabilities).

Output naming
=============

:func:`get_output_names <yobx.sklearn.sklearn_helper.get_output_names>`
determines the list of output tensor names for an estimator:

* **Transformers** that expose ``get_feature_names_out()`` use those
  names (collapsed to a common prefix via
  :func:`longest_prefix <yobx.sklearn.sklearn_helper.longest_prefix>`
  when more than one output is expected).
* **Classifiers** default to ``["label", "probabilities"]``.
* **Regressors** default to ``["predictions"]``.
* Everything else defaults to ``["Y"]``.

Implemented converters
======================

StandardScaler
--------------

:func:`sklearn_standard_scaler
<yobx.sklearn.preprocessing.standard_scaler.sklearn_standard_scaler>`
converts :class:`sklearn.preprocessing.StandardScaler`.

The implementation respects the ``with_mean`` and ``with_std`` flags:

.. code-block:: text

    X  ──Sub(mean)──►  centered  ──Div(scale)──►  output
         (if with_mean)               (if with_std)

When ``with_mean=False`` the ``Sub`` node is skipped; when
``with_std=False`` the ``Div`` node is replaced by an ``Identity``.

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from yobx.sklearn import to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    rng = np.random.default_rng(1)
    X = rng.standard_normal((5, 3)).astype(np.float32)
    model = to_onnx(StandardScaler(with_std=False).fit(X), (X,))
    print(pretty_onnx(model))

LogisticRegression
------------------

:func:`sklearn_logistic_regression
<yobx.sklearn.linear_model.logistic_regression.sklearn_logistic_regression>`
converts :class:`sklearn.linear_model.LogisticRegression` and
:class:`sklearn.linear_model.LogisticRegressionCV`.

The graph structure depends on the number of classes:

**Binary classification** (``coef_.shape[0] == 1``):

.. code-block:: text

    X  ──Gemm(coef, intercept)──►  decision
                                       │
                              ┌────────┴────────┐
                           Sigmoid           Sub(1, ·)
                              │                  │
                           proba_pos          proba_neg
                              └────────┬────────┘
                                    Concat  ──►  probabilities
                                       │
                                    ArgMax ──Cast──Gather(classes) ──►  label

**Multiclass** (``coef_.shape[0] > 1``):

.. code-block:: text

    X  ──Gemm(coef, intercept)──►  decision
                                       │
                                   Softmax  ──►  probabilities
                                       │
                                   ArgMax ──Cast──Gather(classes)  ──►  label

PCA
---

:func:`sklearn_pca
<yobx.sklearn.decomposition.pca.sklearn_pca>`
converts :class:`sklearn.decomposition.PCA`.

The implementation centres the input (when ``mean_`` is not ``None``) and
then projects it onto the principal components:

.. code-block:: text

    X  ──Sub(mean_)──►  centered  ──MatMul(components_.T)──►  output
         (if mean_ is not None)

When ``mean_`` is ``None`` the ``Sub`` node is skipped.

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.decomposition import PCA
    from yobx.sklearn import to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4)).astype(np.float32)
    model = to_onnx(PCA(n_components=2).fit(X), (X,))
    print(pretty_onnx(model))

Pipeline
--------

:func:`sklearn_pipeline
<yobx.sklearn.pipeline.pipeline.sklearn_pipeline>`
converts :class:`sklearn.pipeline.Pipeline` by iterating over the
pipeline steps and chaining each step's converter output into the
next step's input.  Intermediate tensor names are generated with
:meth:`GraphBuilder.unique_name <yobx.xbuilder.GraphBuilder.unique_name>`
to avoid collisions.

.. gdot::
    :script: DOT-SECTION

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from yobx.sklearn import to_onnx
    from yobx.helpers.dot_helper import to_dot

    rng = np.random.default_rng(2)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ]).fit(X, y)

    model = to_onnx(pipe, (X,))
    print("DOT-SECTION", to_dot(model))

StackingRegressor
-----------------

:func:`sklearn_stacking_regressor
<yobx.sklearn.ensemble.stacking.sklearn_stacking_regressor>`
converts :class:`sklearn.ensemble.StackingRegressor`.

For each base estimator the ``predict`` converter is called.  The 1-D
prediction vector is reshaped to ``(N, 1)`` and all base-estimator outputs
are concatenated along axis 1 to form the meta-feature matrix.  When
``passthrough=True`` the original input features are appended to the
meta-feature matrix before the final estimator is applied.

.. code-block:: text

    X ──[est 0 converter]──► pred_0 (N,) ──Reshape(N,1)──┐
    X ──[est 1 converter]──► pred_1 (N,) ──Reshape(N,1)──┤
    ...                                                    │
                                          Concat(axis=1) ─►meta (N, n_est)
                                                           │
                                [Concat(meta, X, axis=1)] ─┤ (passthrough only)
                                                           │
                                    [final_estimator] ──────► predictions

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor
    from yobx.sklearn import to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4)).astype(np.float32)
    y = rng.standard_normal(30).astype(np.float32)

    est = StackingRegressor(
        estimators=[
            ("dt", DecisionTreeRegressor(max_depth=2, random_state=0)),
            ("ridge", Ridge()),
        ],
        final_estimator=Ridge(),
    ).fit(X, y)

    model = to_onnx(est, (X,))
    print(pretty_onnx(model))

StackingClassifier
------------------

:func:`sklearn_stacking_classifier
<yobx.sklearn.ensemble.stacking.sklearn_stacking_classifier>`
converts :class:`sklearn.ensemble.StackingClassifier`.

The meta-feature matrix is assembled by calling each base estimator's
registered converter and extracting the appropriate output columns,
matching the behaviour of
:meth:`sklearn.ensemble.StackingClassifier._concatenate_predictions`:

* ``stack_method_ == 'predict_proba'`` — **binary**: only column 1 of the
  probability matrix is kept (shape ``(N, 1)``); **multiclass**: all columns
  are kept (shape ``(N, n_classes)``).
* ``stack_method_ == 'predict'`` — the label output is cast to the input
  float dtype and reshaped to ``(N, 1)``.

When ``passthrough=True`` the original input features are appended to the
meta-feature matrix before the final estimator is applied.

**Binary classification** (``predict_proba``, ``len(classes_) == 2``):

.. code-block:: text

    X ──[est 0 converter]──► proba_0 (N,2) ──Slice[:,1:]──► (N,1)──┐
    X ──[est 1 converter]──► proba_1 (N,2) ──Slice[:,1:]──► (N,1)──┤
    ...                                                              │
                                                Concat(axis=1) ─────► meta (N, n_est)
                                                                     │
                                      [Concat(meta, X, axis=1)] ─────┤ (passthrough only)
                                                                     │
                                          [final_estimator] ─────────► label [, probabilities]

**Multiclass classification** (``predict_proba``, ``len(classes_) > 2``):

.. code-block:: text

    X ──[est 0 converter]──► proba_0 (N,C) ──┐
    X ──[est 1 converter]──► proba_1 (N,C) ──┤
    ...                                       │
                                 Concat(axis=1)──► meta (N, n_est * n_classes)
                                              │
                      [Concat(meta, X, axis=1)]──┤ (passthrough only)
                                              │
                          [final_estimator] ───► label [, probabilities]

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from yobx.sklearn import to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4)).astype(np.float32)
    y = (rng.standard_normal(30) > 0).astype(int)

    est = StackingClassifier(
        estimators=[
            ("dt", DecisionTreeClassifier(max_depth=2, random_state=0)),
            ("lr", LogisticRegression(max_iter=200)),
        ],
        final_estimator=LogisticRegression(max_iter=200),
    ).fit(X, y)

    model = to_onnx(est, (X,))
    print(pretty_onnx(model))

Adding a new converter
======================

To support a new :epkg:`scikit-learn` estimator:

1. Create a new file (e.g. ``yobx/sklearn/ensemble/random_forest.py``).
2. Implement a converter function following the signature described above.
3. Decorate it with ``@register_sklearn_converter(MyEstimator)``.
4. Add an import in the matching ``register()`` function so the converter
   is loaded when :func:`register_sklearn_converters
   <yobx.sklearn.register_sklearn_converters>` is called.

.. code-block:: python

    # yobx/sklearn/ensemble/random_forest.py
    from sklearn.ensemble import RandomForestClassifier
    from ...typing import GraphBuilderExtendedProtocol
    from ..register import register_sklearn_converter


    @register_sklearn_converter(RandomForestClassifier)
    def convert_random_forest_classifier(
        g: GraphBuilderExtendedProtocol,
        sts: dict,
        outputs: list[str],
        estimator: RandomForestClassifier,
        X: str,
        name: str = "random_forest",
    ):
        # ... emit ONNX nodes via g.op.*
        ...
