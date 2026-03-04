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
    from yobx.xbuilder import GraphBuilder

    @register_sklearn_converter(MyEstimator)
    def convert_my_estimator(
        g: GraphBuilder,
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

Pipeline
--------

:func:`sklearn_pipeline
<yobx.sklearn.pipeline.pipeline.sklearn_pipeline>`
converts :class:`sklearn.pipeline.Pipeline` by iterating over the
pipeline steps and chaining each step's converter output into the
next step's input.  Intermediate tensor names are generated with
:meth:`GraphBuilder.unique_name <yobx.xbuilder.GraphBuilder.unique_name>`
to avoid collisions.

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from yobx.sklearn import to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    rng = np.random.default_rng(2)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ]).fit(X, y)

    model = to_onnx(pipe, (X,))
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
    from ..register import register_sklearn_converter
    from ...xbuilder import GraphBuilder


    @register_sklearn_converter(RandomForestClassifier)
    def convert_random_forest_classifier(
        g: GraphBuilder,
        sts: dict,
        outputs: list[str],
        estimator: RandomForestClassifier,
        X: str,
        name: str = "random_forest",
    ):
        # ... emit ONNX nodes via g.op.*
        ...
