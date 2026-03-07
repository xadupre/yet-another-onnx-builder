.. _l-design-sklearn-custom-converter:

================
Custom Converter
================

The built-in converter registry covers estimators that ship with
:epkg:`scikit-learn`.  When you train a **custom estimator** — or want
to override how a built-in estimator is translated — you can supply your
own converter without touching the package source.

There are two ways:

* **Ad-hoc** via the ``extra_converters`` parameter of
  :func:`to_onnx <yobx.sklearn.to_onnx>` — useful for one-off
  conversions or during development.
* **Permanent** via the
  :func:`register_sklearn_converter
  <yobx.sklearn.register.register_sklearn_converter>` decorator — the
  right choice once a converter is stable and reusable.

Writing a converter function
============================

A converter follows the same contract as all built-in ones:

``(g, sts, outputs, estimator, *input_names, name) → output_name(s)``

================  =====================================================
Parameter         Description
================  =====================================================
``g``             :class:`GraphBuilder <yobx.xbuilder.GraphBuilder>` —
                  call ``g.op.<OpType>(…)`` to emit ONNX nodes.
``sts``           ``Dict`` of metadata (empty ``{}`` in the default
                  path; reserved for future shape propagation).
``outputs``       ``List[str]`` of pre-allocated output tensor names
                  that the converter **must** write to.
``estimator``     The fitted :epkg:`scikit-learn` object.
``*input_names``  One positional ``str`` per graph input tensor.
``name``          String prefix for unique node-name generation.
================  =====================================================

After writing the output tensor the converter should propagate type
and shape information so that downstream nodes can use them:

.. code-block:: python

    if not sts:
        g.set_type(result, g.get_type(X))
        g.set_shape(result, g.get_shape(X))
        if g.has_device(X):
            g.set_device(result, g.get_device(X))

Ad-hoc conversion with ``extra_converters``
===========================================

Pass a ``{EstimatorClass: converter_function}`` mapping to the
``extra_converters`` keyword argument.  Entries in that mapping take
**priority** over built-in converters, so you can also override an
existing converter this way.

The example below defines a custom ``ScaleByConstant`` transformer and
its corresponding ONNX converter, then converts an instance to ONNX and
validates the result numerically.

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.base import BaseEstimator, TransformerMixin
    from yobx.sklearn import to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    # ── 1. Custom estimator ────────────────────────────────────────────

    class ScaleByConstant(TransformerMixin, BaseEstimator):
        """Multiplies every feature by a fixed scalar constant."""

        def __init__(self, scale=2.0):
            self.scale = scale

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X * self.scale

    # ── 2. Converter function ──────────────────────────────────────────

    def convert_scale_by_constant(g, sts, outputs, estimator, X, name="scale"):
        """Emits a single ``Mul`` node: output = X * estimator.scale."""
        scale = np.array([estimator.scale], dtype=np.float32)
        result = g.op.Mul(X, scale, name=name, outputs=outputs)
        if not sts:
            g.set_type(result, g.get_type(X))
            g.set_shape(result, g.get_shape(X))
            if g.has_device(X):
                g.set_device(result, g.get_device(X))
        return result

    # ── 3. Convert ─────────────────────────────────────────────────────

    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 3)).astype(np.float32)

    est = ScaleByConstant(scale=3.0).fit(X)
    onx = to_onnx(est, (X,), extra_converters={ScaleByConstant: convert_scale_by_constant})
    print(pretty_onnx(onx))

Validate numerically
====================

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.base import BaseEstimator, TransformerMixin
    from yobx.sklearn import to_onnx
    from yobx.reference import ExtendedReferenceEvaluator

    class ScaleByConstant(TransformerMixin, BaseEstimator):
        def __init__(self, scale=2.0):
            self.scale = scale

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X * self.scale

    def convert_scale_by_constant(g, sts, outputs, estimator, X, name="scale"):
        scale = np.array([estimator.scale], dtype=np.float32)
        result = g.op.Mul(X, scale, name=name, outputs=outputs)
        if not sts:
            g.set_type(result, g.get_type(X))
            g.set_shape(result, g.get_shape(X))
            if g.has_device(X):
                g.set_device(result, g.get_device(X))
        return result

    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 3)).astype(np.float32)

    est = ScaleByConstant(scale=3.0).fit(X)
    onx = to_onnx(est, (X,), extra_converters={ScaleByConstant: convert_scale_by_constant})

    ref = ExtendedReferenceEvaluator(onx)
    onnx_output = ref.run(None, {"X": X})[0]
    sklearn_output = est.transform(X).astype(np.float32)

    print("max absolute difference:", np.abs(onnx_output - sklearn_output).max())

Overriding a built-in converter
================================

Because ``extra_converters`` entries take priority, you can also replace
the converter for a built-in estimator.  The snippet below replaces the
standard :class:`sklearn.preprocessing.StandardScaler` converter with a
trivial identity (just to illustrate the override mechanism):

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from yobx.sklearn import to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    def identity_scaler(g, sts, outputs, estimator, X, name="scaler"):
        """Pass-through: return the input unchanged."""
        result = g.op.Identity(X, name=name, outputs=outputs)
        if not sts:
            g.set_type(result, g.get_type(X))
            g.set_shape(result, g.get_shape(X))
            if g.has_device(X):
                g.set_device(result, g.get_device(X))
        return result

    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 2)).astype(np.float32)
    ss = StandardScaler().fit(X)

    # The custom converter overrides the built-in one
    onx = to_onnx(ss, (X,), extra_converters={StandardScaler: identity_scaler})
    print(pretty_onnx(onx))

Permanent registration
======================

Once your converter is stable, promote it from an ad-hoc function to a
first-class entry in the registry by using the
:func:`register_sklearn_converter
<yobx.sklearn.register.register_sklearn_converter>` decorator.  This
means you no longer have to pass ``extra_converters`` at every call site:

.. code-block:: python

    # myproject/onnx_converters.py
    import numpy as np
    from sklearn.base import BaseEstimator, TransformerMixin
    from yobx.sklearn.register import register_sklearn_converter
    from yobx.typing import GraphBuilderProtocolExtended
    from yobx.xbuilder import GraphBuilder


    class ScaleByConstant(TransformerMixin, BaseEstimator):
        def __init__(self, scale=2.0):
            self.scale = scale

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X * self.scale


    @register_sklearn_converter(ScaleByConstant)
    def convert_scale_by_constant(
        g: GraphBuilderProtocolExtended,
        sts: dict,
        outputs: list,
        estimator: ScaleByConstant,
        X: str,
        name: str = "scale",
    ) -> str:
        scale = np.array([estimator.scale], dtype=np.float32)
        result = g.op.Mul(X, scale, name=name, outputs=outputs)
        if not sts:
            g.set_type(result, g.get_type(X))
            g.set_shape(result, g.get_shape(X))
            if g.has_device(X):
                g.set_device(result, g.get_device(X))
        return result

Once this module is imported the converter is available globally and
:func:`to_onnx <yobx.sklearn.to_onnx>` will use it automatically:

.. code-block:: python

    import myproject.onnx_converters  # registers the converter

    from yobx.sklearn import to_onnx

    onx = to_onnx(ScaleByConstant(scale=3.0).fit(X), (X,))
    # no extra_converters needed

.. seealso::

    :ref:`l-design-sklearn-converter` — overview of the converter
    registry, the built-in converters, and how to add a new converter to
    the package itself.
