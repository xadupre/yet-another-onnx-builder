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
        return result

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
    from yobx.typing import GraphBuilderExtendedProtocol
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
        g: GraphBuilderExtendedProtocol,
        sts: dict,
        outputs: list,
        estimator: ScaleByConstant,
        X: str,
        name: str = "scale",
    ) -> str:
        scale = np.array([estimator.scale], dtype=np.float32)
        result = g.op.Mul(X, scale, name=name, outputs=outputs)
        return result

Once this module is imported the converter is available globally and
:func:`to_onnx <yobx.sklearn.to_onnx>` will use it automatically:

.. code-block:: python

    import myproject.onnx_converters  # registers the converter

    from yobx.sklearn import to_onnx

    onx = to_onnx(ScaleByConstant(scale=3.0).fit(X), (X,))
    # no extra_converters needed

Using convert options in a custom converter
===========================================

The ``convert_options`` parameter of :func:`to_onnx <yobx.sklearn.to_onnx>`
lets callers request **optional extra outputs** from a converter without
changing the converter signature.  This pattern is used by the built-in tree
converters (``decision_path``, ``decision_leaf``) and is fully available to
custom converters.

How it works
------------

1. **Define an options class** — any object that satisfies the
   :class:`~yobx.typing.ConvertOptionsProtocol` (two methods:
   ``available_options()`` and ``has()``).

2. **``available_options()``** returns the list of option names your class
   recognises.  The framework iterates this list before calling the converter
   and pre-allocates one extra slot in ``outputs`` for every option that
   ``has()`` returns ``True`` for.

3. **``has(option_name, piece, name=None)``** returns ``True`` when the option
   is active for the estimator *piece*.  The optional *name* argument carries
   the pipeline step name so you can enable an option only for a specific
   named step in a :class:`~sklearn.pipeline.Pipeline`.

4. **Check inside the converter** — use ``g.convert_options.has(...)`` to
   decide whether to emit the optional nodes and fill ``outputs[extra_idx]``.

Minimal example
---------------

The snippet below defines a ``ClipTransformer`` whose ONNX converter
optionally emits a boolean clip-mask output controlled by a custom options
class:

.. code-block:: python

    import numpy as np
    from sklearn.base import BaseEstimator, TransformerMixin
    from yobx.sklearn import to_onnx
    from yobx.helpers.onnx_helper import tensor_dtype_to_np_dtype
    from yobx.typing import GraphBuilderExtendedProtocol


    class ClipTransformer(TransformerMixin, BaseEstimator):
        def __init__(self, clip_min=0.0, clip_max=1.0):
            self.clip_min = clip_min
            self.clip_max = clip_max

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return np.clip(X, self.clip_min, self.clip_max)


    class ClipOptions:
        def __init__(self, clip_mask=False):
            self.clip_mask = clip_mask

        def available_options(self):
            return ["clip_mask"]

        def has(self, option_name, piece, name=None):
            if option_name == "clip_mask":
                return bool(self.clip_mask) and hasattr(piece, "clip_min")
            return False


    def convert_clip_transformer(
        g: GraphBuilderExtendedProtocol,
        sts: dict,
        outputs: list,
        estimator: ClipTransformer,
        X: str,
        name: str = "clip",
    ):
        itype = g.get_type(X)
        dtype = tensor_dtype_to_np_dtype(itype)
        low = np.array(estimator.clip_min, dtype=dtype)
        high = np.array(estimator.clip_max, dtype=dtype)

        clipped = g.op.Clip(X, low, high, name=name, outputs=outputs[:1])
        if not sts:
            g.set_type_shape_unary_op(clipped, X)

        if g.convert_options.has("clip_mask", estimator, name):
            below = g.op.Less(X, low, name=f"{name}_below")
            above = g.op.Greater(X, high, name=f"{name}_above")
            mask = g.op.Or(below, above, name=f"{name}_mask", outputs=outputs[1:2])

        return outputs[0] if len(outputs) == 1 else tuple(outputs)


    X = np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32)
    transformer = ClipTransformer(clip_min=-0.5, clip_max=0.5).fit(X)

    # Without options: single output
    onx = to_onnx(
        transformer,
        (X,),
        extra_converters={ClipTransformer: convert_clip_transformer},
    )

    # With clip_mask: two outputs
    onx_with_mask = to_onnx(
        transformer,
        (X,),
        extra_converters={ClipTransformer: convert_clip_transformer},
        convert_options=ClipOptions(clip_mask=True),
    )
    print([o.name for o in onx_with_mask.graph.output])
    # ['Y', 'clip_mask']

.. seealso::

    :ref:`l-plot-sklearn-custom-converter-options` — a full runnable
    gallery example with numerical validation.

    :ref:`l-plot-sklearn-convert-options` — how the built-in
    ``decision_path`` and ``decision_leaf`` options work for tree and
    ensemble models.

    :ref:`l-design-expected-api` — the full ``convert_options`` protocol
    contract and the built-in :class:`~yobx.sklearn.ConvertOptions`
    reference.

.. seealso::

    :ref:`l-design-sklearn-converter` — overview of the converter
    registry, the built-in converters, and how to add a new converter to
    the package itself.
