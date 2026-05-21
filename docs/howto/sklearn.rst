.. _l-howto-sklearn:

scikit-learn
============

This page answers common *"how do I…"* questions for converting
:epkg:`scikit-learn` estimators and pipelines to ONNX with
:func:`yobx.sklearn.to_onnx`.

.. contents:: On this page
   :local:
   :depth: 2

----

How to convert a single estimator
----------------------------------

Train a :epkg:`scikit-learn` estimator, then pass it together with a
representative dummy input (one row is enough) to
:func:`yobx.sklearn.to_onnx`:

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4)).astype(np.float32)

    scaler = StandardScaler().fit(X)
    onx = to_onnx(scaler, (X[:1],))
    print(f"Inputs : {[i.name for i in onx.graph.input]}")
    print(f"Outputs: {[o.name for o in onx.graph.output]}")

The dummy input controls the **dtype** and the **number of features** of
the generated ONNX graph; its batch dimension is replaced by a symbolic
dynamic axis automatically.

----

How to convert a Pipeline
--------------------------

:func:`yobx.sklearn.to_onnx` handles
:class:`~sklearn.pipeline.Pipeline` natively — each step is converted
in sequence and the resulting ONNX nodes are chained together:

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression())]
    ).fit(X, y)

    onx = to_onnx(pipe, (X[:1],))
    print(f"ONNX opset : {onx.opset_import[0].version}")
    print(f"Node types : {[n.op_type for n in onx.graph.node]}")

.. seealso::

    :ref:`l-plot-sklearn-pipeline` — a full runnable gallery example with
    output verification.

----

How to run the exported ONNX model
------------------------------------

Use :epkg:`onnxruntime` to run the converted model and compare its
outputs with :epkg:`scikit-learn`'s own predictions:

.. runpython::
    :showcode:

    import numpy as np
    import onnxruntime
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((80, 4)).astype(np.float32)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    pipe = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression())]
    ).fit(X_train, y_train)

    onx = to_onnx(pipe, (X_train[:1],))

    # Run with onnxruntime
    X_test = rng.standard_normal((20, 4)).astype(np.float32)
    sess = onnxruntime.InferenceSession(
        onx.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    label_onnx, proba_onnx = sess.run(None, {input_name: X_test})

    # Compare with scikit-learn
    label_sk = pipe.predict(X_test)
    assert (label_sk == label_onnx).all(), "Label mismatch!"
    print("Labels match ✓")
    print(f"First 5 labels (sklearn): {label_sk[:5]}")
    print(f"First 5 labels (ONNX)   : {label_onnx[:5]}")

----

How to control dynamic shapes
-------------------------------

By default the batch dimension (axis 0) of every input is made dynamic.
Pass ``dynamic_shapes`` to name that axis explicitly or to mark additional
axes as symbolic:

.. code-block:: python

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4)).astype(np.float32)

    scaler = StandardScaler().fit(X)

    # axis 0 is dynamic and named "batch"
    onx = to_onnx(scaler, (X[:1],), dynamic_shapes=({0: "batch"},))

Pass an empty tuple (``dynamic_shapes=()``) to produce a fully **static**
graph where every dimension is fixed at conversion time:

.. code-block:: python

    onx_static = to_onnx(scaler, (X[:1],), dynamic_shapes=())

----

How to inspect the ONNX graph
------------------------------

Print a compact text representation of the model with
:func:`~yobx.helpers.onnx_helper.pretty_onnx`:

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from yobx.sklearn import to_onnx
    from yobx.helpers.onnx_helper import pretty_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4)).astype(np.float32)

    scaler = StandardScaler().fit(X)
    onx = to_onnx(scaler, (X[:1],))
    print(pretty_onnx(onx))

----

How to save and reload the ONNX model
---------------------------------------

The :class:`~yobx.container.ExportArtifact` returned by
:func:`yobx.sklearn.to_onnx` can be serialised directly to disk and
loaded again later:

.. code-block:: python

    import numpy as np
    import onnx
    from sklearn.preprocessing import StandardScaler
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4)).astype(np.float32)

    scaler = StandardScaler().fit(X)
    onx = to_onnx(scaler, (X[:1],))

    # Save
    onnx.save(onx, "scaler.onnx")

    # Reload
    onx_loaded = onnx.load("scaler.onnx")

.. seealso::

    :ref:`l-sklearn-converter` — full reference for the scikit-learn
    converter, including the converter registry and how to add support for
    custom estimators.

    :ref:`l-plot-sklearn-pipeline` — runnable gallery example.

    :ref:`l-plot-sklearn-function-options` — exporting each pipeline step
    as a separate ONNX local function.

----

How to export a custom estimator
----------------------------------

There are two ways to make :func:`~yobx.sklearn.to_onnx` work with an
estimator that has no built-in converter.

**Option 1 — TraceableMixin (numpy-based transformers)**

If the ``transform`` method uses only standard :epkg:`numpy` operations,
inherit from :class:`~yobx.sklearn.TraceableMixin` together with the usual
sklearn base classes.  The framework traces the method automatically — no
converter function is needed:

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.base import BaseEstimator, TransformerMixin
    from yobx.sklearn import to_onnx, TraceableMixin

    class LogNormTransformer(BaseEstimator, TransformerMixin, TraceableMixin):
        def fit(self, X, y=None):
            self.scale_ = np.abs(X).mean(axis=0, keepdims=True).astype(np.float32)
            return self

        def transform(self, X):
            return np.log(np.abs(X) / self.scale_ + np.float32(1))

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    est = LogNormTransformer().fit(X)
    onx = to_onnx(est, (X[:1],))
    print(f"Nodes: {[n.op_type for n in onx.graph.node]}")

**Option 2 — extra_converters (full control)**

For estimators whose logic cannot be expressed as plain numpy ops — or when
you need fine-grained control over the ONNX graph — write a converter
function and pass it via ``extra_converters``:

.. runpython::
    :showcode:

    import numpy as np
    import onnxruntime
    from sklearn.base import BaseEstimator, TransformerMixin
    from yobx.sklearn import to_onnx
    from yobx.helpers.onnx_helper import tensor_dtype_to_np_dtype

    class ClipTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, clip_min=0.0, clip_max=1.0):
            self.clip_min = clip_min
            self.clip_max = clip_max

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return np.clip(X, self.clip_min, self.clip_max)

    def convert_clip(g, sts, outputs, estimator, X, name="clip"):
        dtype = tensor_dtype_to_np_dtype(g.get_type(X))
        low = np.array(estimator.clip_min, dtype=dtype)
        high = np.array(estimator.clip_max, dtype=dtype)
        res = g.op.Clip(X, low, high, outputs=outputs, name=name)
        g.set_type_shape_unary_op(res, X)
        return res

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    transformer = ClipTransformer(clip_min=-0.5, clip_max=0.5).fit(X)
    onx = to_onnx(
        transformer,
        (X[:1],),
        extra_converters={ClipTransformer: convert_clip},
    )
    print(f"Nodes: {[n.op_type for n in onx.graph.node]}")

    sess = onnxruntime.InferenceSession(
        onx.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    X_test = rng.standard_normal((5, 4)).astype(np.float32)
    (clipped,) = sess.run(None, {"X": X_test})
    expected = transformer.transform(X_test)
    assert np.allclose(clipped, expected, atol=1e-6)
    print("Results match ✓")

.. seealso::

    :ref:`l-plot-sklearn-custom-converter-options` — a full gallery example
    showing a custom converter with optional extra outputs.

    :ref:`l-sklearn-converter` — converter registry and how to write a
    converter for any estimator.

----

How to export with FunctionTransformer
----------------------------------------

:class:`~sklearn.preprocessing.FunctionTransformer` wraps any numpy function
as a scikit-learn transformer.  Because its ``func`` is a plain numpy
function, :func:`~yobx.sklearn.to_onnx` converts it via numpy tracing — no
custom converter is required.

**Basic usage**

.. runpython::
    :showcode:

    import numpy as np
    import onnxruntime
    from sklearn.preprocessing import FunctionTransformer
    from yobx.sklearn import to_onnx

    def log1p_abs(X):
        return np.log1p(np.abs(X))

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    transformer = FunctionTransformer(func=log1p_abs).fit(X)
    onx = to_onnx(transformer, (X[:1],))
    print(f"Nodes: {[n.op_type for n in onx.graph.node]}")

    sess = onnxruntime.InferenceSession(
        onx.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    X_test = rng.standard_normal((5, 4)).astype(np.float32)
    (onnx_out,) = sess.run(None, {"X": X_test})
    expected = transformer.transform(X_test).astype(np.float32)
    assert np.allclose(onnx_out, expected, atol=1e-5)
    print("Results match ✓")

**Passing keyword arguments with kw_args**

Constants can be forwarded to the function via ``kw_args``; the converter
folds them into the ONNX graph as initializers:

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.preprocessing import FunctionTransformer
    from yobx.sklearn import to_onnx

    def scale_shift(X, scale=np.float32(1), shift=np.float32(0)):
        return X * scale + shift

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    transformer = FunctionTransformer(
        func=scale_shift,
        kw_args={"scale": np.float32(2.0), "shift": np.float32(1.0)},
    ).fit(X)
    onx = to_onnx(transformer, (X[:1],))
    print(f"Nodes: {[n.op_type for n in onx.graph.node]}")

**Identity transformer (func=None)**

When ``func=None`` the transformer is a no-op; the converter emits a single
``Identity`` node:

.. runpython::
    :showcode:

    import numpy as np
    from sklearn.preprocessing import FunctionTransformer
    from yobx.sklearn import to_onnx

    X = np.ones((5, 3), dtype=np.float32)
    identity_tf = FunctionTransformer(func=None).fit(X)
    onx = to_onnx(identity_tf, (X[:1],))
    print(f"Nodes: {[n.op_type for n in onx.graph.node]}")

.. seealso::

    :ref:`l-plot-sklearn-function-transformer` — a full gallery example that
    also shows standalone numpy tracing and pipeline embedding.

    :ref:`l-design-function-transformer-tracing` — design doc explaining the
    numpy tracing mechanism.
