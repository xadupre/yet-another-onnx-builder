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
