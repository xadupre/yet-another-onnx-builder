.. _l-getting-started:

Getting Started
===============

This page walks you through the first steps with
**yet-another-onnx-builder** (``yobx``).
See :ref:`l-install` for installation instructions.

The one-line API
----------------

Every framework converter in ``yobx`` follows the same pattern:

.. code-block:: python

    expected = model(*args, **kwargs)          # run the model once
    onnx_model = to_onnx(model, args, kwargs)  # export to ONNX

The call signatures are intentionally uniform so that you can swap
frameworks without learning a new API each time.

scikit-learn
------------

Convert any fitted :epkg:`scikit-learn` estimator or pipeline with
:func:`yobx.sklearn.to_onnx`:

.. code-block:: python

    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from yobx.sklearn import to_onnx

    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ]).fit(X, y)

    onnx_model = to_onnx(pipe, (X,))

By default ``to_onnx`` marks axis 0 of every input as dynamic.  To
customize the dynamic axes, pass a ``dynamic_shapes`` tuple — one
``Dict[int, str]`` per input, mapping axis indices to symbolic dimension
names:

.. code-block:: python

    onnx_model = to_onnx(pipe, (X,), dynamic_shapes=({0: "batch"},))

See :ref:`l-sklearn-converter` for the full scikit-learn conversion guide.

PyTorch
-------

Convert a :class:`torch.nn.Module` with :func:`yobx.torch.interpreter.to_onnx`:

.. code-block:: python

    import torch
    from yobx.torch import to_onnx

    class MyModel(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x)

    model = MyModel()
    x = torch.randn(4, 8)
    onnx_model = to_onnx(model, (x,))

To mark axis 0 as a dynamic batch dimension, pass a ``dynamic_shapes``
dict following the :mod:`torch.export` convention:

.. code-block:: python

    import torch
    from torch.export import Dim
    from yobx.torch import to_onnx

    class MyModel(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x)

    model = MyModel()
    x = torch.randn(4, 8)

    batch = Dim("batch")
    onnx_model = to_onnx(model, (x,), dynamic_shapes={"x": {0: batch}})

See :ref:`l-torch-converter` for the full PyTorch conversion guide,
including dynamic shapes, model patching, and large-model support.

TensorFlow / Keras
------------------

Convert a :epkg:`Keras` model with :func:`yobx.tensorflow.to_onnx`:

.. code-block:: python

    import numpy as np
    import tensorflow as tf
    from yobx.tensorflow import to_onnx

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(2),
    ])

    X = np.random.rand(5, 4).astype(np.float32)
    onnx_model = to_onnx(model, (X,))

By default axis 0 of every input is made dynamic.  To name specific
dynamic axes, pass ``dynamic_shapes``:

.. code-block:: python

    onnx_model = to_onnx(model, (X,), dynamic_shapes=({0: "batch"},))

See :ref:`l-design-tensorflow-converter` for the full TensorFlow/JAX conversion guide.

LiteRT / TFLite
---------------

Convert a ``.tflite`` model file with :func:`yobx.litert.to_onnx`:

.. code-block:: python

    import numpy as np
    from yobx.litert import to_onnx

    X = np.random.rand(1, 4).astype(np.float32)
    onnx_model = to_onnx("model.tflite", (X,))

Axis 0 is dynamic by default.  Control which axes are dynamic with
``dynamic_shapes``:

.. code-block:: python

    onnx_model = to_onnx("model.tflite", (X,), dynamic_shapes=({0: "batch"},))

See :ref:`l-design-litert-converter` for details on dynamic shapes and
custom op converters.

Running the exported model
--------------------------

Use :epkg:`onnxruntime` to run the exported model and verify the output:

.. code-block:: python

    import numpy as np
    import onnxruntime as rt

    sess = rt.InferenceSession(onnx_model.SerializeToString())
    input_name = sess.get_inputs()[0].name
    result = sess.run(None, {input_name: X})
    print(result)

Next steps
----------

* :ref:`l-design-graph-builder` — build and optimize ONNX graphs programmatically.
* :ref:`l-design-pattern-optimizer-patterns` — pattern-based graph rewriting.
* :ref:`l-design-shape` — symbolic shape expressions for dynamic shapes.
* :ref:`l-design-translate` — translate ONNX graphs back to Python code.
* :doc:`api/index` — full API reference.
