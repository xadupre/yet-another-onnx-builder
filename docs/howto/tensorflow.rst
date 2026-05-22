.. _l-howto-tensorflow:

TensorFlow / Keras
==================

This page answers common *"how do I…"* questions for converting
:epkg:`TensorFlow`/:epkg:`Keras` models to ONNX with
:func:`yobx.tensorflow.to_onnx`.

----

How to convert a TensorFlow model
---------------------------------

Build a Keras model, run one forward pass to initialize weights, then call
:func:`yobx.tensorflow.to_onnx` with a representative input array:

.. runpython::
    :showcode:

    import numpy as np
    import tensorflow as tf
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.tensorflow import to_onnx

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(8, activation="relu", input_shape=(4,)),
            tf.keras.layers.Dense(3),
        ]
    )

    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 4)).astype(np.float32)
    _ = model(X)  # initialize model weights

    onx = to_onnx(model, (X,))
    print(pretty_onnx(onx))

----

How to validate ONNX outputs with onnxruntime
---------------------------------------------

Run the exported model with :epkg:`onnxruntime` and compare outputs with the
original Keras model:

.. runpython::
    :showcode:

    import numpy as np
    import onnxruntime
    import tensorflow as tf
    from yobx.tensorflow import to_onnx

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(16, activation="relu", input_shape=(6,)),
            tf.keras.layers.Dense(2),
        ]
    )

    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((20, 6)).astype(np.float32)
    _ = model(X_train)  # initialize model weights
    onx = to_onnx(model, (X_train[:1],))

    X_test = rng.standard_normal((7, 6)).astype(np.float32)
    expected = model(X_test).numpy()

    sess = onnxruntime.InferenceSession(
        onx.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    (got,) = sess.run(None, {"X:0": X_test})

    assert np.allclose(expected, got, atol=1e-5)
    print("Outputs match ✓")

----

How to control dynamic shapes
-----------------------------

By default, the first input axis is dynamic. Pass ``dynamic_shapes`` to name it
explicitly:

.. runpython::
    :showcode:

    import numpy as np
    import tensorflow as tf
    from yobx.tensorflow import to_onnx

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(4, activation="relu", input_shape=(3,)),
            tf.keras.layers.Dense(1),
        ]
    )

    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, 3)).astype(np.float32)
    _ = model(X)  # initialize model weights

    onx = to_onnx(model, (X,), dynamic_shapes=({0: "batch"},))
    dim0 = onx.graph.input[0].type.tensor_type.shape.dim[0]
    print(f"dynamic dim name: {dim0.dim_param!r}")

.. seealso::

    :ref:`l-plot-tensorflow-to-onnx` — full gallery example.

    :ref:`l-design-tensorflow-converter` — converter design and API details.
