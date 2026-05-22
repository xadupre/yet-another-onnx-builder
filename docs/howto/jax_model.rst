.. _l-howto-jax-model:

JAX model
=========

This page answers common *"how do I…"* questions for converting a
:epkg:`jax` model to ONNX with :func:`yobx.tensorflow.to_onnx`.

----

How to convert a JAX model
--------------------------

Write the model as a callable and pass a representative dummy input to
:func:`yobx.tensorflow.to_onnx`:

.. runpython::
    :showcode:

    import jax
    import numpy as np
    from yobx.tensorflow import to_onnx

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    W1 = jax.random.normal(k1, (8, 16), dtype=np.float32)
    b1 = np.zeros(16, dtype=np.float32)
    W2 = jax.random.normal(k2, (16, 4), dtype=np.float32)
    b2 = np.zeros(4, dtype=np.float32)

    def jax_mlp(x):
        h = jax.nn.relu(x @ W1 + b1)
        return h @ W2 + b2

    X = np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32)
    onx = to_onnx(jax_mlp, (X,))
    print(f"nodes={len(onx.graph.node)}")

----

How to verify ONNX outputs against JAX
--------------------------------------

Run the ONNX model with :epkg:`onnxruntime` and compare against JAX outputs:

.. runpython::
    :showcode:

    import jax
    import numpy as np
    import onnxruntime
    from yobx.tensorflow import to_onnx

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    W1 = jax.random.normal(k1, (8, 16), dtype=np.float32)
    b1 = np.zeros(16, dtype=np.float32)
    W2 = jax.random.normal(k2, (16, 4), dtype=np.float32)
    b2 = np.zeros(4, dtype=np.float32)

    def jax_mlp(x):
        h = jax.nn.relu(x @ W1 + b1)
        return h @ W2 + b2

    X = np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32)
    onx = to_onnx(jax_mlp, (X,))
    sess = onnxruntime.InferenceSession(
        onx.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    (got,) = sess.run(None, {input_name: X})
    expected = np.asarray(jax_mlp(X))
    np.testing.assert_allclose(expected, got, atol=1e-2)
    print("Outputs match ✓")

----

How to export with a dynamic batch dimension
--------------------------------------------

Use ``dynamic_shapes`` to name the batch dimension:

.. runpython::
    :showcode:

    import jax
    import numpy as np
    from yobx.tensorflow import to_onnx

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    W1 = jax.random.normal(k1, (8, 16), dtype=np.float32)
    b1 = np.zeros(16, dtype=np.float32)
    W2 = jax.random.normal(k2, (16, 4), dtype=np.float32)
    b2 = np.zeros(4, dtype=np.float32)

    def jax_mlp(x):
        h = jax.nn.relu(x @ W1 + b1)
        return h @ W2 + b2

    X = np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32)
    onx_dyn = to_onnx(jax_mlp, (X,), dynamic_shapes=({0: "batch"},))
    batch_dim = onx_dyn.graph.input[0].type.tensor_type.shape.dim[0]
    assert batch_dim.dim_param
    print("dynamic batch dimension:", batch_dim.dim_param)

----

How to use jax_to_concrete_function explicitly
----------------------------------------------

If needed, convert JAX to a TensorFlow concrete function first:

.. runpython::
    :showcode:

    import jax
    import numpy as np
    from yobx.tensorflow import to_onnx
    from yobx.tensorflow.tensorflow_helper import jax_to_concrete_function

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    W1 = jax.random.normal(k1, (8, 16), dtype=np.float32)
    b1 = np.zeros(16, dtype=np.float32)
    W2 = jax.random.normal(k2, (16, 4), dtype=np.float32)
    b2 = np.zeros(4, dtype=np.float32)

    def jax_mlp(x):
        h = jax.nn.relu(x @ W1 + b1)
        return h @ W2 + b2

    X = np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32)
    cf = jax_to_concrete_function(jax_mlp, (X,), dynamic_shapes=({0: "batch"},))
    onx = to_onnx(cf, (X,), dynamic_shapes=({0: "batch"},))
    print(f"nodes={len(onx.graph.node)}")

.. seealso::

    :ref:`l-plot-jax-to-onnx` — full runnable JAX to ONNX gallery example.
