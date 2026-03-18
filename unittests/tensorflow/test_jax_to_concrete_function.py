"""
Unit tests for yobx.tensorflow.tensorflow_helper.jax_to_concrete_function.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_jax
from yobx.tensorflow.tensorflow_helper import jax_to_concrete_function


@requires_jax()
class TestJaxToConcreteFunction(ExtTestCase):
    def test_simple_elementwise(self):
        """A JAX sin function converts to a ConcreteFunction and runs."""
        import jax.numpy as jnp
        import tensorflow as tf

        def jax_fn(x):
            return jnp.sin(x)

        x = np.random.rand(4, 3).astype(np.float32)
        cf = jax_to_concrete_function(jax_fn, (x,), dynamic_shapes=({0: "batch"},))
        self.assertIsInstance(cf, tf.types.experimental.ConcreteFunction)

        result = cf(x).numpy()
        expected = np.sin(x)
        self.assertEqualArray(expected, result, atol=1e-6)

    @unittest.skip("not implemented yet")
    def test_dynamic_batch_default(self):
        """Without explicit dynamic_shapes, batch dim (axis 0) is made dynamic."""
        import jax.nn as jnn
        import tensorflow as tf

        def jax_fn(x):
            return jnn.relu(x)

        x = np.random.rand(5, 4).astype(np.float32)
        cf = jax_to_concrete_function(jax_fn, (x,))
        self.assertIsInstance(cf, tf.types.experimental.ConcreteFunction)

        # The spec should have None for axis 0.
        spec = cf.structured_input_signature[0][0]
        self.assertIsNone(spec.shape[0])
        self.assertEqual(spec.shape[1], 4)

    def test_concrete_function_accepts_different_batch_sizes(self):
        """ConcreteFunction created with dynamic batch runs on different batch sizes."""
        import jax.numpy as jnp

        def jax_fn(x):
            return jnp.sin(x)

        x = np.random.rand(4, 3).astype(np.float32)
        cf = jax_to_concrete_function(jax_fn, (x,), dynamic_shapes=({0: "batch"},))

        for batch in (2, 7, 10):
            xi = np.random.rand(batch, 3).astype(np.float32)
            result = cf(xi).numpy()
            expected = np.sin(xi)
            self.assertEqualArray(expected, result, atol=1e-6)

    @unittest.skip("not implemented yet")
    def test_export_to_onnx_dynamic_shapes(self):
        """to_onnx() accepts a ConcreteFunction from jax_to_concrete_function.

        With native_serialization=False (the default), jax2tf lowers the JAX
        computation to standard TensorFlow ops (e.g. Sin, MatMul) which have
        registered ONNX converters, so the full JAX→ONNX round-trip works.
        """
        import jax.numpy as jnp
        from yobx.tensorflow import to_onnx

        def jax_fn(x):
            return jnp.sin(x)

        x = np.random.rand(4, 3).astype(np.float32)
        cf = jax_to_concrete_function(jax_fn, (x,), dynamic_shapes=({0: "batch"},))

        onx = to_onnx(cf, (x,), dynamic_shapes=({0: "batch"},))
        # Verify the ONNX graph has a dynamic batch dim.
        inp = onx.graph.input[0]
        batch_dim = inp.type.tensor_type.shape.dim[0]
        self.assertNotEqual(batch_dim.dim_value, 4)  # not a fixed static value

    def test_multiple_inputs(self):
        """JAX fn with two inputs produces a ConcreteFunction with two specs."""
        import jax.numpy as jnp
        import tensorflow as tf

        def jax_fn(x, y):
            return jnp.add(x, y)

        x = np.random.rand(3, 4).astype(np.float32)
        y = np.random.rand(3, 4).astype(np.float32)
        cf = jax_to_concrete_function(jax_fn, (x, y), dynamic_shapes=({0: "batch"}, {0: "batch"}))
        self.assertIsInstance(cf, tf.types.experimental.ConcreteFunction)

        result = cf(x, y).numpy()
        expected = x + y
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_custom_input_names(self):
        """Custom input_names are reflected in the ConcreteFunction specs."""
        import jax.numpy as jnp

        def jax_fn(a):
            return jnp.exp(a)

        a = np.random.rand(2, 5).astype(np.float32)
        cf = jax_to_concrete_function(jax_fn, (a,), input_names=["my_input"])

        spec = cf.structured_input_signature[0][0]
        self.assertEqual(spec.name, "my_input")

    @unittest.skip("not implemented yet")
    def test_to_onnx_auto_detects_jax_function(self):
        """to_onnx() automatically calls jax_to_concrete_function for JAX callables.

        When a plain JAX function is passed as ``model``, ``to_onnx()`` should
        detect the JAX-specific TypeError and fall back to
        ``jax_to_concrete_function`` (with native_serialization=False) automatically,
        producing standard TF ops that can be converted to ONNX.
        """
        import jax.numpy as jnp
        from yobx.tensorflow import to_onnx

        def jax_fn(x):
            return jnp.sin(x)

        x = np.random.rand(4, 3).astype(np.float32)

        # Pass the raw JAX function – to_onnx() should auto-detect it and succeed.
        onx = to_onnx(jax_fn, (x,), dynamic_shapes=({0: "batch"},))
        inp = onx.graph.input[0]
        # A dynamic dimension must have dim_param set (symbolic name like "batch"),
        # not a fixed dim_value; verify the batch axis is not frozen to 4.
        batch_dim = inp.type.tensor_type.shape.dim[0]
        self.assertTrue(
            batch_dim.dim_param != "" or batch_dim.dim_value != 4,
            f"Expected dynamic batch dim; got dim_value={batch_dim.dim_value}, "
            f"dim_param={batch_dim.dim_param!r}",
        )

    def test_input_names_length_mismatch_raises(self):
        """Mismatched input_names length raises ValueError."""
        import jax.numpy as jnp

        def jax_fn(x):
            return jnp.sin(x)

        x = np.random.rand(3, 3).astype(np.float32)
        with self.assertRaises(ValueError):
            jax_to_concrete_function(jax_fn, (x,), input_names=["a", "b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
