"""Tests direct JAX-to-ONNX conversion without TensorFlow."""

import unittest

import numpy as np

from yobx.ext_test_case import ExtTestCase, has_jax


@unittest.skipUnless(has_jax(), "jax not installed")
class TestJaxToOnnx(ExtTestCase):
    """Checks direct JAX export against JAX's own results."""

    def _check(self, fn, args, dynamic_shapes=None, input_names=None, atol=1e-5):
        from onnx_light.onnx import ModelProto
        from onnxruntime import InferenceSession
        from yobx.jax import to_onnx

        artifact = to_onnx(fn, args, dynamic_shapes=dynamic_shapes, input_names=input_names)
        self.assertIsInstance(artifact.proto, ModelProto)
        session = InferenceSession(
            artifact.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        names = [value.name for value in session.get_inputs()]
        if input_names is not None:
            self.assertEqual(names, list(input_names))
        expected = fn(*args)
        if not isinstance(expected, (tuple, list)):
            expected = (expected,)
        for result, value in zip(session.run(None, dict(zip(names, args))), expected):
            self.assertEqualArray(np.asarray(value), result, atol=atol)
        return artifact, session

    def test_elementwise_dynamic_batch(self):
        import jax.numpy as jnp

        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 3)).astype(np.float32)
        model, session = self._check(jnp.sin, (x,), dynamic_shapes=({0: "batch"},))
        dim = model.graph.input[0].type.tensor_type.shape.dim[0]
        self.assertEqual(dim.dim_param, "batch")
        name = session.get_inputs()[0].name
        for batch in (2, 7):
            value = rng.standard_normal((batch, 3)).astype(np.float32)
            (result,) = session.run(None, {name: value})
            self.assertEqualArray(np.sin(value), result, atol=1e-5)

    def test_default_dynamic_batch(self):
        import jax.numpy as jnp

        x = np.ones((4, 3), dtype=np.float32)
        model, _ = self._check(jnp.exp, (x,))
        self.assertTrue(model.graph.input[0].type.tensor_type.shape.dim[0].dim_param)

    def test_tensorflow_entry_point_routes_jax_directly(self):
        import jax.numpy as jnp
        from onnxruntime import InferenceSession
        from yobx.tensorflow import to_onnx

        x = np.ones((4, 3), dtype=np.float32)
        artifact = to_onnx(jnp.sin, (x,))
        session = InferenceSession(
            artifact.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (result,) = session.run(None, {session.get_inputs()[0].name: x})
        self.assertEqualArray(np.asarray(jnp.sin(x)), result, atol=1e-5)

    def test_multiple_inputs_and_custom_names(self):
        import jax.numpy as jnp

        x = np.arange(12, dtype=np.float32).reshape((4, 3))
        y = np.ones((4, 3), dtype=np.float32)
        self._check(
            jnp.add,
            (x, y),
            dynamic_shapes=({0: "batch"}, {0: "batch"}),
            input_names=["left", "right"],
        )

    def test_matrix_multiplication(self):
        import jax.numpy as jnp

        x = np.arange(12, dtype=np.float32).reshape((4, 3))
        weights = np.ones((3, 2), dtype=np.float32)
        self._check(jnp.matmul, (x, weights), dynamic_shapes=({0: "batch"}, {}))

    def test_small_neural_network(self):
        import jax

        weights = np.arange(12, dtype=np.float32).reshape((3, 4)) / 10
        bias = np.ones((4,), dtype=np.float32)
        output = np.arange(8, dtype=np.float32).reshape((4, 2)) / 10

        def network(x):
            return jax.nn.relu(x @ weights + bias) @ output

        x = np.ones((5, 3), dtype=np.float32)
        model, session = self._check(network, (x,), dynamic_shapes=({0: "batch"},))
        self.assertIn("MatMul", [node.op_type for node in model.graph.node])
        larger = np.ones((7, 3), dtype=np.float32)
        (result,) = session.run(None, {session.get_inputs()[0].name: larger})
        self.assertEqualArray(np.asarray(network(larger)), result, atol=1e-5)

    def test_softmax(self):
        import jax

        x = np.arange(20, dtype=np.float32).reshape((4, 5))
        self._check(jax.nn.softmax, (x,), dynamic_shapes=({0: "batch"},))

    def test_dynamic_broadcast(self):
        import jax
        import jax.numpy as jnp

        def broadcast(x):
            return jax.lax.broadcast_in_dim(
                jnp.sum(x, axis=0), x.shape, broadcast_dimensions=(1,)
            )

        x = np.arange(12, dtype=np.float32).reshape((4, 3))
        _, session = self._check(broadcast, (x,), dynamic_shapes=({0: "batch"},))
        larger = np.ones((7, 3), dtype=np.float32)
        (result,) = session.run(None, {session.get_inputs()[0].name: larger})
        self.assertEqualArray(np.asarray(broadcast(larger)), result, atol=1e-5)

    def test_input_names_length_mismatch(self):
        import jax.numpy as jnp
        from yobx.jax import to_onnx

        x = np.ones((3, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            to_onnx(jnp.sin, (x,), input_names=["a", "b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
