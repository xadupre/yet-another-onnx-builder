"""
Tests adapted from
https://github.com/onnx/tensorflow-onnx/blob/main/tests/keras2onnx_unit_tests/test_subclassing.py

Covers Keras subclassed models (TF 2.x only):

* LeNet CNN
* MLP (fully-connected)
* tf.math ops (squared_difference, matmul, cast, expand_dims)
* Variational AutoEncoder with random-normal sampling
* tf.where / SelectV2
"""

import unittest
import numpy as np
from onnxruntime import InferenceSession
import tensorflow as tf
from yobx.ext_test_case import ExtTestCase, requires_tensorflow
from yobx.tensorflow import to_onnx


def _ort_run(onx, feeds):
    """Run an ONNX model with onnxruntime and return all outputs."""
    sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


# ---------------------------------------------------------------------------
# Model definitions (mirror of test_subclassing.py)
# ---------------------------------------------------------------------------

layers = tf.keras.layers


class LeNet(tf.keras.Model):
    """LeNet-style CNN adapted from the tensorflow-onnx test suite."""

    def __init__(self):
        super().__init__()
        self.conv2d_1 = layers.Conv2D(filters=6, kernel_size=(3, 3), activation="relu")
        self.average_pool = layers.AveragePooling2D(pool_size=(2, 2))
        self.conv2d_2 = layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu")
        self.flatten = layers.Flatten()
        self.fc_1 = layers.Dense(120, activation="relu")
        self.fc_2 = layers.Dense(84, activation="relu")
        self.out = layers.Dense(10, activation="softmax")

    def call(self, inputs, **kwargs):
        x = self.conv2d_1(inputs)
        x = self.average_pool(x)
        x = self.conv2d_2(x)
        x = self.average_pool(x)
        x = self.flatten(x)
        x = self.fc_2(self.fc_1(x))
        return self.out(x)


class MLP(tf.keras.Model):
    """Simple fully-connected model."""

    def __init__(self):
        super().__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(units=256, activation=tf.nn.relu)
        self.dense2 = layers.Dense(units=10)

    def call(self, inputs, **kwargs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)


class SimpleWrapperModel(tf.keras.Model):
    """Wraps an arbitrary ``tf.function``-compatible callable."""

    def __init__(self, func):
        super().__init__()
        self.func = func

    def call(self, inputs, **kwargs):
        return self.func(inputs)


class Sampling(layers.Layer):
    """Reparameterisation trick: samples z ~ N(z_mean, exp(z_log_var))."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), seed=12340)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps inputs to (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Decodes latent vector back to data space."""

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_output = layers.Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(tf.keras.Model):
    """Combines Encoder and Decoder into an end-to-end VAE."""

    def __init__(
        self,
        original_dim,
        intermediate_dim=64,
        latent_dim=32,
        name="autoencoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_tensorflow("2.18")
class TestSubclassingModels(ExtTestCase):
    """Tests adapted from tensorflow-onnx test_subclassing.py."""

    # ------------------------------------------------------------------
    # LeNet
    # ------------------------------------------------------------------

    def test_lenet(self):
        """LeNet-style CNN with Conv2D, AveragePooling, Dense → converts and runs."""
        tf.keras.backend.clear_session()
        lenet = LeNet()
        data = np.random.rand(2, 32, 32, 1).astype(np.float32)
        _ = lenet(data)  # build the model

        onx = to_onnx(lenet, (data,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Conv", op_types)
        self.assertIn("AveragePool", op_types)
        self.assertIn("Softmax", op_types)

        expected = lenet(data).numpy()
        input_name = onx.graph.input[0].name
        (result,) = _ort_run(onx, {input_name: data})
        self.assertEqualArray(expected, result, atol=1e-4)

    # ------------------------------------------------------------------
    # MLP
    # ------------------------------------------------------------------

    def test_mlp(self):
        """Multilayer perceptron (Flatten + Dense + Relu + Dense) → converts and runs."""
        tf.keras.backend.clear_session()
        mlp = MLP()
        np_input = np.random.normal(size=(2, 20)).astype(np.float32)
        _ = mlp(np_input)  # build the model

        onx = to_onnx(mlp, (np_input,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)
        self.assertIn("Relu", op_types)

        expected = mlp(np_input).numpy()
        input_name = onx.graph.input[0].name
        (result,) = _ort_run(onx, {input_name: np_input})
        self.assertEqualArray(expected, result, atol=1e-4)

    # ------------------------------------------------------------------
    # TF math ops
    # ------------------------------------------------------------------

    def test_tf_ops(self):
        """squared_difference + batched matmul (adjoint_b) + cast + expand_dims."""
        tf.keras.backend.clear_session()

        @tf.function
        def op_func(x, y):
            diff = tf.math.squared_difference(x, y)
            result = tf.matmul(diff, diff, adjoint_b=True)
            r = tf.rank(result)
            result = result - tf.cast(tf.expand_dims(r, axis=0), tf.float32)
            return result

        x = np.random.normal(size=(3, 2, 20)).astype(np.float32)
        y = np.random.normal(size=(3, 2, 20)).astype(np.float32)

        onx = to_onnx(op_func, (x, y), input_names=["X", "Y"])

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)

        inp_names = [i.name for i in onx.graph.input]
        expected = op_func(x, y).numpy()
        (result,) = _ort_run(onx, {inp_names[0]: x, inp_names[1]: y})
        self.assertEqualArray(expected, result, atol=1e-4)

    # ------------------------------------------------------------------
    # Variational AutoEncoder (random ops)
    # ------------------------------------------------------------------

    def test_variational_auto_encoder(self):
        """VAE with Encoder/Decoder and random-normal sampling converts without error.

        The ONNX random-number generator differs from TensorFlow's, so only
        the output *shape* and the success of OnnxRuntime inference are checked;
        numerical values are not compared.
        """
        tf.keras.backend.clear_session()
        original_dim = 20
        vae = VariationalAutoEncoder(original_dim, 64, 32)
        x = np.random.normal(size=(7, original_dim)).astype(np.float32)
        _ = vae(x)  # build the model

        onx = to_onnx(vae, (x,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("RandomNormalLike", op_types)

        # Verify the converted model runs in OnnxRuntime.
        input_name = onx.graph.input[0].name
        (result,) = _ort_run(onx, {input_name: x})
        self.assertEqual(result.shape, (7, original_dim))

    # ------------------------------------------------------------------
    # tf.where / SelectV2
    # ------------------------------------------------------------------

    def test_tf_where(self):
        """tf.where with scalar and vector conditions → ONNX Where."""
        tf.keras.backend.clear_session()

        @tf.function
        def tf_where_fn(input_0):
            a = tf.where(True, input_0, tf.constant([0, 1, 2, 5, 7], dtype=tf.int32))
            b = tf.where(
                tf.constant([True]),
                tf.expand_dims(input_0, axis=0),
                tf.expand_dims(tf.constant([0, 1, 2, 5, 7], dtype=tf.int32), axis=0),
            )
            c = tf.logical_or(tf.cast(a, tf.bool), tf.cast(b, tf.bool))
            return c

        const_in = np.array([2, 4, 6, 8, 10]).astype(np.int32)

        onx = to_onnx(tf_where_fn, (const_in,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Where", op_types)

        expected = tf_where_fn(const_in).numpy()
        input_name = onx.graph.input[0].name
        (result,) = _ort_run(onx, {input_name: const_in})
        self.assertEqualArray(expected, result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
