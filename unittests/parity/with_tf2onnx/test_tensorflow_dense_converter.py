"""
Parity tests for Dense-layer (fully-connected) TensorFlow→ONNX conversion.

Each test converts the *same* Keras model with both **yobx** and **tf2onnx**,
then runs both ONNX models through onnxruntime and asserts that the numerical
outputs are equivalent.  The goal is not to verify conversion correctness in
isolation (that is done in unittests/tensorflow/) but to detect divergences
between the two converters.
"""

import unittest

import numpy as np
import tensorflow as tf
from onnxruntime import InferenceSession

from yobx.ext_test_case import ExtTestCase, requires_tensorflow, requires_tf2onnx
from yobx.tensorflow import to_onnx


def _ort_run(onnx_model, feeds):
    """Run an ONNX model through onnxruntime and return all outputs."""
    sess = InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


def _tf2onnx_from_keras(model, input_arrays, opset=18):
    """Convert a Keras model to ONNX using tf2onnx.

    ``tf2onnx.convert.from_keras`` does not support Keras 3.x (shipped with
    TF 2.16+) because its internal name-lookup fails for ``keras_tensor_*``
    outputs.  We work around this by wrapping the model in a ``@tf.function``
    and using ``tf2onnx.convert.from_function`` instead, which is the approach
    recommended for Keras 3.x.

    Returns the onnx ModelProto.
    """
    import tf2onnx

    # Use None for the batch dimension so the ONNX model accepts any batch size.
    input_sig = [tf.TensorSpec([None, *arr.shape[1:]], dtype=tf.float32) for arr in input_arrays]

    @tf.function(input_signature=input_sig)
    def model_fn(*args):
        return model(*args)

    onnx_proto, _ = tf2onnx.convert.from_function(
        model_fn, input_signature=input_sig, opset=opset
    )
    return onnx_proto


@requires_tensorflow("2.18")
@requires_tf2onnx()
class TestTensorflowDenseConverterParity(ExtTestCase):
    """Compare yobx and tf2onnx outputs for Keras Dense-layer models."""

    def _compare(self, model, X, atol=1e-4):
        """Convert *model* with both converters and assert output parity.

        Parameters
        ----------
        model : tf.keras.Model
            Trained or freshly built Keras model.
        X : np.ndarray
            Input array (float32).
        atol : float
            Absolute tolerance for the output comparison.
        """
        # ---- yobx conversion ----
        onx_yobx = to_onnx(model, (X,))
        yobx_input_name = onx_yobx.graph.input[0].name
        yobx_out = _ort_run(onx_yobx, {yobx_input_name: X})[0]

        # ---- tf2onnx conversion ----
        onx_tf2onnx = _tf2onnx_from_keras(model, [X])
        tf2onnx_input_name = onx_tf2onnx.graph.input[0].name
        tf2onnx_out = _ort_run(onx_tf2onnx, {tf2onnx_input_name: X})[0]

        # ---- ground truth from TF ----
        tf_out = model(X).numpy()

        np.testing.assert_allclose(yobx_out, tf_out, atol=atol, rtol=1e-4)
        np.testing.assert_allclose(tf2onnx_out, tf_out, atol=atol, rtol=1e-4)
        np.testing.assert_allclose(yobx_out, tf2onnx_out, atol=atol, rtol=1e-4)

    def test_dense_linear(self):
        """Single Dense layer with no activation (linear)."""
        np.random.seed(0)
        model = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(3,))])
        X = np.random.rand(5, 3).astype(np.float32)
        self._compare(model, X)

    def test_dense_relu(self):
        """Single Dense layer with relu activation."""
        np.random.seed(1)
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(8, activation="relu", input_shape=(4,))]
        )
        X = np.random.rand(6, 4).astype(np.float32)
        self._compare(model, X)

    def test_dense_sigmoid(self):
        """Single Dense layer with sigmoid activation."""
        np.random.seed(2)
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(2, activation="sigmoid", input_shape=(3,))]
        )
        X = np.random.rand(4, 3).astype(np.float32)
        self._compare(model, X)

    def test_dense_tanh(self):
        """Single Dense layer with tanh activation."""
        np.random.seed(3)
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(5, activation="tanh", input_shape=(4,))]
        )
        X = np.random.rand(8, 4).astype(np.float32)
        self._compare(model, X)

    def test_sequential_multi_layer(self):
        """Multi-layer Sequential model: Dense(relu) → Dense(relu) → Dense."""
        np.random.seed(4)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(8, activation="relu", input_shape=(4,)),
                tf.keras.layers.Dense(4, activation="relu"),
                tf.keras.layers.Dense(2),
            ]
        )
        X = np.random.rand(5, 4).astype(np.float32)
        self._compare(model, X)

    def test_sequential_two_dense_linear(self):
        """Two stacked linear Dense layers without activation."""
        np.random.seed(5)
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(4, input_shape=(3,)), tf.keras.layers.Dense(2)]
        )
        X = np.random.rand(5, 3).astype(np.float32)
        self._compare(model, X)

    def test_dense_softmax(self):
        """Dense layer followed by softmax activation."""
        np.random.seed(6)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(8, activation="relu", input_shape=(5,)),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )
        X = np.random.rand(10, 5).astype(np.float32)
        self._compare(model, X)


if __name__ == "__main__":
    unittest.main()
