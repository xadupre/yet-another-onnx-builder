"""
Parity tests for TensorFlow activation-function→ONNX conversion.

Each test converts the *same* ``@tf.function`` wrapped activation with both
**yobx** and **tf2onnx**, runs both resulting ONNX models through onnxruntime,
and asserts numerical equivalence.
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


def _tf2onnx_from_function(tf_fn, input_arrays, opset=18):
    """Convert a ``@tf.function`` to ONNX using tf2onnx.

    Returns the onnx ModelProto.
    """
    import tf2onnx

    input_sig = [tf.TensorSpec(arr.shape, dtype=tf.float32) for arr in input_arrays]
    onnx_proto, _ = tf2onnx.convert.from_function(tf_fn, input_signature=input_sig, opset=opset)
    return onnx_proto


@requires_tensorflow("2.18")
@requires_tf2onnx()
class TestTensorflowActivationsConverterParity(ExtTestCase):
    """Compare yobx and tf2onnx outputs for element-wise activation functions."""

    def _compare_fn(self, tf_fn, X, atol=1e-5):
        """Trace *tf_fn* with both converters and assert output parity.

        Parameters
        ----------
        tf_fn : callable
            A ``@tf.function``-decorated function accepting one tensor.
        X : np.ndarray
            Input array (float32).
        atol : float
            Absolute tolerance for the output comparison.
        """
        # ---- yobx conversion ----
        onx_yobx = to_onnx(tf_fn, (X,))
        yobx_input_name = onx_yobx.graph.input[0].name
        yobx_out = _ort_run(onx_yobx, {yobx_input_name: X})[0]

        # ---- tf2onnx conversion ----
        onx_tf2onnx = _tf2onnx_from_function(tf_fn, [X])
        tf2onnx_input_name = onx_tf2onnx.graph.input[0].name
        tf2onnx_out = _ort_run(onx_tf2onnx, {tf2onnx_input_name: X})[0]

        # ---- ground truth from TF ----
        tf_out = tf_fn(X).numpy()

        np.testing.assert_allclose(yobx_out, tf_out, atol=atol, rtol=1e-4)
        np.testing.assert_allclose(tf2onnx_out, tf_out, atol=atol, rtol=1e-4)
        np.testing.assert_allclose(yobx_out, tf2onnx_out, atol=atol, rtol=1e-4)

    def test_relu(self):
        """tf.nn.relu → ONNX Relu."""
        x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)

        @tf.function
        def fn(t):
            return tf.nn.relu(t)

        self._compare_fn(fn, x)

    def test_elu(self):
        """tf.nn.elu → ONNX Elu."""
        x = np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)

        @tf.function
        def fn(t):
            return tf.nn.elu(t)

        self._compare_fn(fn, x)

    def test_selu(self):
        """tf.nn.selu → ONNX Selu."""
        x = np.array([[-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)

        @tf.function
        def fn(t):
            return tf.nn.selu(t)

        self._compare_fn(fn, x)

    def test_leaky_relu(self):
        """tf.nn.leaky_relu → ONNX LeakyRelu."""
        x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)

        @tf.function
        def fn(t):
            return tf.nn.leaky_relu(t, alpha=0.2)

        self._compare_fn(fn, x)

    def test_sigmoid(self):
        """tf.nn.sigmoid → ONNX Sigmoid."""
        x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)

        @tf.function
        def fn(t):
            return tf.nn.sigmoid(t)

        self._compare_fn(fn, x)

    def test_tanh(self):
        """tf.nn.tanh → ONNX Tanh."""
        x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)

        @tf.function
        def fn(t):
            return tf.math.tanh(t)

        self._compare_fn(fn, x)

    def test_softplus(self):
        """tf.nn.softplus → ONNX Softplus."""
        x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)

        @tf.function
        def fn(t):
            return tf.nn.softplus(t)

        self._compare_fn(fn, x)

    def test_exp(self):
        """tf.math.exp → ONNX Exp."""
        x = np.array([[-1.0, 0.0, 0.5, 1.0, 2.0]], dtype=np.float32)

        @tf.function
        def fn(t):
            return tf.math.exp(t)

        self._compare_fn(fn, x)

    def test_log(self):
        """tf.math.log → ONNX Log."""
        x = np.array([[0.1, 0.5, 1.0, 2.0, 10.0]], dtype=np.float32)

        @tf.function
        def fn(t):
            return tf.math.log(t)

        self._compare_fn(fn, x)

    def test_abs(self):
        """tf.math.abs → ONNX Abs."""
        x = np.array([[-3.0, -1.0, 0.0, 1.0, 3.0]], dtype=np.float32)

        @tf.function
        def fn(t):
            return tf.math.abs(t)

        self._compare_fn(fn, x)

    def test_sqrt(self):
        """tf.math.sqrt → ONNX Sqrt."""
        x = np.array([[0.0, 1.0, 4.0, 9.0, 16.0]], dtype=np.float32)

        @tf.function
        def fn(t):
            return tf.math.sqrt(t)

        self._compare_fn(fn, x)

    def test_floor(self):
        """tf.math.floor → ONNX Floor."""
        x = np.array([[-1.9, -0.5, 0.0, 0.7, 1.5]], dtype=np.float32)

        @tf.function
        def fn(t):
            return tf.math.floor(t)

        self._compare_fn(fn, x)

    def test_ceil(self):
        """tf.math.ceil → ONNX Ceil."""
        x = np.array([[-1.9, -0.5, 0.0, 0.7, 1.5]], dtype=np.float32)

        @tf.function
        def fn(t):
            return tf.math.ceil(t)

        self._compare_fn(fn, x)

    def test_square(self):
        """tf.math.square → ONNX Mul (x * x)."""
        x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)

        @tf.function
        def fn(t):
            return tf.math.square(t)

        self._compare_fn(fn, x)

    def test_reciprocal(self):
        """tf.math.reciprocal → ONNX Reciprocal."""
        x = np.array([[0.5, 1.0, 2.0, 4.0, 8.0]], dtype=np.float32)

        @tf.function
        def fn(t):
            return tf.math.reciprocal(t)

        self._compare_fn(fn, x)


if __name__ == "__main__":
    unittest.main()
