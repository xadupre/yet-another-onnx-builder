"""
Unit tests for yobx.tensorflow converters.
"""

import unittest
import numpy as np
import tensorflow as tf
from yobx.ext_test_case import ExtTestCase, requires_tensorflow
from yobx.reference import ExtendedReferenceEvaluator
from yobx.tensorflow import to_onnx


@requires_tensorflow("2.0")
class TestTensorflowBaseConverters(ExtTestCase):
    def test_dense_linear(self):
        """Dense layer with no activation (linear) converts to MatMul+Add."""
        model = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(3,))])
        X = np.random.rand(5, 3).astype(np.float32)
        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X:0": X})[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_dense_relu(self):
        """Dense layer with relu activation."""
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(8, activation="relu", input_shape=(4,))]
        )
        X = np.random.rand(6, 4).astype(np.float32)

        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)
        self.assertIn("Relu", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X:0": X})[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_dense_sigmoid(self):
        """Dense layer with sigmoid activation."""
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(2, activation="sigmoid", input_shape=(3,))]
        )
        X = np.random.rand(4, 3).astype(np.float32)

        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sigmoid", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X:0": X})[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_sequential_multi_layer(self):
        """Sequential model with multiple Dense layers."""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(8, activation="relu", input_shape=(4,)),
                tf.keras.layers.Dense(4, activation="relu"),
                tf.keras.layers.Dense(2),
            ]
        )
        X = np.random.rand(5, 4).astype(np.float32)

        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)
        self.assertIn("Relu", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X:0": X})[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_sequential_dynamic_shape(self):
        """Sequential model with an explicit dynamic batch dimension."""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(4, activation="relu", input_shape=(3,)),
                tf.keras.layers.Dense(2),
            ]
        )
        X = np.random.rand(7, 3).astype(np.float32)

        onx = to_onnx(model, (X,), dynamic_shapes=({0: "batch"},))

        input_shape = onx.graph.input[0].type.tensor_type.shape
        # The first dimension should be dynamic (a dim_param, not a fixed dim_value).
        self.assertNotEqual(input_shape.dim[0].dim_value, 7)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X:0": X})[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_plain_tf_function_no_keras(self):
        """A model defined as a plain @tf.function with no Keras layers.

        The function captures ``W`` and ``b`` as tf.Variable closures so that
        the converter can pick them up as graph initializers.  This exercises
        the ``hasattr(model, "get_concrete_function")`` branch in
        :func:`yobx.tensorflow.to_onnx`.
        """
        W = tf.Variable(np.random.rand(3, 4).astype(np.float32))
        b = tf.Variable(np.random.rand(4).astype(np.float32))

        @tf.function
        def model(x):
            return tf.nn.relu(tf.matmul(x, W) + b)

        X = np.random.rand(5, 3).astype(np.float32)
        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)
        self.assertIn("Relu", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X:0": X})[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_tf_module_no_keras(self):
        """A model defined as a tf.Module subclass with no Keras dependency.

        ``tf.Module`` holds trainable variables but does not inherit from any
        Keras class.  Because ``__call__`` is a plain Python method without a
        ``get_concrete_function`` attribute, :func:`yobx.tensorflow.to_onnx`
        wraps it in ``tf.function`` internally (the ``else`` branch).
        """

        class LinearRelu(tf.Module):
            def __init__(self):
                super().__init__()
                self.W = tf.Variable(np.random.rand(3, 4).astype(np.float32))
                self.b = tf.Variable(np.random.rand(4).astype(np.float32))

            def __call__(self, x):
                return tf.nn.relu(tf.matmul(x, self.W) + self.b)

        model = LinearRelu()
        X = np.random.rand(5, 3).astype(np.float32)
        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)
        self.assertIn("Relu", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X:0": X})[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_custom_op_converter_with_extra_converters(self):
        """extra_converters can override how a specific TF op type is converted."""
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(4, activation="relu", input_shape=(3,))]
        )
        X = np.random.rand(5, 3).astype(np.float32)

        called = []

        def custom_relu_converter(g, sts, outputs, op):
            """Override: apply Relu but also track the call."""
            called.append(True)
            return g.op.Relu(op.inputs[0].name, outputs=outputs, name="custom_relu")

        onx = to_onnx(model, (X,), extra_converters={"Relu": custom_relu_converter})

        self.assertTrue(called, "custom Relu converter was not called")
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Relu", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X:0": X})[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
