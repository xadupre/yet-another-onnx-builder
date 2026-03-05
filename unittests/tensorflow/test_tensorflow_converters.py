"""
Unit tests for yobx.tensorflow converters.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_tensorflow
from yobx.reference import ExtendedReferenceEvaluator


@requires_tensorflow("2.0")
class TestTensorflowBaseConverters(ExtTestCase):
    def test_dense_linear(self):
        """Dense layer with no activation (linear) converts to MatMul+Add."""
        import tensorflow as tf
        from yobx.tensorflow import to_onnx

        model = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(3,))])
        X = np.random.rand(5, 3).astype(np.float32)

        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)
        self.assertIn("Add", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_dense_relu(self):
        """Dense layer with relu activation."""
        import tensorflow as tf
        from yobx.tensorflow import to_onnx

        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(8, activation="relu", input_shape=(4,))]
        )
        X = np.random.rand(6, 4).astype(np.float32)

        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)
        self.assertIn("Relu", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_dense_sigmoid(self):
        """Dense layer with sigmoid activation."""
        import tensorflow as tf
        from yobx.tensorflow import to_onnx

        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(2, activation="sigmoid", input_shape=(3,))]
        )
        X = np.random.rand(4, 3).astype(np.float32)

        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sigmoid", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_sequential_multi_layer(self):
        """Sequential model with multiple Dense layers."""
        import tensorflow as tf
        from yobx.tensorflow import to_onnx

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
        result = ref.run(None, {"X": X})[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_sequential_dynamic_shape(self):
        """Sequential model with an explicit dynamic batch dimension."""
        import tensorflow as tf
        from yobx.tensorflow import to_onnx

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
        result = ref.run(None, {"X": X})[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_custom_layer_with_extra_converters(self):
        """extra_converters allows converting unsupported custom layers at the top level."""
        import tensorflow as tf
        from yobx.tensorflow import to_onnx

        class ScaleLayer(tf.keras.layers.Layer):
            """Custom layer that multiplies inputs by a scalar."""

            def __init__(self, scale=2.0, **kwargs):
                super().__init__(**kwargs)
                self.scale = scale

            def call(self, inputs):
                return inputs * self.scale

        def convert_scale_layer(g, sts, outputs, layer, X, name="scale"):
            scale = np.array([layer.scale], dtype=np.float32)
            res = g.op.Mul(X, scale, name=name, outputs=outputs)
            if not sts:
                g.set_type(res, g.get_type(X))
                g.set_shape(res, g.get_shape(X))
                if g.has_device(X):
                    g.set_device(res, g.get_device(X))
            return res

        # Call the layer once to build it (establishes its input/output shapes).
        layer = ScaleLayer(scale=3.0)
        X = np.array([[1, 2, 3, 4]], dtype=np.float32)
        _ = layer(X)

        onx = to_onnx(layer, (X,), extra_converters={ScaleLayer: convert_scale_layer})

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Mul", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = layer(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
