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

    def test_custom_op_converter_with_extra_converters(self):
        """extra_converters can override how a specific TF op type is converted."""
        import tensorflow as tf
        from yobx.tensorflow import to_onnx

        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(4, activation="relu", input_shape=(3,))]
        )
        X = np.random.rand(5, 3).astype(np.float32)

        called = []

        def custom_relu_converter(g, op, ctx, verbose=0):
            """Override: apply Relu but also track the call."""
            called.append(True)
            a = ctx.get(op.inputs[0].name)
            if a is not None:
                result = g.op.Relu(a, name="custom_relu")
                assert isinstance(result, str)
                ctx[op.outputs[0].name] = result

        onx = to_onnx(model, (X,), extra_converters={"Relu": custom_relu_converter})

        self.assertTrue(called, "custom Relu converter was not called")
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Relu", op_types)

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X": X})[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
