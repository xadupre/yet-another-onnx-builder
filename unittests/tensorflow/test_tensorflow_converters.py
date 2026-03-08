"""
Unit tests for yobx.tensorflow converters.
"""

import unittest
import numpy as np
from onnxruntime import InferenceSession
import tensorflow as tf
from yobx.ext_test_case import ExtTestCase, requires_tensorflow
from yobx.reference import ExtendedReferenceEvaluator
from yobx.tensorflow import to_onnx


def _ort_run(onx, feeds):
    """Run an ONNX model with onnxruntime; returns the first output."""
    sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)[0]


@requires_tensorflow("2.18")
class TestTensorflowBaseConverters(ExtTestCase):
    def test_dense_linear(self):
        """Dense layer with no activation (linear) converts to MatMul+Add."""
        model = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(3,))])
        X = np.random.rand(5, 3).astype(np.float32)
        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, feeds)[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-5)

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

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, feeds)[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-3)

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-3)

    def test_dense_sigmoid(self):
        """Dense layer with sigmoid activation."""
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(2, activation="sigmoid", input_shape=(3,))]
        )
        X = np.random.rand(4, 3).astype(np.float32)

        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sigmoid", op_types)

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, feeds)[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-5)

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
        expected = model(X).numpy()

        onx = to_onnx(model, (X,))

        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)
        self.assertIn("Relu", op_types)

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, result, atol=1e-3)

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-3)

    def test_sequential_dynamic_shape(self):
        """Sequential model with an explicit dynamic batch dimension."""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(4, activation="relu", input_shape=(3,)),
                tf.keras.layers.Dense(2),
            ]
        )
        X = np.random.rand(7, 3).astype(np.float32)
        expected = model(X).numpy()

        onx = to_onnx(model, (X,), dynamic_shapes=({0: "batch"},))

        input_shape = onx.graph.input[0].type.tensor_type.shape
        # The first dimension should be dynamic (a dim_param, not a fixed dim_value).
        self.assertNotEqual(input_shape.dim[0].dim_value, 7)

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, result, atol=1e-5)

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_sequential_initializer_confusion_regression(self):
        """Regression: multi-layer Sequential models must produce numerically correct outputs.

        Verifies that conversion does not raise and that the ONNX model matches
        TF predictions, covering the previously broken code path where variable
        handles were paired with the wrong tf.Variable values.
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(4, input_shape=(3,)),
                tf.keras.layers.Dense(2),
            ]
        )
        X = np.random.rand(5, 3).astype(np.float32)
        expected = model(X).numpy()

        onx = to_onnx(model, (X,))

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, feeds)[0]
        self.assertEqualArray(expected, result, atol=1e-5)

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-5)

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
        self.assertIn("Gemm", op_types)
        self.assertIn("Relu", op_types)

        expected = model(X).numpy()

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X:0": X})[0]
        self.assertEqualArray(expected, result, atol=1e-5)

        ort_session = self.check_ort(onx)
        ort_result = ort_session.run(None, {"X:0": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

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
        self.assertIn("Gemm", op_types)
        self.assertIn("Relu", op_types)

        expected = model(X).numpy()

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X:0": X})[0]
        self.assertEqualArray(expected, result, atol=1e-5)

        ort_session = self.check_ort(onx)
        ort_result = ort_session.run(None, {"X:0": X})[0]
        self.assertEqualArray(expected, ort_result, atol=1e-5)

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

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, feeds)[0]
        expected = model(X).numpy()
        self.assertEqualArray(expected, result, atol=1e-5)

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-5)


@requires_tensorflow("2.18")
class TestTensorflowBinaryOpConverters(ExtTestCase):
    """Tests for the binary-op converters added in binary_ops.py."""

    def _run_binary_op(self, tf_fn, a, b, disable_ort=False):
        """Trace tf_fn(a, b) to ONNX and compare TF vs ONNX vs ORT results."""

        @tf.function
        def model(x, y):
            return tf_fn(x, y)

        onx = to_onnx(model, (a, b), input_names=["X", "Y"])
        expected = model(a, b).numpy()

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X:0": a, "Y:0": b})[0]
        self.assertEqualArray(expected, result, atol=1e-5)

        if disable_ort:
            # Some operator is not implemented in onnxruntime.
            return onx
        ort_result = _ort_run(onx, {"X:0": a, "Y:0": b})
        self.assertEqualArray(expected, ort_result, atol=1e-5)
        return onx

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def test_sub(self):
        """TF Sub → ONNX Sub."""
        a = np.random.rand(3, 4).astype(np.float32)
        b = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_binary_op(tf.subtract, a, b)
        self.assertIn("Sub", [n.op_type for n in onx.graph.node])

    def test_mul(self):
        """TF Mul → ONNX Mul."""
        a = np.random.rand(3, 4).astype(np.float32)
        b = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_binary_op(tf.multiply, a, b)
        self.assertIn("Mul", [n.op_type for n in onx.graph.node])

    def test_real_div(self):
        """TF RealDiv → ONNX Div."""
        a = np.random.rand(3, 4).astype(np.float32) + 0.5
        b = np.random.rand(3, 4).astype(np.float32) + 0.5
        onx = self._run_binary_op(tf.divide, a, b)
        self.assertIn("Div", [n.op_type for n in onx.graph.node])

    def test_minimum(self):
        """TF Minimum → ONNX Min."""
        a = np.random.rand(3, 4).astype(np.float32)
        b = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_binary_op(tf.minimum, a, b)
        self.assertIn("Min", [n.op_type for n in onx.graph.node])

    def test_maximum(self):
        """TF Maximum → ONNX Max."""
        a = np.random.rand(3, 4).astype(np.float32)
        b = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_binary_op(tf.maximum, a, b)
        self.assertIn("Max", [n.op_type for n in onx.graph.node])

    def test_pow(self):
        """TF Pow → ONNX Pow."""
        a = np.random.rand(3, 4).astype(np.float32) + 0.1
        b = np.random.rand(3, 4).astype(np.float32) + 0.1
        onx = self._run_binary_op(tf.pow, a, b)
        self.assertIn("Pow", [n.op_type for n in onx.graph.node])

    def test_squared_difference(self):
        """TF SquaredDifference → ONNX Sub + Mul."""
        a = np.random.rand(3, 4).astype(np.float32)
        b = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_binary_op(tf.math.squared_difference, a, b)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sub", op_types)
        self.assertIn("Mul", op_types)

    def test_floor_mod(self):
        """TF FloorMod → ONNX Mod(fmod=0)."""
        a = (np.random.rand(3, 4).astype(np.float32) * 10 + 1).astype(np.float32)
        b = (np.random.rand(3, 4).astype(np.float32) * 3 + 1).astype(np.float32)
        onx = self._run_binary_op(tf.math.floormod, a, b, disable_ort=True)
        self.assertIn("Mod", [n.op_type for n in onx.graph.node])

    def test_truncate_mod(self):
        """TF TruncateMod → ONNX Mod(fmod=1)."""
        a = (np.random.rand(3, 4).astype(np.float32) * 10 + 1).astype(np.float32)
        b = (np.random.rand(3, 4).astype(np.float32) * 3 + 1).astype(np.float32)
        onx = self._run_binary_op(tf.truncatemod, a, b)
        self.assertIn("Mod", [n.op_type for n in onx.graph.node])

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def test_equal(self):
        """TF Equal → ONNX Equal."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        b = np.array([[1.0, 0.0, 3.0], [4.0, 9.0, 6.0]], dtype=np.float32)
        onx = self._run_binary_op(tf.equal, a, b)
        self.assertIn("Equal", [n.op_type for n in onx.graph.node])

    def test_not_equal(self):
        """TF NotEqual → ONNX Not(Equal)."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        b = np.array([[1.0, 0.0, 3.0], [4.0, 9.0, 6.0]], dtype=np.float32)
        onx = self._run_binary_op(tf.not_equal, a, b)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Equal", op_types)
        self.assertIn("Not", op_types)

    def test_greater(self):
        """TF Greater → ONNX Greater."""
        a = np.random.rand(3, 4).astype(np.float32)
        b = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_binary_op(tf.greater, a, b)
        self.assertIn("Greater", [n.op_type for n in onx.graph.node])

    def test_greater_equal(self):
        """TF GreaterEqual → ONNX GreaterOrEqual."""
        a = np.random.rand(3, 4).astype(np.float32)
        b = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_binary_op(tf.greater_equal, a, b)
        self.assertIn("GreaterOrEqual", [n.op_type for n in onx.graph.node])

    def test_less(self):
        """TF Less → ONNX Less."""
        a = np.random.rand(3, 4).astype(np.float32)
        b = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_binary_op(tf.less, a, b)
        self.assertIn("Less", [n.op_type for n in onx.graph.node])

    def test_less_equal(self):
        """TF LessEqual → ONNX LessOrEqual."""
        a = np.random.rand(3, 4).astype(np.float32)
        b = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_binary_op(tf.less_equal, a, b)
        self.assertIn("LessOrEqual", [n.op_type for n in onx.graph.node])

    # ------------------------------------------------------------------
    # Logical
    # ------------------------------------------------------------------

    def test_logical_and(self):
        """TF LogicalAnd → ONNX And."""
        a = np.array([[True, False, True], [False, True, False]])
        b = np.array([[True, True, False], [False, False, True]])
        onx = self._run_binary_op(tf.logical_and, a, b)
        self.assertIn("And", [n.op_type for n in onx.graph.node])

    def test_logical_not(self):
        """TF LogicalNot → ONNX Not."""
        a = np.array([[True, False, True], [False, True, False]])

        @tf.function
        def model(x):
            return tf.logical_not(x)

        from yobx.tensorflow import to_onnx as _to_onnx

        onx = _to_onnx(model, (a,), input_names=["X"])
        expected = model(a).numpy()

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X:0": a})[0]
        self.assertEqualArray(expected, result)

        ort_result = _ort_run(onx, {"X:0": a})
        self.assertEqualArray(expected, ort_result)
        self.assertIn("Not", [n.op_type for n in onx.graph.node])

    def test_logical_or(self):
        """TF LogicalOr → ONNX Or."""
        a = np.array([[True, False, True], [False, True, False]])
        b = np.array([[True, True, False], [False, False, True]])
        onx = self._run_binary_op(tf.logical_or, a, b)
        self.assertIn("Or", [n.op_type for n in onx.graph.node])

    def test_logical_xor(self):
        """TF LogicalXor → ONNX Xor."""
        a = np.array([[True, False, True], [False, True, False]])
        b = np.array([[True, True, False], [False, False, True]])
        self._run_binary_op(tf.math.logical_xor, a, b)
        # Xor may be not used...


if __name__ == "__main__":
    unittest.main(verbosity=2)
