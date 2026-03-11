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

    def test_dense_linear_large_model(self):
        """to_onnx with large_model=True returns an ExtendedModelContainer."""
        from yobx.container import ExtendedModelContainer

        model = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(3,))])
        X = np.random.rand(5, 3).astype(np.float32)
        container = to_onnx(model, (X,), large_model=True)
        self.assertIsInstance(container, ExtendedModelContainer)


@requires_tensorflow("2.18")
class TestTensorflowUnaryOpConverters(ExtTestCase):
    """Tests for the unary element-wise op converters in unary_ops.py."""

    def _run_unary_op(self, tf_fn, x, disable_ort=False):
        """Trace tf_fn(x) to ONNX and compare TF vs ONNX vs ORT results."""

        @tf.function
        def model(inp):
            return tf_fn(inp)

        onx = to_onnx(model, (x,), input_names=["X"])
        expected = model(x).numpy()

        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X:0": x})[0]
        self.assertEqualArray(expected, result, atol=1e-5)

        if disable_ort:
            return onx
        ort_result = _ort_run(onx, {"X:0": x})
        self.assertEqualArray(expected, ort_result, atol=1e-5)
        return onx

    # ------------------------------------------------------------------
    # Exponential / Logarithm
    # ------------------------------------------------------------------

    def test_exp(self):
        """TF Exp → ONNX Exp."""
        x = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_unary_op(tf.exp, x)
        self.assertIn("Exp", [n.op_type for n in onx.graph.node])

    def test_log(self):
        """TF Log → ONNX Log."""
        x = np.random.rand(3, 4).astype(np.float32) + 0.1
        onx = self._run_unary_op(tf.math.log, x)
        self.assertIn("Log", [n.op_type for n in onx.graph.node])

    # ------------------------------------------------------------------
    # Trigonometric
    # ------------------------------------------------------------------

    def test_cos(self):
        """TF Cos → ONNX Cos."""
        x = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_unary_op(tf.math.cos, x)
        self.assertIn("Cos", [n.op_type for n in onx.graph.node])

    def test_sin(self):
        """TF Sin → ONNX Sin."""
        x = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_unary_op(tf.math.sin, x)
        self.assertIn("Sin", [n.op_type for n in onx.graph.node])

    def test_tan(self):
        """TF Tan → ONNX Tan."""
        x = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_unary_op(tf.math.tan, x)
        self.assertIn("Tan", [n.op_type for n in onx.graph.node])

    def test_acos(self):
        """TF Acos → ONNX Acos."""
        x = np.random.rand(3, 4).astype(np.float32) * 2 - 1
        onx = self._run_unary_op(tf.math.acos, x)
        self.assertIn("Acos", [n.op_type for n in onx.graph.node])

    def test_asin(self):
        """TF Asin → ONNX Asin."""
        x = np.random.rand(3, 4).astype(np.float32) * 2 - 1
        onx = self._run_unary_op(tf.math.asin, x)
        self.assertIn("Asin", [n.op_type for n in onx.graph.node])

    def test_atan(self):
        """TF Atan → ONNX Atan."""
        x = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_unary_op(tf.math.atan, x)
        self.assertIn("Atan", [n.op_type for n in onx.graph.node])

    # ------------------------------------------------------------------
    # Hyperbolic
    # ------------------------------------------------------------------

    def test_cosh(self):
        """TF Cosh → ONNX Cosh."""
        x = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_unary_op(tf.math.cosh, x)
        self.assertIn("Cosh", [n.op_type for n in onx.graph.node])

    def test_sinh(self):
        """TF Sinh → ONNX Sinh."""
        x = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_unary_op(tf.math.sinh, x)
        self.assertIn("Sinh", [n.op_type for n in onx.graph.node])

    # ------------------------------------------------------------------
    # Rounding / Magnitude
    # ------------------------------------------------------------------

    def test_abs(self):
        """TF Abs → ONNX Abs."""
        x = np.random.rand(3, 4).astype(np.float32) - 0.5
        onx = self._run_unary_op(tf.math.abs, x)
        self.assertIn("Abs", [n.op_type for n in onx.graph.node])

    def test_neg(self):
        """TF Neg → ONNX Neg."""
        x = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_unary_op(tf.math.negative, x)
        self.assertIn("Neg", [n.op_type for n in onx.graph.node])

    def test_sign(self):
        """TF Sign → ONNX Sign."""
        x = np.random.rand(3, 4).astype(np.float32) - 0.5
        onx = self._run_unary_op(tf.math.sign, x)
        self.assertIn("Sign", [n.op_type for n in onx.graph.node])

    def test_floor(self):
        """TF Floor → ONNX Floor."""
        x = np.random.rand(3, 4).astype(np.float32) * 10 - 5
        onx = self._run_unary_op(tf.math.floor, x)
        self.assertIn("Floor", [n.op_type for n in onx.graph.node])

    def test_ceil(self):
        """TF Ceil → ONNX Ceil."""
        x = np.random.rand(3, 4).astype(np.float32) * 10 - 5
        onx = self._run_unary_op(tf.math.ceil, x)
        self.assertIn("Ceil", [n.op_type for n in onx.graph.node])

    def test_round(self):
        """TF Round → ONNX Round."""
        x = np.random.rand(3, 4).astype(np.float32) * 10 - 5
        onx = self._run_unary_op(tf.math.round, x)
        self.assertIn("Round", [n.op_type for n in onx.graph.node])

    # ------------------------------------------------------------------
    # Square-root family
    # ------------------------------------------------------------------

    def test_sqrt(self):
        """TF Sqrt → ONNX Sqrt."""
        x = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_unary_op(tf.math.sqrt, x)
        self.assertIn("Sqrt", [n.op_type for n in onx.graph.node])

    def test_rsqrt(self):
        """TF Rsqrt → ONNX Reciprocal(Sqrt(x))."""
        x = np.random.rand(3, 4).astype(np.float32) + 0.1
        onx = self._run_unary_op(tf.math.rsqrt, x)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Sqrt", op_types)
        self.assertIn("Reciprocal", op_types)

    def test_square(self):
        """TF Square → ONNX Mul(x, x)."""
        x = np.random.rand(3, 4).astype(np.float32)
        onx = self._run_unary_op(tf.math.square, x)
        self.assertIn("Mul", [n.op_type for n in onx.graph.node])

    def test_reciprocal(self):
        """TF Reciprocal → ONNX Reciprocal."""
        x = np.random.rand(3, 4).astype(np.float32) + 0.1
        onx = self._run_unary_op(tf.math.reciprocal, x)
        self.assertIn("Reciprocal", [n.op_type for n in onx.graph.node])


@requires_tensorflow("2.18")
class TestTensorflowConvPoolPadConverters(ExtTestCase):
    """Tests for Conv2D, MaxPool, AvgPool, Pad, PadV2, and MirrorPad converters."""

    # ------------------------------------------------------------------
    # Pad / PadV2 / MirrorPad
    # ------------------------------------------------------------------

    def test_pad_constant(self):
        """TF Pad (zero padding) → ONNX Pad."""
        x = np.random.rand(2, 4, 4, 3).astype(np.float32)

        @tf.function
        def fn(t):
            return tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]])

        onx = to_onnx(fn, (x,))
        self.assertIn("Pad", [n.op_type for n in onx.graph.node])

        expected = fn(x).numpy()
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X:0": x})[0]
        self.assertEqualArray(expected, result, atol=1e-6)

        ort_result = _ort_run(onx, {"X:0": x})
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_pad_v2_constant_value(self):
        """TF PadV2 (constant value padding) → ONNX Pad."""
        x = np.random.rand(2, 4, 4, 3).astype(np.float32)

        @tf.function
        def fn(t):
            return tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]], constant_values=-1.0)

        onx = to_onnx(fn, (x,))
        self.assertIn("Pad", [n.op_type for n in onx.graph.node])

        expected = fn(x).numpy()
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X:0": x})[0]
        self.assertEqualArray(expected, result, atol=1e-6)

        ort_result = _ort_run(onx, {"X:0": x})
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_mirror_pad_reflect(self):
        """TF MirrorPad REFLECT → ONNX Pad(mode='reflect')."""
        x = np.random.rand(2, 6, 6, 3).astype(np.float32)

        @tf.function
        def fn(t):
            return tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")

        onx = to_onnx(fn, (x,))
        self.assertIn("Pad", [n.op_type for n in onx.graph.node])

        expected = fn(x).numpy()
        ref = ExtendedReferenceEvaluator(onx)
        result = ref.run(None, {"X:0": x})[0]
        self.assertEqualArray(expected, result, atol=1e-6)

        ort_result = _ort_run(onx, {"X:0": x})
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_keras_zero_padding2d(self):
        """Keras ZeroPadding2D → ONNX Pad."""
        model = tf.keras.Sequential(
            [tf.keras.layers.ZeroPadding2D(padding=(1, 1), input_shape=(8, 8, 3))]
        )
        X = np.random.rand(2, 8, 8, 3).astype(np.float32)
        expected = model(X).numpy()

        onx = to_onnx(model, (X,))
        self.assertIn("Pad", [n.op_type for n in onx.graph.node])

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    # ------------------------------------------------------------------
    # MaxPool
    # ------------------------------------------------------------------

    def test_maxpool_valid(self):
        """TF MaxPool with VALID padding → ONNX MaxPool."""
        model = tf.keras.Sequential(
            [tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid",
                                          input_shape=(8, 8, 4))]
        )
        X = np.random.rand(2, 8, 8, 4).astype(np.float32)
        expected = model(X).numpy()

        onx = to_onnx(model, (X,))
        self.assertIn("MaxPool", [n.op_type for n in onx.graph.node])

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_maxpool_same(self):
        """TF MaxPool with SAME padding → ONNX MaxPool with explicit pads."""
        model = tf.keras.Sequential(
            [tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same",
                                          input_shape=(8, 8, 4))]
        )
        X = np.random.rand(2, 8, 8, 4).astype(np.float32)
        expected = model(X).numpy()

        onx = to_onnx(model, (X,))
        self.assertIn("MaxPool", [n.op_type for n in onx.graph.node])

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    def test_maxpool_strides(self):
        """TF MaxPool with custom strides → ONNX MaxPool."""
        model = tf.keras.Sequential(
            [tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                          padding="valid", input_shape=(8, 8, 4))]
        )
        X = np.random.rand(2, 8, 8, 4).astype(np.float32)
        expected = model(X).numpy()

        onx = to_onnx(model, (X,))
        self.assertIn("MaxPool", [n.op_type for n in onx.graph.node])

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-6)

    # ------------------------------------------------------------------
    # AvgPool
    # ------------------------------------------------------------------

    def test_avgpool_valid(self):
        """TF AvgPool with VALID padding → ONNX AveragePool."""
        model = tf.keras.Sequential(
            [tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding="valid",
                                              input_shape=(8, 8, 4))]
        )
        X = np.random.rand(2, 8, 8, 4).astype(np.float32)
        expected = model(X).numpy()

        onx = to_onnx(model, (X,))
        self.assertIn("AveragePool", [n.op_type for n in onx.graph.node])

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_avgpool_same(self):
        """TF AvgPool with SAME padding → ONNX AveragePool with explicit pads."""
        model = tf.keras.Sequential(
            [tf.keras.layers.AveragePooling2D(pool_size=(3, 3), padding="same",
                                              strides=(1, 1), input_shape=(8, 8, 4))]
        )
        X = np.random.rand(2, 8, 8, 4).astype(np.float32)
        expected = model(X).numpy()

        onx = to_onnx(model, (X,))
        self.assertIn("AveragePool", [n.op_type for n in onx.graph.node])

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    # ------------------------------------------------------------------
    # Conv2D
    # ------------------------------------------------------------------

    def test_conv2d_valid(self):
        """TF Conv2D with VALID padding → ONNX Conv."""
        model = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(8, (3, 3), padding="valid",
                                    use_bias=False, input_shape=(8, 8, 3))]
        )
        X = np.random.rand(2, 8, 8, 3).astype(np.float32)
        expected = model(X).numpy()

        onx = to_onnx(model, (X,))
        self.assertIn("Conv", [n.op_type for n in onx.graph.node])

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_conv2d_same(self):
        """TF Conv2D with SAME padding → ONNX Conv with explicit pads."""
        model = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(8, (3, 3), padding="same",
                                    use_bias=False, input_shape=(8, 8, 3))]
        )
        X = np.random.rand(2, 8, 8, 3).astype(np.float32)
        expected = model(X).numpy()

        onx = to_onnx(model, (X,))
        self.assertIn("Conv", [n.op_type for n in onx.graph.node])

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_conv2d_with_bias(self):
        """TF Conv2D with bias → ONNX Conv + BiasAdd (Add)."""
        model = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(8, (3, 3), padding="valid",
                                    use_bias=True, input_shape=(8, 8, 3))]
        )
        X = np.random.rand(2, 8, 8, 3).astype(np.float32)
        expected = model(X).numpy()

        onx = to_onnx(model, (X,))
        self.assertIn("Conv", [n.op_type for n in onx.graph.node])

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_conv2d_strides(self):
        """TF Conv2D with custom strides → ONNX Conv."""
        model = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(4, (3, 3), strides=(2, 2), padding="valid",
                                    use_bias=False, input_shape=(8, 8, 3))]
        )
        X = np.random.rand(2, 8, 8, 3).astype(np.float32)
        expected = model(X).numpy()

        onx = to_onnx(model, (X,))
        self.assertIn("Conv", [n.op_type for n in onx.graph.node])

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-5)

    def test_conv2d_relu(self):
        """TF Conv2D followed by ReLU activation → ONNX Conv + Relu."""
        model = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="valid",
                                    use_bias=False, input_shape=(8, 8, 3))]
        )
        X = np.random.rand(2, 8, 8, 3).astype(np.float32)
        expected = model(X).numpy()

        onx = to_onnx(model, (X,))
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("Conv", op_types)
        self.assertIn("Relu", op_types)

        input_name = onx.graph.input[0].name
        feeds = {input_name: X}

        ort_result = _ort_run(onx, feeds)
        self.assertEqualArray(expected, ort_result, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
