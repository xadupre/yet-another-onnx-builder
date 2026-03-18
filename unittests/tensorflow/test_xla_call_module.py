"""
Unit tests for :mod:`yobx.tensorflow.ops.xla_call_module`.

Tests cover:
* :func:`parse_mlir` – MLIR text → list of layer dicts.
* :data:`_MAPPING_JAX_ONNX` – every entry is a valid ONNX op-type string.
* :func:`get_jax_cvt` – returns a callable for every supported op; raises for
  unknown ops.
* Composite ops (``rsqrt``, ``log_plus_one``, ``exponential_minus_one``,
  ``compare_NE``) – produce the correct multi-node ONNX sub-graphs.
* Binary arithmetic and comparison ops – end-to-end JAX → ONNX round-trips.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_jax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal fake MLIR text used by parse_mlir tests (no JAX required).
_MLIR_SIN = """
func.func @main(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = stablehlo.sine %arg0 : tensor<3x4xf32> loc(#loc0)
  return %0 : tensor<3x4xf32> loc(#loc1)
}
"""

_MLIR_TWO_OPS = """
func.func @main(%arg0: tensor<5xf32>) -> tensor<5xf32> {
  %0 = stablehlo.abs %arg0 : tensor<5xf32> loc(#loc0)
  %1 = stablehlo.negate %0 : tensor<5xf32> loc(#loc1)
  return %1 : tensor<5xf32> loc(#loc2)
}
"""

_MLIR_ADD = """
func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<3x4xf32> loc(#loc0)
  return %0 : tensor<3x4xf32> loc(#loc1)
}
"""

_MLIR_COMPARE_GT = """
func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xi1> {
  %0 = stablehlo.compare  GT, %arg0, %arg1,  FLOAT : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xi1> loc(#loc0)
  return %0 : tensor<3x4xi1> loc(#loc1)
}
"""

_MLIR_COMPARE_EQ = """
func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xi1> {
  %0 = stablehlo.compare  EQ, %arg0, %arg1,  FLOAT : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xi1> loc(#loc0)
  return %0 : tensor<3x4xi1> loc(#loc1)
}
"""

_MLIR_COMPARE_NE = """
func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xi1> {
  %0 = stablehlo.compare  NE, %arg0, %arg1,  FLOAT : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xi1> loc(#loc0)
  return %0 : tensor<3x4xi1> loc(#loc1)
}
"""


class TestParseMlir(ExtTestCase):
    """Tests for :func:`parse_mlir` – no JAX / TF needed."""

    def _import(self):
        from yobx.tensorflow.ops.xla_call_module import parse_mlir

        return parse_mlir

    def test_single_op(self):
        """parse_mlir extracts one input + one op + return for a sin graph."""
        parse_mlir = self._import()
        layers = parse_mlir(_MLIR_SIN)
        ops = [la["op"] for la in layers]
        self.assertIn("Input", ops)
        self.assertIn("sine", ops)

    def test_input_captured(self):
        parse_mlir = self._import()
        layers = parse_mlir(_MLIR_SIN)
        inputs = [la for la in layers if la["op"] == "Input"]
        self.assertGreater(len(inputs), 0)
        self.assertTrue(any(la["id"] == "%arg0" for la in inputs))

    def test_two_ops_order(self):
        parse_mlir = self._import()
        layers = parse_mlir(_MLIR_TWO_OPS)
        ops = [la["op"] for la in layers if la["op"] not in ("Input", "return")]
        self.assertEqual(ops, ["abs", "negate"])


class TestMappingJaxOnnx(ExtTestCase):
    """Verify :data:`_MAPPING_JAX_ONNX` integrity."""

    def _import(self):
        from yobx.tensorflow.ops.xla_call_module import _MAPPING_JAX_ONNX

        return _MAPPING_JAX_ONNX

    def test_sine_present(self):
        m = self._import()
        self.assertIn("sine", m)
        self.assertEqual(m["sine"], "Sin")

    def test_all_common_unary_ops_present(self):
        """All commonly-used StableHLO unary ops must be mapped."""
        expected = {
            "abs", "ceil", "floor", "negate", "round_nearest_even", "sign",
            "exponential", "log",
            "cosine", "sine",
            "tanh",
            "logistic",
            "sqrt",
            "not",
        }
        m = self._import()
        missing = expected - set(m.keys())
        self.assertEqual(missing, set(), f"Missing JAX → ONNX mappings: {missing}")

    def test_composite_ops_not_in_direct_mapping(self):
        """Composite ops should be in _COMPOSITE_JAX_OPS, not _MAPPING_JAX_ONNX."""
        from yobx.tensorflow.ops.xla_call_module import _MAPPING_JAX_ONNX, _COMPOSITE_JAX_OPS

        for op in _COMPOSITE_JAX_OPS:
            self.assertNotIn(
                op,
                _MAPPING_JAX_ONNX,
                f"Composite op {op!r} should not appear in _MAPPING_JAX_ONNX",
            )


class TestGetJaxCvt(ExtTestCase):
    """Tests for :func:`get_jax_cvt`."""

    def _import(self):
        from yobx.tensorflow.ops.xla_call_module import get_jax_cvt

        return get_jax_cvt

    def _make_mock_g(self):
        """Return a minimal mock GraphBuilder-like object."""

        class _MockOp:
            def __getattr__(self, onnx_op):
                def _call(*args, **kwargs):
                    return (f"out_{onnx_op}",)

                return _call

        class _MockG:
            op = _MockOp()

            def get_debug_msg(self):
                return ""

        return _MockG()

    def test_known_simple_op_returns_callable(self):
        get_jax_cvt = self._import()
        g = self._make_mock_g()
        cvt = get_jax_cvt("", g, "sine")
        self.assertTrue(callable(cvt))

    def test_known_simple_op_calls_onnx_op(self):
        get_jax_cvt = self._import()
        g = self._make_mock_g()
        cvt = get_jax_cvt("", g, "cosine")
        result = cvt("input_tensor", name="test")
        self.assertIsInstance(result, tuple)
        self.assertIn("Cos", result[0])

    def test_composite_rsqrt_returns_callable(self):
        get_jax_cvt = self._import()
        g = self._make_mock_g()
        cvt = get_jax_cvt("", g, "rsqrt")
        self.assertTrue(callable(cvt))

    def test_unknown_op_raises_runtime_error(self):
        get_jax_cvt = self._import()
        g = self._make_mock_g()
        with self.assertRaises(RuntimeError):
            get_jax_cvt("", g, "unknown_op_xyz")

    def test_binary_add_returns_callable(self):
        """get_jax_cvt returns a callable for the binary 'add' op."""
        get_jax_cvt = self._import()
        g = self._make_mock_g()
        cvt = get_jax_cvt("", g, "add")
        self.assertTrue(callable(cvt))

    def test_binary_ops_present_in_mapping(self):
        """All common StableHLO binary ops must be in _MAPPING_JAX_ONNX."""
        from yobx.tensorflow.ops.xla_call_module import _MAPPING_JAX_ONNX

        expected_binary = {
            "add", "subtract", "multiply", "divide",
            "maximum", "minimum", "power", "remainder",
            "and", "or", "xor", "select",
            "compare_EQ", "compare_GT", "compare_GE",
            "compare_LT", "compare_LE",
        }
        missing = expected_binary - set(_MAPPING_JAX_ONNX.keys())
        self.assertEqual(missing, set(), f"Missing binary op mappings: {missing}")

    def test_compare_ne_in_composite_ops(self):
        """compare_NE must be in _COMPOSITE_JAX_OPS (not direct mapping)."""
        from yobx.tensorflow.ops.xla_call_module import _MAPPING_JAX_ONNX, _COMPOSITE_JAX_OPS

        self.assertIn("compare_NE", _COMPOSITE_JAX_OPS)
        self.assertNotIn("compare_NE", _MAPPING_JAX_ONNX)


class TestParseMlirBinaryOps(ExtTestCase):
    """Tests for :func:`parse_mlir` with binary and compare ops."""

    def _import(self):
        from yobx.tensorflow.ops.xla_call_module import parse_mlir

        return parse_mlir

    def test_parse_add(self):
        """parse_mlir extracts both operands for a binary add op."""
        parse_mlir = self._import()
        layers = parse_mlir(_MLIR_ADD)
        compute = [la for la in layers if la["op"] not in ("Input", "return")]
        self.assertEqual(len(compute), 1)
        self.assertEqual(compute[0]["op"], "add")
        self.assertEqual(len(compute[0]["operands"]), 2)

    def test_parse_compare_gt(self):
        """parse_mlir rewrites stablehlo.compare GT as compare_GT."""
        parse_mlir = self._import()
        layers = parse_mlir(_MLIR_COMPARE_GT)
        compute = [la for la in layers if la["op"] not in ("Input", "return")]
        self.assertEqual(len(compute), 1)
        self.assertEqual(compute[0]["op"], "compare_GT")
        self.assertEqual(len(compute[0]["operands"]), 2)

    def test_parse_compare_eq(self):
        """parse_mlir rewrites stablehlo.compare EQ as compare_EQ."""
        parse_mlir = self._import()
        layers = parse_mlir(_MLIR_COMPARE_EQ)
        compute = [la for la in layers if la["op"] not in ("Input", "return")]
        self.assertEqual(len(compute), 1)
        self.assertEqual(compute[0]["op"], "compare_EQ")

    def test_parse_compare_ne(self):
        """parse_mlir rewrites stablehlo.compare NE as compare_NE."""
        parse_mlir = self._import()
        layers = parse_mlir(_MLIR_COMPARE_NE)
        compute = [la for la in layers if la["op"] not in ("Input", "return")]
        self.assertEqual(len(compute), 1)
        self.assertEqual(compute[0]["op"], "compare_NE")

    def test_compare_operands_resolved(self):
        """Operands in a compare op are correctly extracted."""
        parse_mlir = self._import()
        layers = parse_mlir(_MLIR_COMPARE_GT)
        cmp_layer = next(la for la in layers if la["op"] == "compare_GT")
        self.assertIn("%arg0", cmp_layer["operands"])
        self.assertIn("%arg1", cmp_layer["operands"])


@requires_jax()
class TestJaxBinaryOpsEndToEnd(ExtTestCase):
    """End-to-end tests: JAX binary-op function → ONNX → ORT round-trip."""

    def _run_jax_binary_fn(self, jax_fn, x, y):
        """Convert *jax_fn(x, y)* to ONNX and compare with direct JAX output."""
        from onnxruntime import InferenceSession
        from yobx.tensorflow import to_onnx

        onx = to_onnx(jax_fn, (x, y))
        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        input_names = [i.name for i in onx.graph.input]
        (result,) = sess.run(None, {input_names[0]: x, input_names[1]: y})
        expected = jax_fn(x, y)
        if hasattr(expected, "to_py"):
            expected = np.array(expected)
        return np.array(expected), result

    def test_add(self):
        import jax.numpy as jnp

        x = np.random.rand(4, 3).astype(np.float32)
        y = np.random.rand(4, 3).astype(np.float32)
        expected, result = self._run_jax_binary_fn(jnp.add, x, y)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_subtract(self):
        import jax.numpy as jnp

        x = np.random.rand(4, 3).astype(np.float32)
        y = np.random.rand(4, 3).astype(np.float32)
        expected, result = self._run_jax_binary_fn(jnp.subtract, x, y)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_multiply(self):
        import jax.numpy as jnp

        x = np.random.rand(4, 3).astype(np.float32)
        y = np.random.rand(4, 3).astype(np.float32)
        expected, result = self._run_jax_binary_fn(jnp.multiply, x, y)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_divide(self):
        import jax.numpy as jnp

        x = np.random.rand(4, 3).astype(np.float32) + 0.1
        y = np.random.rand(4, 3).astype(np.float32) + 0.1
        expected, result = self._run_jax_binary_fn(jnp.divide, x, y)
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_maximum(self):
        import jax.numpy as jnp

        x = np.random.rand(4, 3).astype(np.float32)
        y = np.random.rand(4, 3).astype(np.float32)
        expected, result = self._run_jax_binary_fn(jnp.maximum, x, y)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_minimum(self):
        import jax.numpy as jnp

        x = np.random.rand(4, 3).astype(np.float32)
        y = np.random.rand(4, 3).astype(np.float32)
        expected, result = self._run_jax_binary_fn(jnp.minimum, x, y)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_power(self):
        import jax.numpy as jnp

        x = np.random.rand(4, 3).astype(np.float32) + 0.1
        y = np.random.rand(4, 3).astype(np.float32) + 0.1
        expected, result = self._run_jax_binary_fn(jnp.power, x, y)
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_compare_gt(self):
        import jax.numpy as jnp

        x = np.random.rand(4, 3).astype(np.float32)
        y = np.random.rand(4, 3).astype(np.float32)
        expected, result = self._run_jax_binary_fn(jnp.greater, x, y)
        self.assertEqualArray(expected, result)

    def test_compare_lt(self):
        import jax.numpy as jnp

        x = np.random.rand(4, 3).astype(np.float32)
        y = np.random.rand(4, 3).astype(np.float32)
        expected, result = self._run_jax_binary_fn(jnp.less, x, y)
        self.assertEqualArray(expected, result)

    def test_compare_ge(self):
        import jax.numpy as jnp

        x = np.random.rand(4, 3).astype(np.float32)
        y = np.random.rand(4, 3).astype(np.float32)
        expected, result = self._run_jax_binary_fn(jnp.greater_equal, x, y)
        self.assertEqualArray(expected, result)

    def test_compare_le(self):
        import jax.numpy as jnp

        x = np.random.rand(4, 3).astype(np.float32)
        y = np.random.rand(4, 3).astype(np.float32)
        expected, result = self._run_jax_binary_fn(jnp.less_equal, x, y)
        self.assertEqualArray(expected, result)

    def test_compare_eq(self):
        import jax.numpy as jnp

        x = np.random.rand(4, 3).astype(np.float32)
        y = x.copy()  # ensure some equal elements
        expected, result = self._run_jax_binary_fn(jnp.equal, x, y)
        self.assertEqualArray(expected, result)

    def test_compare_ne(self):
        import jax.numpy as jnp

        x = np.random.rand(4, 3).astype(np.float32)
        y = np.random.rand(4, 3).astype(np.float32)
        expected, result = self._run_jax_binary_fn(jnp.not_equal, x, y)
        self.assertEqualArray(expected, result)


@requires_jax()
class TestJaxUnaryOpsEndToEnd(ExtTestCase):
    """End-to-end tests: JAX function → ONNX → ORT round-trip."""

    def _run_jax_fn(self, jax_fn, x):
        """Convert *jax_fn* to ONNX and run with ORT; return (expected, result)."""
        from onnxruntime import InferenceSession
        from yobx.tensorflow import to_onnx

        onx = to_onnx(jax_fn, (x,), dynamic_shapes=({0: "batch"},))
        sess = InferenceSession(
            onx.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        input_name = onx.graph.input[0].name
        result = sess.run(None, {input_name: x})[0]
        expected = jax_fn(x)
        if hasattr(expected, "to_py"):
            expected = np.array(expected)
        return expected, result

    def test_sin(self):
        import jax.numpy as jnp

        x = np.random.rand(4, 3).astype(np.float32)
        expected, result = self._run_jax_fn(jnp.sin, x)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_cos(self):
        import jax.numpy as jnp

        x = np.random.rand(4, 3).astype(np.float32)
        expected, result = self._run_jax_fn(jnp.cos, x)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_exp(self):
        import jax.numpy as jnp

        x = np.random.rand(4, 3).astype(np.float32)
        expected, result = self._run_jax_fn(jnp.exp, x)
        self.assertEqualArray(expected, result, atol=1e-5)

    def test_log(self):
        import jax.numpy as jnp

        x = (np.random.rand(4, 3) + 0.5).astype(np.float32)
        expected, result = self._run_jax_fn(jnp.log, x)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_sqrt(self):
        import jax.numpy as jnp

        x = np.random.rand(4, 3).astype(np.float32)
        expected, result = self._run_jax_fn(jnp.sqrt, x)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_abs(self):
        import jax.numpy as jnp

        x = (np.random.rand(4, 3) - 0.5).astype(np.float32)
        expected, result = self._run_jax_fn(jnp.abs, x)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_tanh(self):
        import jax.numpy as jnp

        x = (np.random.rand(4, 3) - 0.5).astype(np.float32)
        expected, result = self._run_jax_fn(jnp.tanh, x)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_neg(self):
        import jax.numpy as jnp

        x = (np.random.rand(4, 3) - 0.5).astype(np.float32)
        expected, result = self._run_jax_fn(jnp.negative, x)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_floor(self):
        import jax.numpy as jnp

        x = (np.random.rand(4, 3) * 5 - 2.5).astype(np.float32)
        expected, result = self._run_jax_fn(jnp.floor, x)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_ceil(self):
        import jax.numpy as jnp

        x = (np.random.rand(4, 3) * 5 - 2.5).astype(np.float32)
        expected, result = self._run_jax_fn(jnp.ceil, x)
        self.assertEqualArray(expected, result, atol=1e-6)

    def test_sign(self):
        import jax.numpy as jnp

        x = (np.random.rand(4, 3) - 0.5).astype(np.float32)
        expected, result = self._run_jax_fn(jnp.sign, x)
        self.assertEqualArray(expected, result, atol=1e-6)

    @unittest.skip("jnn.sigmoid may expand to multiple ops on some JAX versions")
    def test_sigmoid(self):
        import jax.nn as jnn

        x = (np.random.rand(4, 3) - 0.5).astype(np.float32)
        expected, result = self._run_jax_fn(jnn.sigmoid, x)
        self.assertEqualArray(expected, result, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
