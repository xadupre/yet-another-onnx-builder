"""
Unit tests for :mod:`yobx.tensorflow.ops.xla_call_module`.

Tests cover:
* :class:`XlaLayer` – layer class replacing plain dicts.
* :func:`parse_mlir` – MLIR text → list of :class:`XlaLayer` objects.
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
  %0 = stablehlo.compare  GT, %arg0, %arg1,  FLOAT :
        (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xi1> loc(#loc0)
  return %0 : tensor<3x4xi1> loc(#loc1)
}
"""

_MLIR_COMPARE_EQ = """
func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xi1> {
  %0 = stablehlo.compare  EQ, %arg0, %arg1,  FLOAT :
        (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xi1> loc(#loc0)
  return %0 : tensor<3x4xi1> loc(#loc1)
}
"""

_MLIR_COMPARE_NE = """
func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xi1> {
  %0 = stablehlo.compare  NE, %arg0, %arg1,  FLOAT :
        (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xi1> loc(#loc0)
  return %0 : tensor<3x4xi1> loc(#loc1)
}
"""


class TestXlaLayer(ExtTestCase):
    """Tests for :class:`~yobx.tensorflow.ops.xla_call_module_layer.XlaLayer`."""

    def _import(self):
        from yobx.tensorflow.ops.xla_call_module_layer import XlaLayer

        return XlaLayer

    def test_core_fields_attribute_access(self):
        """Core fields are accessible as attributes."""
        XlaLayer = self._import()
        layer = XlaLayer(
            id="%0", op="sine", operands=["%arg0"], shape="tensor<3x4xf32>", loc="#loc0"
        )
        self.assertEqual(layer.id, "%0")
        self.assertEqual(layer.op, "sine")
        self.assertEqual(layer.operands, ["%arg0"])
        self.assertEqual(layer.shape, "tensor<3x4xf32>")
        self.assertEqual(layer.loc, "#loc0")

    def test_dict_style_getitem(self):
        """Dict-style ``layer["key"]`` access works for all core fields."""
        XlaLayer = self._import()
        layer = XlaLayer(id="%1", op="add", operands=["%a", "%b"], shape="tensor<f32>")
        self.assertEqual(layer["id"], "%1")
        self.assertEqual(layer["op"], "add")
        self.assertEqual(layer["operands"], ["%a", "%b"])
        self.assertEqual(layer["shape"], "tensor<f32>")

    def test_dict_style_setitem(self):
        """Dict-style ``layer["key"] = value`` assignment works."""
        XlaLayer = self._import()
        layer = XlaLayer(id="%0", op="sine", operands=["%arg0"])
        layer["shape"] = "tensor<5xf32>"
        self.assertEqual(layer.shape, "tensor<5xf32>")

    def test_get_with_default(self):
        """``layer.get(key, default)`` returns the value or the default."""
        XlaLayer = self._import()
        layer = XlaLayer(id="%0", op="constant", operands=[], dense_content="1.0")
        self.assertEqual(layer.get("dense_content", ""), "1.0")
        self.assertEqual(layer.get("axes", []), [])
        self.assertEqual(layer.get("func", ""), "")
        self.assertIsNone(layer.get("nonexistent_key"))
        self.assertEqual(layer.get("nonexistent_key", "default"), "default")

    def test_getitem_missing_key_raises(self):
        """``layer["missing"]`` raises :exc:`KeyError`."""
        XlaLayer = self._import()
        layer = XlaLayer(id="%0", op="sine", operands=[])
        with self.assertRaises(KeyError):
            _ = layer["nonexistent_field"]

    def test_optional_fields_default_to_empty(self):
        """Optional fields default to empty lists or empty strings."""
        XlaLayer = self._import()
        layer = XlaLayer(id="%0", op="sine", operands=["%arg0"])
        self.assertEqual(layer.dense_content, "")
        self.assertEqual(layer.dims, [])
        self.assertEqual(layer.axes, [])
        self.assertEqual(layer.func, "")
        self.assertEqual(layer.lhs_contracting, [])
        self.assertEqual(layer.rhs_contracting, [])

    def test_constant_layer(self):
        """``constant`` layer stores ``dense_content``."""
        XlaLayer = self._import()
        layer = XlaLayer(
            id="%c", op="constant", operands=[], shape="tensor<3xf32>", dense_content="1.0"
        )
        self.assertEqual(layer["op"], "constant")
        self.assertEqual(layer["dense_content"], "1.0")
        self.assertEqual(layer.get("dense_content", ""), "1.0")

    def test_reduce_layer_axes(self):
        """``reduce_*`` layers store ``axes``."""
        XlaLayer = self._import()
        layer = XlaLayer(id="%r", op="reduce_max", operands=["%x"], axes=[0, 1])
        self.assertEqual(layer.axes, [0, 1])
        self.assertEqual(layer["axes"], [0, 1])
        self.assertEqual(layer.get("axes", []), [0, 1])

    def test_dot_general_contracting_dims(self):
        """``dot_general`` layers store ``lhs_contracting`` / ``rhs_contracting``."""
        XlaLayer = self._import()
        layer = XlaLayer(
            id="%mm",
            op="dot_general",
            operands=["%a", "%b"],
            lhs_contracting=[1],
            rhs_contracting=[0],
        )
        self.assertEqual(layer.lhs_contracting, [1])
        self.assertEqual(layer.rhs_contracting, [0])

    def test_call_layer_func(self):
        """``call`` layers store the callee ``func`` name."""
        XlaLayer = self._import()
        layer = XlaLayer(id="%out", op="call", operands=["%arg0"], func="relu")
        self.assertEqual(layer.func, "relu")
        self.assertEqual(layer["func"], "relu")

    def test_repr_contains_op_and_id(self):
        """``repr`` includes at least the op and id."""
        XlaLayer = self._import()
        layer = XlaLayer(id="%0", op="tanh", operands=["%arg0"])
        r = repr(layer)
        self.assertIn("tanh", r)
        self.assertIn("%0", r)

    def test_parse_mlir_returns_xla_layers(self):
        """``parse_mlir`` returns a list of :class:`XlaLayer` objects."""
        from yobx.tensorflow.ops.xla_call_module_layer import XlaLayer
        from yobx.tensorflow.ops.xla_call_module import parse_mlir

        layers = parse_mlir(_MLIR_SIN)
        for layer in layers:
            self.assertIsInstance(layer, XlaLayer)

    def test_parse_mlir_xla_layer_field_access(self):
        """Layers returned by ``parse_mlir`` support both attribute and dict access."""
        from yobx.tensorflow.ops.xla_call_module import parse_mlir

        layers = parse_mlir(_MLIR_TWO_OPS)
        compute = [la for la in layers if la.op not in ("Input", "return")]
        self.assertEqual(len(compute), 2)
        # Attribute access
        self.assertEqual(compute[0].op, "abs")
        self.assertEqual(compute[1].op, "negate")
        # Dict-style access
        self.assertEqual(compute[0]["op"], "abs")
        self.assertEqual(compute[1]["op"], "negate")


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
            "abs",
            "ceil",
            "floor",
            "negate",
            "round_nearest_even",
            "sign",
            "exponential",
            "log",
            "cosine",
            "sine",
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
            "add",
            "subtract",
            "multiply",
            "divide",
            "maximum",
            "minimum",
            "power",
            "remainder",
            "and",
            "or",
            "xor",
            "select",
            "compare_EQ",
            "compare_GT",
            "compare_GE",
            "compare_LT",
            "compare_LE",
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
        sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
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
        sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
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

    def test_sigmoid(self):
        import jax.nn as jnn

        x = (np.random.rand(4, 3) - 0.5).astype(np.float32)
        expected, result = self._run_jax_fn(jnn.sigmoid, x)
        self.assertEqualArray(expected, result, atol=1e-6)


# ---------------------------------------------------------------------------
# New ops: constant, dot_general, broadcast_in_dim, reduce, call (relu)
# ---------------------------------------------------------------------------

_MLIR_CONSTANT = """
func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %cst = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32> loc(#loc0)
  %0 = stablehlo.add %arg0, %cst : tensor<2xf32> loc(#loc1)
  return %0 : tensor<2xf32> loc(#loc2)
}
"""

_MLIR_DOT_GENERAL = """
func.func @main(%arg0: tensor<3x4xf32>, %arg1: tensor<4x2xf32>) -> tensor<3x2xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
        : (tensor<3x4xf32>, tensor<4x2xf32>) -> tensor<3x2xf32> loc(#loc0)
  return %0 : tensor<3x2xf32> loc(#loc1)
}
"""

_MLIR_BROADCAST_IN_DIM = """
func.func @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst = stablehlo.constant dense<1.0> : tensor<f32> loc(#loc0)
  %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3xf32> loc(#loc1)
  %1 = stablehlo.add %arg0, %0 : tensor<3xf32> loc(#loc2)
  return %1 : tensor<3xf32> loc(#loc3)
}
"""


class TestParseMlirNewOps(ExtTestCase):
    """Tests for new ops in :func:`parse_mlir`: constant, dot_general, broadcast_in_dim."""

    def _import(self):
        from yobx.tensorflow.ops.xla_call_module import parse_mlir

        return parse_mlir

    def test_parse_constant(self):
        """parse_mlir extracts a stablehlo.constant layer."""
        parse_mlir = self._import()
        layers = parse_mlir(_MLIR_CONSTANT)
        const_layers = [la for la in layers if la["op"] == "constant"]
        self.assertEqual(len(const_layers), 1)
        self.assertEqual(const_layers[0]["id"], "%cst")
        self.assertIn("dense_content", const_layers[0])

    def test_parse_dot_general(self):
        """parse_mlir extracts a stablehlo.dot_general layer with contracting dims."""
        parse_mlir = self._import()
        layers = parse_mlir(_MLIR_DOT_GENERAL)
        dot_layers = [la for la in layers if la["op"] == "dot_general"]
        self.assertEqual(len(dot_layers), 1)
        self.assertEqual(dot_layers[0]["operands"], ["%arg0", "%arg1"])
        self.assertEqual(dot_layers[0]["lhs_contracting"], [1])
        self.assertEqual(dot_layers[0]["rhs_contracting"], [0])

    def test_parse_broadcast_in_dim(self):
        """parse_mlir extracts stablehlo.broadcast_in_dim layers."""
        parse_mlir = self._import()
        layers = parse_mlir(_MLIR_BROADCAST_IN_DIM)
        bcast_layers = [la for la in layers if la["op"] == "broadcast_in_dim"]
        self.assertGreater(len(bcast_layers), 0)


class TestParseTensorType(ExtTestCase):
    """Tests for :func:`_parse_tensor_type`."""

    def _import(self):
        from yobx.tensorflow.ops.xla_call_module import _parse_tensor_type

        return _parse_tensor_type

    def test_scalar_f32(self):
        _parse_tensor_type = self._import()
        shape, dtype = _parse_tensor_type("tensor<f32>")
        self.assertEqual(shape, ())
        self.assertEqual(dtype, np.float32)

    def test_1d_f32(self):
        _parse_tensor_type = self._import()
        shape, dtype = _parse_tensor_type("tensor<16xf32>")
        self.assertEqual(shape, (16,))
        self.assertEqual(dtype, np.float32)

    def test_2d_f32(self):
        _parse_tensor_type = self._import()
        shape, dtype = _parse_tensor_type("tensor<8x16xf32>")
        self.assertEqual(shape, (8, 16))
        self.assertEqual(dtype, np.float32)

    def test_dynamic_dim(self):
        _parse_tensor_type = self._import()
        shape, dtype = _parse_tensor_type("tensor<?x10xf32>")
        self.assertEqual(shape, (-1, 10))
        self.assertEqual(dtype, np.float32)

    def test_scalar_i32(self):
        _parse_tensor_type = self._import()
        shape, dtype = _parse_tensor_type("tensor<i32>")
        self.assertEqual(shape, ())
        self.assertEqual(dtype, np.int32)

    def test_1d_i32(self):
        _parse_tensor_type = self._import()
        shape, dtype = _parse_tensor_type("tensor<1xi32>")
        self.assertEqual(shape, (1,))
        self.assertEqual(dtype, np.int32)


class TestParseDenseValue(ExtTestCase):
    """Tests for :func:`_parse_dense_value`."""

    def _import(self):
        from yobx.tensorflow.ops.xla_call_module import _parse_dense_value

        return _parse_dense_value

    def test_scalar_zero(self):
        _parse_dense_value = self._import()
        arr = _parse_dense_value("0.000000e+00", (4,), np.float32)
        np.testing.assert_array_equal(arr, [0.0, 0.0, 0.0, 0.0])

    def test_hex_neg_inf(self):
        """0xFF800000 is the IEEE 754 big-endian bit pattern for -inf."""
        _parse_dense_value = self._import()
        val = _parse_dense_value("0xFF800000", (), np.float32)
        self.assertTrue(np.isneginf(val))

    def test_nested_list(self):
        _parse_dense_value = self._import()
        arr = _parse_dense_value("[[1.0, 2.0], [3.0, 4.0]]", (2, 2), np.float32)
        np.testing.assert_allclose(arr, [[1.0, 2.0], [3.0, 4.0]])

    def test_scalar_integer(self):
        _parse_dense_value = self._import()
        val = _parse_dense_value("1", (), np.int32)
        self.assertEqual(int(val), 1)


@requires_jax()
class TestJaxMlpEndToEnd(ExtTestCase):
    """End-to-end tests for the MLP example from :ref:`l-plot-jax-to-onnx`."""

    def _run(self, jax_fn, x, dynamic_shapes=None):
        from onnxruntime import InferenceSession
        from yobx.tensorflow import to_onnx

        kwargs = {}
        if dynamic_shapes is not None:
            kwargs["dynamic_shapes"] = dynamic_shapes
        onx = to_onnx(jax_fn, (x,), **kwargs)
        sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        input_name = onx.graph.input[0].name
        (result,) = sess.run(None, {input_name: x})
        expected = np.asarray(jax_fn(x))
        return expected, result, onx

    def test_mlp_static(self):
        """Static MLP: MatMul, relu (Max), MatMul → correct predictions."""
        import jax

        rng = np.random.default_rng(0)
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        W1 = jax.random.normal(k1, (8, 16), dtype=np.float32)
        b1 = np.zeros(16, dtype=np.float32)
        W2 = jax.random.normal(k2, (16, 4), dtype=np.float32)
        b2 = np.zeros(4, dtype=np.float32)

        def jax_mlp(x):
            h = jax.nn.relu(x @ W1 + b1)
            return h @ W2 + b2

        X = rng.standard_normal((10, 8)).astype(np.float32)
        expected, result, onx = self._run(jax_mlp, X)
        self.assertEqualArray(expected, result, atol=1e-2)
        op_types = [n.op_type for n in onx.graph.node]
        self.assertIn("MatMul", op_types)

    def test_mlp_dynamic_batch(self):
        """Dynamic-batch MLP: model accepts any batch size."""
        import jax

        rng = np.random.default_rng(1)
        key = jax.random.PRNGKey(7)
        k1, k2 = jax.random.split(key)
        W1 = jax.random.normal(k1, (4, 8), dtype=np.float32)
        b1 = np.zeros(8, dtype=np.float32)
        W2 = jax.random.normal(k2, (8, 3), dtype=np.float32)
        b2 = np.zeros(3, dtype=np.float32)

        def mlp(x):
            return jax.nn.relu(x @ W1 + b1) @ W2 + b2

        X = rng.standard_normal((5, 4)).astype(np.float32)
        _, _, onx = self._run(mlp, X, dynamic_shapes=({0: "batch"},))
        batch_dim = onx.graph.input[0].type.tensor_type.shape.dim[0]
        self.assertTrue(batch_dim.dim_param, "Expected a named dynamic batch dim")

        from onnxruntime import InferenceSession

        sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        iname = onx.graph.input[0].name
        for n in (1, 3, 9):
            Xn = rng.standard_normal((n, 4)).astype(np.float32)
            (out,) = sess.run(None, {iname: Xn})
            self.assertEqualArray(np.asarray(mlp(Xn)), out, atol=1e-5)

    def test_softmax(self):
        """Softmax via explicit jax_to_concrete_function."""
        import jax

        from yobx.tensorflow import to_onnx
        from yobx.tensorflow.tensorflow_helper import jax_to_concrete_function
        from onnxruntime import InferenceSession

        def jax_softmax(x):
            return jax.nn.softmax(x, axis=-1)

        rng = np.random.default_rng(42)
        X = rng.standard_normal((6, 10)).astype(np.float32)
        cf = jax_to_concrete_function(jax_softmax, (X,), dynamic_shapes=({0: "batch"},))
        onx = to_onnx(cf, (X,), dynamic_shapes=({0: "batch"},))
        sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
        (result,) = sess.run(None, {onx.graph.input[0].name: X})
        self.assertEqualArray(np.asarray(jax_softmax(X)), result, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
