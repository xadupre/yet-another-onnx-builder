"""
Compatibility tests: jax2onnx vs yobx JAX→ONNX converter.

Each test converts a JAX function with both ``jax2onnx.to_onnx`` (reference
converter) and ``yobx.tensorflow.to_onnx`` (converter under test) and verifies
that both produce ONNX models whose ORT-evaluated outputs match the original
JAX function output.

The test cases are inspired by jax2onnx's own test suite (primitives and small
composite models) so as to check that yobx produces similarly correct ONNX
models for the same inputs.
"""

import unittest
import numpy as np
from yobx.ext_test_case import ExtTestCase, requires_jax, requires_jax2onnx

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run_jax2onnx(fn, *arrays):
    """Convert *fn* with jax2onnx and return (onnx_model, ort_outputs)."""
    import jax
    from jax2onnx import to_onnx
    from onnxruntime import InferenceSession

    input_specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in arrays]
    model = to_onnx(fn, input_specs)
    sess = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    outputs = sess.run(None, dict(zip(input_names, arrays)))
    return model, outputs


def _run_yobx(fn, *arrays, dynamic_shapes=None):
    """Convert *fn* with yobx and return (export_artifact, ort_outputs)."""
    from yobx.tensorflow import to_onnx
    from onnxruntime import InferenceSession

    kwargs = {}
    if dynamic_shapes is not None:
        kwargs["dynamic_shapes"] = dynamic_shapes
    artifact = to_onnx(fn, tuple(arrays), **kwargs)
    sess = InferenceSession(artifact.SerializeToString(), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    outputs = sess.run(None, dict(zip(input_names, arrays)))
    return artifact, outputs


# ---------------------------------------------------------------------------
# Unary ops (inspired by jax2onnx primitive tests)
# ---------------------------------------------------------------------------


@requires_jax2onnx()
@requires_jax()
class TestJax2OnnxUnaryOpsCompat(ExtTestCase):
    """Unary JAX ops: verify yobx and jax2onnx produce numerically identical results."""

    _rng = np.random.default_rng(0)

    def _check_unary(self, fn, x, atol=1e-5):
        """Run *fn* through both converters and compare outputs with JAX reference."""
        expected = np.asarray(fn(x))
        _, ref_outputs = _run_jax2onnx(fn, x)
        _, yobx_outputs = _run_yobx(fn, x, dynamic_shapes=({0: "batch"},))

        self.assertEqualArray(expected, ref_outputs[0], atol=atol)
        self.assertEqualArray(expected, yobx_outputs[0], atol=atol)

    def test_sin(self):
        import jax.numpy as jnp

        x = self._rng.standard_normal((4, 3)).astype(np.float32)
        self._check_unary(jnp.sin, x)

    def test_cos(self):
        import jax.numpy as jnp

        x = self._rng.standard_normal((4, 3)).astype(np.float32)
        self._check_unary(jnp.cos, x)

    def test_tanh(self):
        import jax.numpy as jnp

        x = self._rng.standard_normal((4, 3)).astype(np.float32)
        self._check_unary(jnp.tanh, x)

    def test_exp(self):
        import jax.numpy as jnp

        x = self._rng.standard_normal((4, 3)).astype(np.float32)
        self._check_unary(jnp.exp, x)

    def test_log(self):
        import jax.numpy as jnp

        x = (self._rng.standard_normal((4, 3)) + 2.0).astype(np.float32)
        self._check_unary(jnp.log, x)

    def test_sqrt(self):
        import jax.numpy as jnp

        x = np.abs(self._rng.standard_normal((4, 3))).astype(np.float32) + 0.1
        self._check_unary(jnp.sqrt, x)

    def test_abs(self):
        import jax.numpy as jnp

        x = self._rng.standard_normal((4, 3)).astype(np.float32)
        self._check_unary(jnp.abs, x)

    def test_neg(self):
        import jax.numpy as jnp

        x = self._rng.standard_normal((4, 3)).astype(np.float32)
        self._check_unary(jnp.negative, x)

    def test_ceil(self):
        import jax.numpy as jnp

        x = (self._rng.standard_normal((4, 3)) * 3).astype(np.float32)
        self._check_unary(jnp.ceil, x)

    def test_floor(self):
        import jax.numpy as jnp

        x = (self._rng.standard_normal((4, 3)) * 3).astype(np.float32)
        self._check_unary(jnp.floor, x)

    def test_relu(self):
        import jax.nn as jnn

        x = self._rng.standard_normal((4, 3)).astype(np.float32)
        self._check_unary(jnn.relu, x)

    def test_softmax(self):
        import jax.nn as jnn

        x = self._rng.standard_normal((4, 8)).astype(np.float32)
        self._check_unary(lambda t: jnn.softmax(t, axis=-1), x)


# ---------------------------------------------------------------------------
# Binary ops (inspired by jax2onnx primitive tests)
# ---------------------------------------------------------------------------


@requires_jax2onnx()
@requires_jax()
class TestJax2OnnxBinaryOpsCompat(ExtTestCase):
    """Binary JAX ops: verify yobx and jax2onnx produce numerically identical results."""

    _rng = np.random.default_rng(1)

    def _check_binary(self, fn, x, y, atol=1e-5):
        """Run binary *fn(x, y)* through both converters and compare."""
        expected = np.asarray(fn(x, y))
        _, ref_outputs = _run_jax2onnx(fn, x, y)
        _, yobx_outputs = _run_yobx(fn, x, y)

        self.assertEqualArray(expected, ref_outputs[0], atol=atol)
        self.assertEqualArray(expected, yobx_outputs[0], atol=atol)

    def test_add(self):
        import jax.numpy as jnp

        x = self._rng.standard_normal((4, 3)).astype(np.float32)
        y = self._rng.standard_normal((4, 3)).astype(np.float32)
        self._check_binary(jnp.add, x, y)

    def test_subtract(self):
        import jax.numpy as jnp

        x = self._rng.standard_normal((4, 3)).astype(np.float32)
        y = self._rng.standard_normal((4, 3)).astype(np.float32)
        self._check_binary(jnp.subtract, x, y)

    def test_multiply(self):
        import jax.numpy as jnp

        x = self._rng.standard_normal((4, 3)).astype(np.float32)
        y = self._rng.standard_normal((4, 3)).astype(np.float32)
        self._check_binary(jnp.multiply, x, y)

    def test_divide(self):
        import jax.numpy as jnp

        x = self._rng.standard_normal((4, 3)).astype(np.float32)
        y = (self._rng.standard_normal((4, 3)) + 2.0).astype(np.float32)
        self._check_binary(jnp.divide, x, y, atol=1e-5)


# ---------------------------------------------------------------------------
# Composite functions (MLP, matmul, layer-norm)
# ---------------------------------------------------------------------------


@requires_jax2onnx()
@requires_jax()
class TestJax2OnnxCompositeCompat(ExtTestCase):
    """Composite JAX functions: yobx vs jax2onnx parity for MLP, matmul, layer-norm."""

    _rng = np.random.default_rng(2)

    def test_matmul(self):
        """Matrix multiplication with dynamic batch: (batch,3) @ (3,2) → (batch,2)."""
        import jax.numpy as jnp

        a = self._rng.standard_normal((4, 3)).astype(np.float32)
        b = self._rng.standard_normal((3, 2)).astype(np.float32)

        def fn(x, y):
            return jnp.matmul(x, y)

        expected = np.asarray(fn(a, b))
        _, ref_out = _run_jax2onnx(fn, a, b)
        # Only the first (batch) input has a dynamic axis 0; the weight matrix b
        # must keep its static shape so the contracting dimension (axis 0 of b = 3)
        # can be matched against axis 1 of a (= 3).
        _, yobx_out = _run_yobx(fn, a, b, dynamic_shapes=({0: "batch"}, {}))
        self.assertEqualArray(expected, ref_out[0], atol=1e-5)
        self.assertEqualArray(expected, yobx_out[0], atol=1e-5)

    def test_mlp(self):
        """Two-layer MLP with ReLU activation: compare converters end-to-end."""
        import jax

        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        W1 = jax.random.normal(k1, (4, 8), dtype=np.float32)
        b1 = np.zeros(8, dtype=np.float32)
        W2 = jax.random.normal(k2, (8, 2), dtype=np.float32)
        b2 = np.zeros(2, dtype=np.float32)

        def mlp(x):
            h = jax.nn.relu(x @ W1 + b1)
            return h @ W2 + b2

        X = self._rng.standard_normal((6, 4)).astype(np.float32)
        expected = np.asarray(mlp(X))
        _, ref_out = _run_jax2onnx(mlp, X)
        _, yobx_out = _run_yobx(mlp, X)
        self.assertEqualArray(expected, ref_out[0], atol=1e-4)
        self.assertEqualArray(expected, yobx_out[0], atol=1e-4)

    def test_layer_norm(self):
        """Manual layer normalisation: mean/std along last axis."""
        import jax.numpy as jnp

        def layer_norm(x):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            std = jnp.std(x, axis=-1, keepdims=True)
            return (x - mean) / (std + 1e-5)

        X = self._rng.standard_normal((5, 8)).astype(np.float32)
        expected = np.asarray(layer_norm(X))
        _, ref_out = _run_jax2onnx(layer_norm, X)
        _, yobx_out = _run_yobx(layer_norm, X)
        self.assertEqualArray(expected, ref_out[0], atol=1e-5)
        self.assertEqualArray(expected, yobx_out[0], atol=1e-5)

    def test_dot_add_tanh(self):
        """Linear + tanh: a single dense layer with non-linear activation."""
        import jax.numpy as jnp

        W = self._rng.standard_normal((4, 6)).astype(np.float32)
        b = self._rng.standard_normal(6).astype(np.float32)

        def fn(x):
            return jnp.tanh(x @ W + b)

        X = self._rng.standard_normal((3, 4)).astype(np.float32)
        expected = np.asarray(fn(X))
        _, ref_out = _run_jax2onnx(fn, X)
        _, yobx_out = _run_yobx(fn, X)
        self.assertEqualArray(expected, ref_out[0], atol=1e-5)
        self.assertEqualArray(expected, yobx_out[0], atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
