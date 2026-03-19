"""
.. _l-plot-jax-to-onnx:

Converting a JAX function to ONNX
==================================

:func:`yobx.tensorflow.to_onnx` can also convert :epkg:`JAX` functions to
ONNX.  Under the hood it uses :func:`jax.experimental.jax2tf.convert` to
lower the JAX computation to a :class:`tensorflow.ConcreteFunction` and then
applies the same TF→ONNX conversion pipeline used for Keras models.

Alternatively, :func:`yobx.tensorflow.tensorflow_helper.jax_to_concrete_function`
can be called explicitly to obtain the intermediate
:class:`~tensorflow.ConcreteFunction` before passing it to
:func:`~yobx.tensorflow.to_onnx`.

The workflow is:

1. **Write** a plain JAX function (or wrap a :mod:`flax`/:mod:`equinox` model
   in a function).
2. Call :func:`yobx.tensorflow.to_onnx` with a representative *dummy input*.
   The converter detects that the callable is a JAX function and automatically
   routes it through :func:`~yobx.tensorflow.tensorflow_helper.jax_to_concrete_function`.
3. **Run** the exported ONNX model with any ONNX runtime — this example uses
   :epkg:`onnxruntime`.
4. **Verify** that the ONNX outputs match JAX's own outputs.
"""

# %%
import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime
from yobx.doc import plot_dot
from yobx.helpers.onnx_helper import pretty_onnx
from yobx.tensorflow import to_onnx
from yobx.tensorflow.tensorflow_helper import jax_to_concrete_function

# %%
# 1. Simple element-wise function
# --------------------------------
#
# We start with the simplest possible JAX function: an element-wise
# ``sin`` applied to a float32 matrix.  :func:`to_onnx` auto-detects that
# the callable is a JAX function and converts it transparently.

rng = np.random.default_rng(0)
X = rng.standard_normal((5, 4)).astype(np.float32)


def jax_sin(x):
    return jnp.sin(x)


onx_sin = to_onnx(jax_sin, (X,))

print("Opset            :", onx_sin.opset_import[0].version)
print("Number of nodes  :", len(onx_sin.graph.node))
print("Node op-types    :", [n.op_type for n in onx_sin.graph.node])

# %%
# Run and compare
# ~~~~~~~~~~~~~~~~
#
# Verify that the ONNX model reproduces the JAX output.

ref_sin = onnxruntime.InferenceSession(
    onx_sin.SerializeToString(), providers=["CPUExecutionProvider"]
)
input_name = ref_sin.get_inputs()[0].name
(result_sin,) = ref_sin.run(None, {input_name: X})

expected_sin = np.asarray(jax_sin(X))
print("\nJAX  output (first row):", expected_sin[0])
print("ONNX output (first row):", result_sin[0])
assert np.allclose(expected_sin, result_sin, atol=1e-5), "Mismatch!"
print("Outputs match ✓")

# %%
# 2. Multi-layer MLP in JAX
# --------------------------
#
# A slightly more complex function: a two-layer MLP with ReLU activations
# whose weights are stored as JAX arrays captured in a closure.

key = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(key)

W1 = jax.random.normal(k1, (8, 16), dtype=np.float32)
b1 = np.zeros(16, dtype=np.float32)
W2 = jax.random.normal(k2, (16, 4), dtype=np.float32)
b2 = np.zeros(4, dtype=np.float32)


def jax_mlp(x):
    h = jax.nn.relu(x @ W1 + b1)
    return h @ W2 + b2


X_mlp = rng.standard_normal((10, 8)).astype(np.float32)
onx_mlp = to_onnx(jax_mlp, (X_mlp,))

op_types = [n.op_type for n in onx_mlp.graph.node]
print("\nOp-types in the MLP graph:", op_types)
assert "MatMul" in op_types

# %%
# Display the model.
print(pretty_onnx(onx_mlp))

# %%
# Verify predictions on a held-out batch.

ref_mlp = onnxruntime.InferenceSession(
    onx_mlp.SerializeToString(), providers=["CPUExecutionProvider"]
)
input_name_mlp = ref_mlp.get_inputs()[0].name
(result_mlp,) = ref_mlp.run(None, {input_name_mlp: X_mlp})

expected_mlp = np.asarray(jax_mlp(X_mlp))
np.testing.assert_allclose(expected_mlp, result_mlp, atol=1e-2)
print("MLP predictions match ✓")

# %%
# 3. Dynamic batch dimension
# ---------------------------
#
# By default :func:`to_onnx` marks axis 0 as a dynamic (symbolic) batch
# dimension.  The converted model runs correctly for any batch size.

onx_dyn = to_onnx(jax_mlp, (X_mlp,), dynamic_shapes=({0: "batch"},))

input_shape = onx_dyn.graph.input[0].type.tensor_type.shape
batch_dim = input_shape.dim[0]
print("\nBatch dimension param  :", batch_dim.dim_param)
assert batch_dim.dim_param, "Expected a named dynamic dimension"

ref_dyn = onnxruntime.InferenceSession(
    onx_dyn.SerializeToString(), providers=["CPUExecutionProvider"]
)
input_name_dyn = ref_dyn.get_inputs()[0].name
for n in (1, 7, 20):
    X_batch = rng.standard_normal((n, 8)).astype(np.float32)
    (out,) = ref_dyn.run(None, {input_name_dyn: X_batch})
    expected = np.asarray(jax_mlp(X_batch))
    np.testing.assert_allclose(expected, out, atol=1e-2)

print("Dynamic-batch model verified for batch sizes 1, 7, 20 ✓")

# %%
# 4. Explicit jax_to_concrete_function
# ---------------------------------------
#
# :func:`~yobx.tensorflow.tensorflow_helper.jax_to_concrete_function` can be
# called directly when you want to inspect or reuse the intermediate
# :class:`~tensorflow.ConcreteFunction` before exporting to ONNX.


def jax_softmax(x):
    return jax.nn.softmax(x, axis=-1)


X_cls = rng.standard_normal((6, 10)).astype(np.float32)

cf = jax_to_concrete_function(jax_softmax, (X_cls,), dynamic_shapes=({0: "batch"},))
onx_cls = to_onnx(cf, (X_cls,), dynamic_shapes=({0: "batch"},))

ref_cls = onnxruntime.InferenceSession(
    onx_cls.SerializeToString(), providers=["CPUExecutionProvider"]
)
input_name_cls = ref_cls.get_inputs()[0].name
(result_cls,) = ref_cls.run(None, {input_name_cls: X_cls})

expected_cls = np.asarray(jax_softmax(X_cls))
np.testing.assert_allclose(expected_cls, result_cls, atol=1e-5)
print("Explicit jax_to_concrete_function verified ✓")

# %%
# 5. Visualize the ONNX graph
# ----------------------------
#
plot_dot(onx_mlp)
