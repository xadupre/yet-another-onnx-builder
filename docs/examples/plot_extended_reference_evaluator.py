"""
.. _l-plot-extended-reference-evaluator:

ExtendedReferenceEvaluator: running models with contrib operators
=================================================================

:class:`ExtendedReferenceEvaluator
<yobx.reference.evaluator.ExtendedReferenceEvaluator>` extends
:class:`onnx.reference.ReferenceEvaluator` with additional operator kernels for
non-standard domains such as ``com.microsoft``.

This makes it possible to execute and test ONNX models that contain ONNX
Runtime contrib operators (e.g. ``FusedMatMul``, ``QuickGelu``) without
needing a full ONNX Runtime installation just for unit-testing an
optimization pattern.

This example shows:

1. Running a model with standard ONNX operators.
2. Running a model that uses the ``FusedMatMul`` contrib operator.
3. Running a model that uses the ``QuickGelu`` contrib operator.
4. Adding a custom operator implementation via ``new_ops``.
"""

import numpy as np
import onnx
import onnx.helper as oh
from yobx.reference import ExtendedReferenceEvaluator

TFLOAT = onnx.TensorProto.FLOAT

# %%
# 1. Standard ONNX operators
# --------------------------
#
# :class:`ExtendedReferenceEvaluator` is a drop-in replacement for
# :class:`onnx.reference.ReferenceEvaluator`.  Any model that runs
# with the standard evaluator also runs here.

model_add = oh.make_model(
    oh.make_graph(
        [oh.make_node("Add", ["X", "Y"], ["Z"])],
        "add_graph",
        [
            oh.make_tensor_value_info("X", TFLOAT, [None, None]),
            oh.make_tensor_value_info("Y", TFLOAT, [None, None]),
        ],
        [oh.make_tensor_value_info("Z", TFLOAT, [None, None])],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=10,
)

x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
ref = ExtendedReferenceEvaluator(model_add)
(result,) = ref.run(None, {"X": x, "Y": x})
print("Add result:\n", result)
assert np.allclose(result, x + x)

# %%
# 2. FusedMatMul (com.microsoft contrib operator)
# ------------------------------------------------
#
# ``FusedMatMul`` is an ONNX Runtime contrib operator that fuses a matrix
# multiplication with optional transpositions.  The standard
# :class:`onnx.reference.ReferenceEvaluator` does not know about it, but
# :class:`ExtendedReferenceEvaluator` does.

model_fmm = oh.make_model(
    oh.make_graph(
        [oh.make_node("FusedMatMul", ["X", "Y"], ["Z"], domain="com.microsoft")],
        "fused_matmul_graph",
        [
            oh.make_tensor_value_info("X", TFLOAT, None),
            oh.make_tensor_value_info("Y", TFLOAT, None),
        ],
        [oh.make_tensor_value_info("Z", TFLOAT, None)],
    ),
    opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
)

a = np.arange(4, dtype=np.float32).reshape(2, 2)
ref_fmm = ExtendedReferenceEvaluator(model_fmm)
(z,) = ref_fmm.run(None, {"X": a, "Y": a})
print("FusedMatMul result:\n", z)
assert np.allclose(z, a @ a)

# %%
# With ``transA=1`` the first operand is transposed before the multiplication.

model_fmm_t = oh.make_model(
    oh.make_graph(
        [oh.make_node("FusedMatMul", ["X", "Y"], ["Z"], domain="com.microsoft", transA=1)],
        "fused_matmul_transA_graph",
        [
            oh.make_tensor_value_info("X", TFLOAT, None),
            oh.make_tensor_value_info("Y", TFLOAT, None),
        ],
        [oh.make_tensor_value_info("Z", TFLOAT, None)],
    ),
    opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
)

ref_fmm_t = ExtendedReferenceEvaluator(model_fmm_t)
(z_t,) = ref_fmm_t.run(None, {"X": a, "Y": a})
print("FusedMatMul(transA=1) result:\n", z_t)
assert np.allclose(z_t, a.T @ a)

# %%
# 3. QuickGelu (com.microsoft contrib operator)
# ----------------------------------------------
#
# ``QuickGelu`` applies the gated sigmoid activation
# ``x * sigmoid(alpha * x)`` element-wise.

model_gelu = oh.make_model(
    oh.make_graph(
        [oh.make_node("QuickGelu", ["X"], ["Z"], domain="com.microsoft", alpha=1.702)],
        "quick_gelu_graph",
        [oh.make_tensor_value_info("X", TFLOAT, None)],
        [oh.make_tensor_value_info("Z", TFLOAT, None)],
    ),
    opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
)

x_gelu = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
ref_gelu = ExtendedReferenceEvaluator(model_gelu)
(z_gelu,) = ref_gelu.run(None, {"X": x_gelu})
print("QuickGelu result:", z_gelu)


# %%
# 4. Adding a custom operator via ``new_ops``
# -------------------------------------------
#
# Any :class:`OpRun <onnx.reference.op_run.OpRun>` subclass can be passed
# through the ``new_ops`` argument.  The built-in :attr:`default_ops
# <yobx.reference.evaluator.ExtendedReferenceEvaluator.default_ops>` are
# always merged in automatically, so you only need to list your additions.

from onnx.reference.op_run import OpRun  # noqa: E402


class Scale(OpRun):
    """Multiplies every element of X by a constant *factor*."""

    op_domain = "my.domain"

    def _run(self, X, factor=2.0):  # type: ignore[override]
        return (X * np.float32(factor),)


model_custom = oh.make_model(
    oh.make_graph(
        [oh.make_node("Scale", ["X"], ["Z"], domain="my.domain", factor=3.0)],
        "scale_graph",
        [oh.make_tensor_value_info("X", TFLOAT, [None])],
        [oh.make_tensor_value_info("Z", TFLOAT, [None])],
    ),
    opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("my.domain", 1)],
    ir_version=10,
)

x_s = np.array([1.0, 2.0, 3.0], dtype=np.float32)
ref_custom = ExtendedReferenceEvaluator(model_custom, new_ops=[Scale])
(z_s,) = ref_custom.run(None, {"X": x_s})
print("Scale(factor=3) result:", z_s)
assert np.allclose(z_s, x_s * 3.0)

# %%
# 5. Listing the default operators
# ---------------------------------
#
# :attr:`default_ops` shows all operator implementations that are
# registered automatically.

import pprint  # noqa: E402

pprint.pprint(ExtendedReferenceEvaluator.default_ops)
