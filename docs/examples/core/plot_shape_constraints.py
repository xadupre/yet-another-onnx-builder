"""
.. _l-plot-shape-constraints:

Constraint Mechanism in Shape Inference
========================================

When :class:`BasicShapeBuilder <yobx.xshape.shape_builder_impl.BasicShapeBuilder>`
encounters a broadcasting operation between a tensor with a **symbolic** (unknown)
dimension and a tensor with a **concrete** (known) dimension, it registers a
*constraint* that records the equality between the two.

This avoids the need to backtrack through already-processed nodes to update
their output shapes: the concrete value is used immediately as the broadcast
result, and the constraint makes the relationship available for subsequent
validation and dimension renaming.

Constraint registration rules
------------------------------

The constraint is registered by
:func:`broadcast_shape <yobx.xshape.shape_type_compute.broadcast_shape>`
whenever a symbolic string dimension is paired with a non-1 integer:

+--------------------+------------------+--------+------------------------------+
| Dimension A        | Dimension B      | Result | Constraint registered         |
+====================+==================+========+==============================+
| ``"d_model"`` (str)| ``64`` (int)     | ``64`` | ``"d_model" = 64``            |
+--------------------+------------------+--------+------------------------------+
| ``64`` (int)       | ``"d_model"`` (str)| ``64``| ``"d_model" = 64``            |
+--------------------+------------------+--------+------------------------------+
| ``"seq"`` (str)    | ``1`` (int)      | ``"seq"``| *(none — 1 broadcasts freely)*|
+--------------------+------------------+--------+------------------------------+
| ``"a"`` (str)      | ``"b"`` (str)    | ``"a^b"``| *(none — both symbolic)*   |
+--------------------+------------------+--------+------------------------------+

The concrete integer always wins in the output shape.  The constraint records the
symbolic name so it can be recovered later (e.g., for dimension renaming or
downstream equality checks).

See :ref:`l-design-shape` for a detailed description of the full
:class:`ShapeBuilder <yobx.xshape.ShapeBuilder>` design.

See also
--------

* :class:`yobx.xshape.BasicShapeBuilder` — main implementation
* :func:`yobx.xshape.shape_type_compute.broadcast_shape` — broadcast helper
* :meth:`yobx.xshape.ShapeBuilder.register_constraint_dimension` — low-level API
* :meth:`yobx.xshape.ShapeBuilder.get_registered_constraints` — inspect constraints
"""

# %%
# Broadcasting a symbolic dimension against a concrete one
# ---------------------------------------------------------
#
# The simplest scenario: add a bias vector of size 64 to a tensor whose last
# dimension ``d_model`` is unknown at graph-construction time.
#
# When :func:`broadcast_shape <yobx.xshape.shape_type_compute.broadcast_shape>`
# aligns the two shapes it finds ``"d_model"`` (symbolic) against ``64``
# (concrete).  The concrete value ``64`` is chosen as the output dimension
# and the constraint ``d_model = 64`` is registered.

import onnx
import numpy as np
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.xshape import BasicShapeBuilder
from yobx.xshape.shape_type_compute import broadcast_shape

TFLOAT = onnx.TensorProto.FLOAT

# Direct call to broadcast_shape
builder_direct = BasicShapeBuilder()
result = broadcast_shape(("batch", "seq", "d_model"), (64,), graph_builder=builder_direct)

print("Input shapes : ('batch', 'seq', 'd_model')  and  (64,)")
print("Broadcast result :", result)
print("Registered constraints:", builder_direct.get_registered_constraints())

# %%
# Full model: Add with a constant bias
# --------------------------------------
#
# Embedding the same operation inside an ONNX model and running
# :class:`BasicShapeBuilder <yobx.xshape.shape_builder_impl.BasicShapeBuilder>`
# produces the same result: the output shape resolves ``d_model`` to the
# concrete ``64``, and the constraint is stored for later use.

model = oh.make_model(
    oh.make_graph(
        [oh.make_node("Add", ["X", "bias"], ["Z"])],
        "graph",
        [oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", "d_model"])],
        [oh.make_tensor_value_info("Z", TFLOAT, [None, None, None])],
        [onh.from_array(np.zeros((64,), dtype=np.float32), name="bias")],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=10,
)

builder = BasicShapeBuilder()
builder.run_model(model)

print("\nShape of X :", builder.get_shape("X"))
print("Shape of Z :", builder.get_shape("Z"))  # ('batch', 'seq', 64) — concrete 64, not 'd_model'
print("Constraints:", builder.get_registered_constraints())  # {'d_model': {64}}

# %%
# Why does this matter?  Avoiding backtracking
# --------------------------------------------
#
# Consider the graph ``X → [Add with bias] → Z → [MatMul with W] → Out``.
#
# * Without constraints: ``Z`` might retain the symbolic shape
#   ``('batch', 'seq', 'd_model')``.  The ``MatMul`` handler would then have
#   to check compatibility between the symbolic ``d_model`` and the weight's
#   first dimension — which may not be possible without extra information.
#
# * With constraints: ``Z`` has the concrete shape ``('batch', 'seq', 64)``.
#   The ``MatMul`` handler immediately knows that ``d_model == 64`` is
#   compatible with ``W.shape[0] == 64``, and can produce a precise output
#   shape without looking back at earlier nodes.

W = np.random.randn(64, 32).astype(np.float32)

model2 = oh.make_model(
    oh.make_graph(
        [oh.make_node("Add", ["X", "bias"], ["Z"]), oh.make_node("MatMul", ["Z", "W"], ["Out"])],
        "graph",
        [oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", "d_model"])],
        [oh.make_tensor_value_info("Out", TFLOAT, [None, None, None])],
        [
            onh.from_array(np.zeros((64,), dtype=np.float32), name="bias"),
            onh.from_array(W, name="W"),
        ],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=10,
)

builder2 = BasicShapeBuilder()
builder2.run_model(model2)

for name in ["X", "Z", "Out"]:
    print(f"  {name:5s}  shape={builder2.get_shape(name)}")
print("Constraints:", builder2.get_registered_constraints())

# %%
# No constraint when the integer dimension is 1
# ----------------------------------------------
#
# A dimension of ``1`` broadcasts freely to any symbolic dimension.  In this
# case no constraint is registered: the symbolic name is carried through as-is.

builder_ones = BasicShapeBuilder()
result_ones = broadcast_shape(("batch", "seq"), (1, 1), graph_builder=builder_ones)

print("Input shapes : ('batch', 'seq')  and  (1, 1)")
print("Broadcast result :", result_ones)
print("Registered constraints:", builder_ones.get_registered_constraints())  # {}

# %%
# Inspecting constraints with ``get_registered_constraints``
# ----------------------------------------------------------
#
# :meth:`get_registered_constraints <yobx.xshape.ShapeBuilder.get_registered_constraints>`
# returns a dictionary mapping each constrained symbolic dimension name to the
# set of values (integers or other symbolic names) it has been equated to.
# Multiple constraints on the same dimension are accumulated in the set.

b = BasicShapeBuilder()
b.register_constraint_dimension("n_heads", 8)
b.register_constraint_dimension("n_heads", "h")
b.register_constraint_dimension("d_k", 64)

print("Constraints after manual registration:")
for dim, values in b.get_registered_constraints().items():
    print(f"  {dim!r} is constrained to: {values}")

# %%
# Plot: constraint registration summary
# ----------------------------------------
#
# The figure below summarises the constraint-registration rules for
# :func:`broadcast_shape <yobx.xshape.shape_type_compute.broadcast_shape>`.

import matplotlib.pyplot as plt  # noqa: E402

rows = [
    ['"d_model" (str)', '"64" (int ≠ 1)', "64", '"d_model" = 64'],
    ['"64" (int ≠ 1)', '"d_model" (str)', "64", '"d_model" = 64'],
    ['"seq" (str)', '"1" (int = 1)', '"seq"', "(none)"],
    ['"a" (str)', '"b" (str)', '"a^b"', "(none)"],
    ['"5" (int)', '"5" (int)', "5", "(none)"],
]

fig, ax = plt.subplots(figsize=(10, 2.8))
ax.axis("off")
tbl = ax.table(
    cellText=rows,
    colLabels=["Dimension A", "Dimension B", "Result", "Constraint registered"],
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.auto_set_column_width([0, 1, 2, 3])
for col in range(4):
    tbl[0, col].set_facecolor("#4c72b0")
    tbl[0, col].set_text_props(color="white", fontweight="bold")
# Highlight the rows where a constraint is registered
for row in [1, 2]:
    tbl[row, 3].set_facecolor("#d5e8d4")
ax.set_title("broadcast_shape constraint registration rules", fontsize=10, pad=6)
plt.tight_layout()
plt.show()
