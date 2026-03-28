"""
.. _l-plot-computed-shapes:

Computed Shapes: Add + Concat + Reshape
========================================

This example shows how :class:`BasicShapeBuilder
<yobx.xshape.shape_builder_impl.BasicShapeBuilder>` tracks symbolic dimension
expressions through a sequence of ``Add``, ``Concat``, and ``Reshape`` nodes,
and compares the result with the standard
:func:`onnx.shape_inference.infer_shapes` and the
`onnx-shape-inference <https://pypi.org/project/onnx-shape-inference/>`_
package from PyPI.

The key difference is that ``onnx.shape_inference.infer_shapes`` can only
propagate shapes when dimensions are statically known integers.  When the
model contains dynamic (symbolic) dimensions it typically assigns ``None``
(unknown) to most intermediate results. :class:`BasicShapeBuilder` instead
keeps the dimensions as symbolic arithmetic expressions so that output shapes
are expressed in terms of the input dimension names.

The ``onnx-shape-inference`` package also performs symbolic shape inference
using `SymPy <https://www.sympy.org>`_ to track dimension expressions across
nodes.  It operates on the :pypi:`onnx-ir` representation of the model.

See :ref:`l-design-shape` for a detailed description of how
:class:`BasicShapeBuilder <yobx.xshape.shape_builder_impl.BasicShapeBuilder>`
works and a comparison table with :func:`onnx.shape_inference.infer_shapes`.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas
import onnx
import onnx_ir as ir
import onnxruntime
import onnx.helper as oh
import onnx.numpy_helper as onh
from onnx_shape_inference import infer_symbolic_shapes
from yobx.xshape import BasicShapeBuilder

TFLOAT = onnx.TensorProto.FLOAT


# %%
# Helper functions for shape inference
# --------------------------------------
#
# The three shape-inference approaches used throughout this example are each
# wrapped in a small helper so the same logic can be reused without repetition.


def infer_shapes_onnx(model: onnx.ModelProto) -> dict:
    """Run :func:`onnx.shape_inference.infer_shapes`; return ``{name: shape}``."""
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for vi in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        t = vi.type.tensor_type
        if t.HasField("shape"):
            shapes[vi.name] = tuple(
                d.dim_param if d.dim_param else (d.dim_value if d.dim_value else None)
                for d in t.shape.dim
            )
        else:
            shapes[vi.name] = "unknown"
    return shapes


def infer_shapes_onnx_ir(model: onnx.ModelProto) -> dict:
    """Run onnx-shape-inference :func:`infer_symbolic_shapes`; return ``{name: shape}``."""
    ir_model = ir.serde.deserialize_model(model)
    ir_model = infer_symbolic_shapes(ir_model)
    shapes = {}
    for v in ir_model.graph.inputs:
        shapes[v.name] = str(v.shape)
    for node in ir_model.graph:
        for out in node.outputs:
            shapes[out.name] = str(out.shape)
    return shapes


def infer_shapes_basic(model: onnx.ModelProto) -> BasicShapeBuilder:
    """Run :class:`BasicShapeBuilder` over *model*; return the populated builder."""
    b = BasicShapeBuilder()
    b.run_model(model)
    return b


# %%
# Build a small model
# --------------------
#
# The graph performs the following steps:
#
# 1. ``Add(X, Y)``  — element-wise addition of two tensors with shape
#    ``(batch, seq, d_model)``.
# 2. ``Concat(added, X, axis=2)``  — concatenate the result with the original
#    ``X`` along the last axis, giving shape ``(batch, seq, 2*d_model)``.
# 3. ``Reshape(concat_out, shape)``  — flatten the last two dimensions using
#    a fixed shape constant ``[0, 0, -1]``, which collapses
#    ``(batch, seq, 2*d_model)`` back to ``(batch, seq, 2*d_model)``.

model = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Add", ["X", "Y"], ["added"]),
            oh.make_node("Concat", ["added", "X"], ["concat_out"], axis=2),
            oh.make_node("Reshape", ["concat_out", "reshape_shape"], ["Z"]),
        ],
        "add_concat_reshape",
        [
            oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", "d_model"]),
            oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", "d_model"]),
        ],
        [oh.make_tensor_value_info("Z", TFLOAT, [None, None, None])],
        [onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="reshape_shape")],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=10,
)

# %%
# Shape inference with ONNX
# --------------------------
#
# ``onnx.shape_inference.infer_shapes`` propagates shapes through the model.
# For dynamic dimensions the inferred shapes for intermediate results are
# often unknown (``None``).

print("=== onnx.shape_inference.infer_shapes ===")
for name, shape in infer_shapes_onnx(model).items():
    print(f"  {name:15s}  shape={shape}")

# %%
# Shape inference with onnx-shape-inference (PyPI)
# --------------------------------------------------
#
# The `onnx-shape-inference <https://pypi.org/project/onnx-shape-inference/>`_
# package offers a second symbolic approach.  It works on the
# :class:`onnx_ir.Model` representation and uses SymPy to track dimension
# expressions.  Install it with ``pip install onnx-shape-inference``.
#
# Compared with ``onnx.shape_inference.infer_shapes``, it successfully
# resolves the ``Concat`` output to ``(batch, seq, 2*d_model)``.  The
# ``Reshape`` output receives a freshly-generated symbol (``_d0``) because
# the ``[0, 0, -1]`` constant shape tensor is not yet fully evaluated by this
# library.

onnx_ir_shapes = infer_shapes_onnx_ir(model)

print("=== onnx-shape-inference (infer_symbolic_shapes) ===")
for name in ["X", "Y", "added", "concat_out", "Z"]:
    print(f"  {name:15s}  shape={onnx_ir_shapes.get(name, 'unknown')}")

# %%
# Shape inference with BasicShapeBuilder
# ----------------------------------------
#
# :class:`BasicShapeBuilder <yobx.xshape.shape_builder_impl.BasicShapeBuilder>`
# keeps the shapes as symbolic expressions.  Because ``reshape_shape`` is a
# constant ``[0, 0, -1]``, the builder can evaluate the ``Reshape`` and express
# the output shape as a function of the input dimensions.


builder = infer_shapes_basic(model)

print("\n=== BasicShapeBuilder ===")
for name in ["X", "Y", "added", "concat_out", "Z"]:
    print(f"  {name:15s}  shape={builder.get_shape(name)}")

# %%
# Evaluate symbolic shapes with concrete values
# -----------------------------------------------
#
# Once the concrete values of the dynamic dimensions are known,
# :meth:`evaluate_shape <yobx.xshape.ShapeBuilder.evaluate_shape>` resolves
# each symbolic expression to its actual integer value.

context = dict(batch=2, seq=5, d_model=8)
for name in ["X", "Y", "added", "concat_out", "Z"]:
    concrete = builder.evaluate_shape(name, context)
    print(f"  {name:15s}  concrete shape={concrete}")

# %%
# Verify with real data
# ----------------------
#
# Finally, run the model with concrete numpy arrays and confirm that the
# shapes predicted by :class:`BasicShapeBuilder` match the actual output
# shapes.

feeds = {
    "X": np.random.rand(2, 5, 8).astype(np.float32),
    "Y": np.random.rand(2, 5, 8).astype(np.float32),
}

session = onnxruntime.InferenceSession(
    model.SerializeToString(), providers=["CPUExecutionProvider"]
)
outputs = session.run(None, feeds)
result = builder.compare_with_true_inputs(feeds, outputs)
print("\n=== shape comparison ===")
data = []
for name, dims in result.items():
    obs = dict(result=name)
    for i, dim in enumerate(dims):
        for c, v in zip(["expression", "expected", "computed"], dim):
            data.append(dict(result=name, dimension=i, col=c, value=v))
print(pandas.DataFrame(data).pivot(index=["result", "dimension"], columns="col", values="value"))

# %%
# Impact of named output shapes on constraints (NonZero)
# -------------------------------------------------------
#
# Some operators—such as ``NonZero``—introduce a *fresh* symbolic dimension for
# their output because the number of results depends on the *values* of the
# input tensor, not merely its shape.  :class:`BasicShapeBuilder` assigns an
# internal name like ``NEWDIM_nonzero_0`` to that dimension.
#
# When the graph output is declared **without** named dimensions (all ``None``),
# the internal name is kept as-is and no constraint is registered.
#
# When the graph output is declared **with** named dimensions (e.g.
# ``["rank", "nnz"]``), :meth:`run_value_info
# <yobx.xshape.shape_builder_impl.BasicShapeBuilder.run_value_info>` detects
# the mismatch between the computed internal name ``NEWDIM_nonzero_0`` and the
# user-supplied name ``nnz``, and registers the constraint
# ``NEWDIM_nonzero_0 = nnz``.  The dimension naming step then renames the
# internal token to the user-visible name across all shapes.

TINT64 = onnx.TensorProto.INT64

# %%
# Without named output shape
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The output of ``NonZero`` has shape ``(rank_of_input, num_nonzero_elements)``.
# With no dimension names declared on the graph output, the second dimension
# receives the internal placeholder ``NEWDIM_nonzero_0`` and no constraint is
# stored.
#
# The graph surrounds ``NonZero`` with additional operators to form a realistic
# seven-node pipeline — four pre-processing operators feed into ``NonZero`` and
# two post-processing operators consume its output:
#
# 1. ``Abs``       — take the absolute value of the input
# 2. ``Relu``      — keep positive elements, zero out others
# 3. ``Add``       — double the positive values
# 4. ``Mul``       — square the doubled values
# 5. ``NonZero``   — collect indices of non-zero elements (output: ``nz``)
# 6. ``Transpose`` — swap axes of the index matrix (``[rank, nnz]`` → ``[nnz, rank]``)
# 7. ``Cast``      — convert the INT64 indices to FLOAT (output: ``nz_float``)
#
# Both ``nz`` (the NonZero output) and ``nz_float`` (the final FLOAT output) are
# exposed as graph outputs so that shape constraints can be declared on ``nz``.

nz_model_anon = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Abs", ["X"], ["abs_out"]),
            oh.make_node("Relu", ["abs_out"], ["relu_out"]),
            oh.make_node("Add", ["relu_out", "relu_out"], ["double_out"]),
            oh.make_node("Mul", ["double_out", "relu_out"], ["mul_out"]),
            oh.make_node("NonZero", ["mul_out"], ["nz"]),
            oh.make_node("Transpose", ["nz"], ["transposed_nz"]),
            oh.make_node("Cast", ["transposed_nz"], ["nz_float"], to=TFLOAT),
        ],
        "nonzero_anon",
        [oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq"])],
        [
            oh.make_tensor_value_info("nz", TINT64, [None, None]),
            oh.make_tensor_value_info("nz_float", TFLOAT, [None, None]),
        ],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=10,
)

builder_anon = infer_shapes_basic(nz_model_anon)

print("=== NonZero — anonymous output shape ===")
print(f"  {'nz':10s}  shape={builder_anon.get_shape('nz')}")
print(f"  {'nz_float':10s}  shape={builder_anon.get_shape('nz_float')}")
print(f"  Constraints: {builder_anon.get_registered_constraints()}")

# %%
# With named output shape
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# Declaring the output shape as ``["rank", "nnz"]`` causes
# :meth:`run_value_info` to register the constraint
# ``NEWDIM_nonzero_0 = nnz``.  The naming step then replaces the internal
# token throughout all tracked shapes so the user sees ``nnz`` everywhere.

nz_model_named = oh.make_model(
    oh.make_graph(
        [
            oh.make_node("Abs", ["X"], ["abs_out"]),
            oh.make_node("Relu", ["abs_out"], ["relu_out"]),
            oh.make_node("Add", ["relu_out", "relu_out"], ["double_out"]),
            oh.make_node("Mul", ["double_out", "relu_out"], ["mul_out"]),
            oh.make_node("NonZero", ["mul_out"], ["nz"]),
            oh.make_node("Transpose", ["nz"], ["transposed_nz"]),
            oh.make_node("Cast", ["transposed_nz"], ["nz_float"], to=TFLOAT),
        ],
        "nonzero_named",
        [oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq"])],
        [
            oh.make_tensor_value_info("nz", TINT64, ["rank", "nnz"]),
            oh.make_tensor_value_info("nz_float", TFLOAT, [None, None]),
        ],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=10,
)

builder_named = infer_shapes_basic(nz_model_named)

print("\n=== NonZero — named output shape ===")
print(f"  {'nz':10s}  shape={builder_named.get_shape('nz')}")
print(f"  {'nz_float':10s}  shape={builder_named.get_shape('nz_float')}")
print(f"  Constraints: {builder_named.get_registered_constraints()}")

# %%
# Plot: named vs anonymous output shape for NonZero
# --------------------------------------------------
#
# The table below contrasts what each shape-inference tool reports for the
# ``NonZero`` output under both configurations.

nz_onnx_shapes = {
    "anon": str(infer_shapes_onnx(nz_model_anon).get("nz", "—")),
    "named": str(infer_shapes_onnx(nz_model_named).get("nz", "—")),
}

# onnx-shape-inference shapes (for the 'nz' node output specifically)
nz_ir_shapes = {
    "anon": infer_shapes_onnx_ir(nz_model_anon).get("nz", "—"),
    "named": infer_shapes_onnx_ir(nz_model_named).get("nz", "—"),
}

# BasicShapeBuilder shapes
nz_builder_shapes = {
    "anon": str(builder_anon.get_shape("nz")),
    "named": str(builder_named.get_shape("nz")),
}
nz_constraints = {
    "anon": str(builder_anon.get_registered_constraints()),
    "named": str(builder_named.get_registered_constraints()),
}

col_labels_nz = [
    "output declaration",
    "onnx.shape_inference",
    "onnx-shape-inference",
    "BasicShapeBuilder",
    "constraints",
]
table_data_nz = [
    [
        "[None, None]",
        nz_onnx_shapes.get("anon", "—"),
        nz_ir_shapes.get("anon", "—"),
        nz_builder_shapes["anon"],
        nz_constraints["anon"],
    ],
    [
        '["rank", "nnz"]',
        nz_onnx_shapes.get("named", "—"),
        nz_ir_shapes.get("named", "—"),
        nz_builder_shapes["named"],
        nz_constraints["named"],
    ],
]

fig, ax = plt.subplots(figsize=(14, 2.0))
ax.axis("off")
tbl_nz = ax.table(cellText=table_data_nz, colLabels=col_labels_nz, loc="center", cellLoc="center")
tbl_nz.auto_set_font_size(False)
tbl_nz.set_fontsize(8)
tbl_nz.auto_set_column_width(list(range(len(col_labels_nz))))
for col in range(len(col_labels_nz)):
    tbl_nz[0, col].set_facecolor("#4c72b0")
    tbl_nz[0, col].set_text_props(color="white", fontweight="bold")
# Highlight the row where the constraint is registered
for col in range(len(col_labels_nz)):
    tbl_nz[2, col].set_facecolor("#d5e8d4")
ax.set_title("NonZero: impact of named output shapes on constraints", fontsize=10, pad=8)
plt.tight_layout()
plt.show()
