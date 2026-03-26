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
import onnx
import onnxruntime
import onnx.helper as oh
import onnx.numpy_helper as onh
from yobx.xshape import BasicShapeBuilder

TFLOAT = onnx.TensorProto.FLOAT


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

inferred = onnx.shape_inference.infer_shapes(model)

print("=== onnx.shape_inference.infer_shapes ===")
for vi in (
    list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
):
    t = vi.type.tensor_type
    if t.HasField("shape"):
        shape = tuple(
            d.dim_param if d.dim_param else (d.dim_value if d.dim_value else None)
            for d in t.shape.dim
        )
    else:
        shape = "unknown"
    print(f"  {vi.name:15s}  shape={shape}")

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

try:
    import onnx_ir as ir
    from onnx_shape_inference import infer_symbolic_shapes

    ir_model = ir.serde.deserialize_model(model)
    ir_model = infer_symbolic_shapes(ir_model)

    # Build a name → shape mapping for all values in the graph.
    onnx_ir_shapes: dict[str, str] = {}
    for v in ir_model.graph.inputs:
        onnx_ir_shapes[v.name] = str(v.shape)
    for node in ir_model.graph:
        for out in node.outputs:
            onnx_ir_shapes[out.name] = str(out.shape)

    print("=== onnx-shape-inference (infer_symbolic_shapes) ===")
    for name in ["X", "Y", "added", "concat_out", "Z"]:
        print(f"  {name:15s}  shape={onnx_ir_shapes.get(name, 'unknown')}")
except ImportError:
    onnx_ir_shapes = None
    print("onnx-shape-inference is not installed; skipping this section.")

# %%
# Shape inference with BasicShapeBuilder
# ----------------------------------------
#
# :class:`BasicShapeBuilder <yobx.xshape.shape_builder_impl.BasicShapeBuilder>`
# keeps the shapes as symbolic expressions.  Because ``reshape_shape`` is a
# constant ``[0, 0, -1]``, the builder can evaluate the ``Reshape`` and express
# the output shape as a function of the input dimensions.


builder = BasicShapeBuilder()
builder.run_model(model)

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
print("\n=== shape comparison (expr, expected, computed) ===")
for name, dims in result.items():
    print(f"  {name}: {dims}")

# %%
# Plot: symbolic vs inferred shapes
# -----------------------------------
#
# The table below compares the shapes reported by
# ``onnx.shape_inference.infer_shapes`` (which may leave dimensions as
# ``None`` for dynamic axes), the ``onnx-shape-inference`` package (which uses
# SymPy for symbolic propagation), and the symbolic expressions computed by
# :class:`BasicShapeBuilder`.

import matplotlib.pyplot as plt  # noqa: E402

tensor_names = ["X", "Y", "added", "concat_out", "Z"]

# Collect onnx.shape_inference shapes as strings
onnx_shapes = {}
for vi in (
    list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
):
    t = vi.type.tensor_type
    if t.HasField("shape"):
        shape = tuple(
            d.dim_param if d.dim_param else (d.dim_value if d.dim_value else "?")
            for d in t.shape.dim
        )
    else:
        shape = ("unknown",)
    onnx_shapes[vi.name] = str(shape)

# Collect BasicShapeBuilder shapes as strings
builder_shapes = {name: str(builder.get_shape(name)) for name in tensor_names}

if onnx_ir_shapes is not None:
    col_labels = ["tensor", "onnx.shape_inference", "onnx-shape-inference", "BasicShapeBuilder"]
    table_data = [
        [
            name,
            onnx_shapes.get(name, "—"),
            onnx_ir_shapes.get(name, "—"),
            builder_shapes.get(name, "—"),
        ]
        for name in tensor_names
    ]
    fig, ax = plt.subplots(figsize=(11, 2.5))
else:
    col_labels = ["tensor", "onnx.shape_inference", "BasicShapeBuilder"]
    table_data = [
        [name, onnx_shapes.get(name, "—"), builder_shapes.get(name, "—")]
        for name in tensor_names
    ]
    fig, ax = plt.subplots(figsize=(8, 2.5))

ax.axis("off")
tbl = ax.table(
    cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.auto_set_column_width(list(range(len(col_labels))))
# Highlight header
for col in range(len(col_labels)):
    tbl[0, col].set_facecolor("#4c72b0")
    tbl[0, col].set_text_props(color="white", fontweight="bold")
ax.set_title(
    "Shape inference: onnx vs onnx-shape-inference vs BasicShapeBuilder",
    fontsize=10,
    pad=8,
)
plt.tight_layout()
plt.show()
