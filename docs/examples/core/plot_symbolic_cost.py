"""
.. _l-plot-symbolic-cost:

Symbolic Cost of a Model: Transpose Fusion
==========================================

This example shows how to compute the **symbolic FLOPs cost** of an ONNX
model and how the native ``GraphGraph`` optimizer removes a redundant
``Transpose``/``Transpose`` pair through the registered pattern name
``TransposeTranspose``.
"""

import numpy as np
from onnx_light import onnx
from onnx_light.onnx import helper
from onnx_light.onnx_core.optimization import standard_pattern_names

from yobx.builder.onnxlight import OnnxLightGraphBuilder, OnnxLightOptimizationOptions
from yobx.xshape import InferenceMode, NativeShapeInference

TFLOAT = onnx.TensorProto.FLOAT

assert "TransposeTranspose" in standard_pattern_names()


# %%
# 1. Build a tiny model with a redundant transpose pair
# -----------------------------------------------------

model = helper.make_model(
    helper.make_graph(
        [
            helper.make_node("Transpose", ["X"], ["T"], perm=[1, 0]),
            helper.make_node("Transpose", ["T"], ["Y"], perm=[1, 0]),
            helper.make_node("Relu", ["Y"], ["Z"]),
        ],
        "transpose_transpose",
        [helper.make_tensor_value_info("X", TFLOAT, ["batch", "seq"])],
        [helper.make_tensor_value_info("Z", TFLOAT, ["batch", "seq"])],
    ),
    opset_imports=[helper.make_opsetid("", 18)],
    ir_version=8,
)

print("Nodes in the original model:")
for node in model.graph.node:
    print(f"  {node.op_type:12s}  inputs={list(node.input)}  outputs={list(node.output)}")


# %%
# 2. Compute the symbolic cost before optimization
# -------------------------------------------------

builder_before = NativeShapeInference()
cost_before = builder_before.run_model(model, inference=InferenceMode.COST)

print("Symbolic FLOPs per node (before optimization):")
for op_type, flops, _ in cost_before:
    if flops:
        print(f"  {op_type:12s}  {flops}")


# %%
# 3. Evaluate the symbolic FLOPs with concrete input shapes
# ----------------------------------------------------------

batch, seq = 2, 64
rng = np.random.default_rng(42)
feeds = {"X": rng.standard_normal((batch, seq)).astype(np.float32)}
cost_concrete_before = builder_before.evaluate_cost_with_true_inputs(feeds, cost_before)

print("Concrete FLOPs per node (before optimization):")
total_before = 0
for op_type, flops, _ in cost_concrete_before:
    total_before += flops or 0
    if flops:
        print(f"  {op_type:12s}  {flops:>10,}")
print(f"  {'TOTAL':12s}  {total_before:>10,}")


# %%
# 4. Apply the native TransposeTranspose optimization
# ----------------------------------------------------

builder = OnnxLightGraphBuilder(
    model, optimization_options=OnnxLightOptimizationOptions(patterns=["TransposeTranspose"])
)
opt_artifact = builder.to_onnx(return_optimize_report=True)
opt_model = opt_artifact.proto

print("Nodes in the optimized model:")
for node in opt_model.graph.node:
    print(f"  {node.op_type:12s}  inputs={list(node.input)}  outputs={list(node.output)}")

print("Native rewrite report:", opt_artifact.report.extra)
print("Rewrites:", opt_artifact.report.stats)


# %%
# 5. Compute the symbolic cost of the optimized model
# ---------------------------------------------------

builder_after = NativeShapeInference()
cost_after = builder_after.run_model(opt_model, inference=InferenceMode.COST)

print("Symbolic FLOPs per node (after optimization):")
for op_type, flops, _ in cost_after:
    if flops:
        print(f"  {op_type:12s}  {flops}")

cost_concrete_after = builder_after.evaluate_cost_with_true_inputs(feeds, cost_after)

print("Concrete FLOPs per node (after optimization):")
total_after = 0
for op_type, flops, _ in cost_concrete_after:
    total_after += flops or 0
    if flops:
        print(f"  {op_type:12s}  {flops:>10,}")
print(f"  {'TOTAL':12s}  {total_after:>10,}")
print(
    f"\nFLOPs saved: {total_before - total_after:,}  "
    f"({(total_before - total_after) / total_before:.2%})"
)
