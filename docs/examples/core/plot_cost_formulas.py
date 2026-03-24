"""
.. _l-plot-cost-formulas:

Computation Cost: How It Works and Supported Operator Formulas
===============================================================

This example explains how **FLOPs (floating-point operations) cost** is estimated
for ONNX models in *yobx*, and programmatically lists the formula used for every
supported operator.

The estimator is built around :func:`~yobx.xshape.estimate_node_flops` and is
exposed through :class:`~yobx.xshape.BasicShapeBuilder` via
``inference=InferenceMode.COST``.  When model inputs have *symbolic* dimensions
(strings like ``"batch"`` or ``"seq"``), the cost values are symbolic arithmetic
expressions that can be evaluated later with concrete shapes.

For a complete worked example using a real attention model, see
:ref:`l-plot-symbolic-cost`.
"""

# %%
# 1. Quick start: cost of a tiny model
# -------------------------------------
#
# We build a small two-node ONNX graph (``MatMul`` + ``Relu``) with symbolic
# input dimensions and compute its cost with
# :meth:`~yobx.xshape.shape_builder_impl.BasicShapeBuilder.run_model`.

import onnx
import onnx.helper as oh

from yobx.xshape import BasicShapeBuilder, InferenceMode

TFLOAT = onnx.TensorProto.FLOAT

model = oh.make_model(
    oh.make_graph(
        [oh.make_node("MatMul", ["A", "B"], ["C"]), oh.make_node("Relu", ["C"], ["out"])],
        "tiny",
        [
            oh.make_tensor_value_info("A", TFLOAT, ["batch", "M", "K"]),
            oh.make_tensor_value_info("B", TFLOAT, ["batch", "K", "N"]),
        ],
        [oh.make_tensor_value_info("out", TFLOAT, None)],
    ),
    opset_imports=[oh.make_opsetid("", 18)],
    ir_version=10,
)

builder = BasicShapeBuilder()
cost_list = builder.run_model(model, inference=InferenceMode.COST)

print("Symbolic FLOPs per node:")
for op_type, flops, _ in cost_list:
    print(f"  {op_type:<12s}  {flops}")


# %%
# 2. Evaluating symbolic costs with concrete input shapes
# --------------------------------------------------------
#
# Once the graph has been analysed with symbolic shapes, pass actual numpy
# arrays to
# :meth:`~yobx.xshape.shape_builder_impl.BasicShapeBuilder.evaluate_cost_with_true_inputs`
# to substitute the dimension values and get integer FLOPs counts.

import numpy as np  # noqa: E402

rng = np.random.default_rng(0)
feeds = {
    "A": rng.standard_normal((4, 32, 64)).astype(np.float32),
    "B": rng.standard_normal((4, 64, 16)).astype(np.float32),
}

concrete = builder.evaluate_cost_with_true_inputs(feeds, cost_list)

print("Concrete FLOPs per node:")
total = 0
for op_type, flops, _ in concrete:
    total += flops or 0
    print(f"  {op_type:<12s}  {flops:>12,}")
print(f"  {'TOTAL':<12s}  {total:>12,}")


# %%
# 3. How the cost estimator works
# --------------------------------
#
# Each ONNX operator type is mapped to a *handler function* in
# :mod:`yobx.xshape.cost_inference`.  The handler receives the ONNX node plus
# two callables for resolving tensor shapes and integer literals, and returns
# the FLOPs count (integer, symbolic string, or ``None`` when shapes are
# unavailable).
#
# Operators are grouped by their counting convention:
#
# ====================================  ==========================================
# Group                                 Formula
# ====================================  ==========================================
# Element-wise unary (Relu, Sqrt, …)   1 FLOPs per output element
# Element-wise binary (Add, Mul, …)    1 FLOPs per output element
# Sigmoid                               3 FLOPs per element (exp+add+div)
# Softmax / LogSoftmax                  3 FLOPs per element (exp+sum+div)
# MatMul                                2·batch·M·K·N
# Gemm                                  2·M·K·N + M·N
# Conv / ConvTranspose                  2·N·C_out·C_in/group·kernel·spatial_out
# MaxPool / AveragePool                 N·C·spatial_out·kernel_size
# GlobalAveragePool / GlobalMaxPool     N·C·spatial_in
# BatchNormalization                    2 FLOPs per output element
# LayerNorm / GroupNorm / InstanceNorm  6 FLOPs per output element
# ReduceSum / ReduceMean / … (9 ops)   Input element count
# LSTM                                  2·seq·batch·(input+hidden)·4·hidden
# GRU                                   2·seq·batch·(input+hidden)·3·hidden
# RNN                                   2·seq·batch·(input+hidden)·hidden
# Data-movement (Cast, Transpose, …)   Output element count
# Shape-manipulation (Reshape, …)      Rank of output tensor
# Identity                              0 (zero cost)
# ====================================  ==========================================
#
# The full list of supported operators (and the exact description used) is
# returned by :func:`~yobx.xshape.list_op_cost_formulas` — see section 4 below.


# %%
# 4. Programmatic listing of all supported operator formulas
# -----------------------------------------------------------
#
# :func:`~yobx.xshape.list_op_cost_formulas` returns a sorted dictionary that
# maps every registered ``op_type`` to the **symbolic FLOPs expression** obtained
# by running the cost estimator on a representative ONNX backend test example.
# All static input dimensions are first replaced by symbolic variables
# (``DIM<n>``) so that the result shows the general formula rather than a
# single concrete number.

from yobx.xshape import list_op_cost_formulas  # noqa: E402

formulas = list_op_cost_formulas()

print(f"{'Op type':<35s}  Symbolic FLOPs")
print("-" * 80)
for op_type, formula in formulas.items():
    print(f"{op_type:<35s}  {formula}")
