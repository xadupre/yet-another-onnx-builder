
.. _l-design-cost:

==========================
Computation Cost Inference
==========================

Overview
========

Knowing the **computational cost** of an ONNX model — expressed as a count of
*floating-point operations* (FLOPs) — is useful for model comparison, profiling,
and guiding optimization decisions.

:mod:`yobx.xshape.cost_inference` implements a lightweight per-operator FLOPs
estimator.  The main entry point is :func:`~yobx.xshape.estimate_node_flops`,
which accepts a single ONNX node together with two callables that resolve tensor
shapes and integer literals, and returns the estimated FLOPs count.

When input shapes contain **symbolic dimensions** (strings such as ``"batch"`` or
``"seq"``), the returned value is a symbolic arithmetic expression (also a string)
that can be evaluated once concrete shapes are known.  Static shapes yield plain
integer counts.

Integration with BasicShapeBuilder
====================================

:class:`~yobx.xshape.BasicShapeBuilder` integrates cost inference through the
:meth:`~yobx.xshape.shape_builder_impl.BasicShapeBuilder.run_model` method.
Pass ``inference=InferenceMode.COST`` to enable it:

.. code-block:: python

    from yobx.xshape import BasicShapeBuilder, InferenceMode

    builder = BasicShapeBuilder()
    cost_list = builder.run_model(model, inference=InferenceMode.COST)
    # cost_list: list of (op_type, flops, node) tuples

Each element of *cost_list* is a ``(op_type, flops, node)`` triple where *flops*
is either an integer, a symbolic string expression, or ``None`` (unsupported op or
unknown shapes).

To substitute concrete dimension values and obtain integer FLOPs counts, call
:meth:`~yobx.xshape.shape_builder_impl.BasicShapeBuilder.evaluate_cost_with_true_inputs`
with the actual input tensors:

.. code-block:: python

    import numpy as np

    feeds = {"X": np.random.randn(2, 64, 64).astype("float32")}
    concrete = builder.evaluate_cost_with_true_inputs(feeds, cost_list)
    total = sum(f or 0 for _, f, _ in concrete)
    print(f"Total FLOPs: {total:,}")

For a full worked example including a before/after optimization comparison see
:ref:`l-plot-symbolic-cost`.

How FLOPs Are Counted
======================

The estimator assigns FLOPs to operators using a set of simple, well-established
counting conventions.  Operators are partitioned into groups, each governed by a
uniform formula:

**Element-wise unary operators** (``Relu``, ``Sigmoid``, ``Exp``, ``Sqrt``, …)
    1 FLOPs per output element.  For ``Sigmoid`` specifically the formula accounts
    for the ``exp + add + div`` decomposition: **3 FLOPs per element**.
    For ``Softmax`` / ``LogSoftmax``: **3 FLOPs per element**.

**Element-wise binary operators** (``Add``, ``Mul``, ``Sub``, ``Div``, …)
    1 FLOPs per output element.

**Matrix multiplication (MatMul)**
    For inputs of shape ``(..., M, K)`` and ``(..., K, N)`` the formula is:

    .. math::

        \text{FLOPs} = 2 \times \prod(\text{batch dims}) \times M \times K \times N

    The factor of 2 accounts for one multiply-accumulate per inner-product step.

**General matrix multiply (Gemm)**
    For an ``alpha * A @ B + beta * C`` operation with shapes ``(M, K)`` and
    ``(K, N)``:

    .. math::

        \text{FLOPs} = 2 \times M \times K \times N + M \times N

    The additional ``M*N`` term models the bias addition.

**Convolution (Conv / ConvTranspose)**
    For output shape ``(N, C_out, *spatial_out)`` and weight shape
    ``(C_out, C_in_per_group, *kernel)``:

    .. math::

        \text{FLOPs} = 2 \times N \times C_{out} \times C_{in/group}
                       \times \prod(\text{kernel}) \times \prod(\text{spatial\_out})

**Windowed pooling (MaxPool / AveragePool)**

    .. math::

        \text{FLOPs} = N \times C \times \prod(\text{spatial\_out})
                       \times \prod(\text{kernel\_shape})

**Global pooling (GlobalAveragePool / GlobalMaxPool)**

    .. math::

        \text{FLOPs} = N \times C \times \prod(\text{spatial dims of input})

**Normalization (BatchNormalization)**
    mean + var + normalise ≈ **2 FLOPs per output element**.

**Normalization (LayerNormalization / GroupNormalization / InstanceNormalization)**
    mean + var + sub + div + scale + bias ≈ **6 FLOPs per output element**.

**Reduction operators** (``ReduceSum``, ``ReduceMean``, ``ReduceMax``, …)
    1 FLOPs per *input* element (one comparison or accumulation step).

**Recurrent cells**

    * **LSTM** — ``2 * seq * batch * (input_size + hidden) * 4 * hidden``
    * **GRU**  — ``2 * seq * batch * (input_size + hidden) * 3 * hidden``
    * **RNN**  — ``2 * seq * batch * (input_size + hidden) * hidden``

**Data-movement operators** (``Cast``, ``Transpose``, ``Gather``, ``Pad``, …)
    Element count of the first output tensor (one read + one write per element).

**Shape-manipulation operators** (``Reshape``, ``Squeeze``, ``Unsqueeze``)
    Rank of the output tensor (one metadata operation per dimension).
    ``Shape`` uses the rank of the *input* tensor.

**Zero-cost operators** (``Identity``)
    0 FLOPs — these are purely logical copies that a compiler can eliminate.

Operators not covered by any of the above categories return ``None``.

Programmatic Formula Listing
==============================

The function :func:`~yobx.xshape.list_op_cost_formulas` returns a sorted
dictionary mapping every supported ``op_type`` to the **symbolic FLOPs
expression** that the estimator produces on a representative ONNX backend test
example.  All static input dimensions of that example are replaced by symbolic
variables (``DIM<n>``) before running the estimator, so the result shows the
general formula rather than a single concrete number.

The complete table of supported operators and their symbolic FLOPs formulas is
generated below by calling :func:`~yobx.xshape.list_op_cost_formulas` at
documentation-build time:

.. runpython::
    :showcode:
    :rst:

    from yobx.xshape import list_op_cost_formulas

    formulas = list_op_cost_formulas()

    rows = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 30 70",
        "",
        "   * - Op type",
        "     - Symbolic FLOPs",
    ]
    for op_type, formula in formulas.items():
        rows.append(f"   * - ``{op_type}``")
        rows.append(f"     - ``{formula}``")

    print("\n".join(rows))

See also the gallery example :ref:`l-plot-cost-formulas` for additional
context and worked examples.
