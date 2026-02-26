.. _l-design-evaluator:

==========================
ExtendedReferenceEvaluator
==========================

:class:`yobx.reference.ExtendedReferenceEvaluator` extends
:class:`onnx.reference.ReferenceEvaluator` with additional operator kernels
for non-standard domains such as ``com.microsoft`` and ``ai.onnx.complex``.

The standard :class:`onnx.reference.ReferenceEvaluator` only knows about
operators defined in the ONNX standard.  ONNX Runtime ships many *contrib*
operators (domain ``com.microsoft``) that are widely used in production
models — for example ``FusedMatMul``, ``QuickGelu`` and ``Attention``.
:class:`~yobx.reference.ExtendedReferenceEvaluator` makes it possible to
run and unit-test such models with pure Python, without requiring a full
ONNX Runtime installation.

Built-in operators
==================

The following table lists the operator implementations that are registered
automatically.  They are available as :attr:`default_ops
<yobx.reference.evaluator.ExtendedReferenceEvaluator.default_ops>`.

+------------------------------+-------------------+--------------------------------------------------------------------------------------------------+
| Class                        | Domain            | Description                                                                                      |
+==============================+===================+==================================================================================================+
| ``Attention``                | com.microsoft     | Multi-head self-attention with optional mask                                                     |
+------------------------------+-------------------+--------------------------------------------------------------------------------------------------+
| ``BiasSoftmax``              | com.microsoft     | Softmax with an additive bias term                                                               |
+------------------------------+-------------------+--------------------------------------------------------------------------------------------------+
| ``ComplexModule``            | ai.onnx.complex   | Element-wise modulus of a complex tensor                                                         |
+------------------------------+-------------------+--------------------------------------------------------------------------------------------------+
| ``FusedMatMul``              | com.microsoft     | Matrix multiplication with optional transpositions (``transA``/``transB``) and ``alpha`` scaling |
+------------------------------+-------------------+--------------------------------------------------------------------------------------------------+
| ``MemcpyFromHost``           | (default)         | Identity copy (device ↔ host no-op)                                                              |
+------------------------------+-------------------+--------------------------------------------------------------------------------------------------+
| ``MemcpyToHost``             | (default)         | Identity copy (device ↔ host no-op)                                                              |
+------------------------------+-------------------+--------------------------------------------------------------------------------------------------+
| ``QLinearAveragePool``       | com.microsoft     | Quantized average pooling                                                                        |
+------------------------------+-------------------+--------------------------------------------------------------------------------------------------+
| ``QLinearConv``              | com.microsoft     | Quantized 2-D convolution                                                                        |
+------------------------------+-------------------+--------------------------------------------------------------------------------------------------+
| ``QuickGelu``                | com.microsoft     | Gated sigmoid activation ``x·σ(α·x)``                                                            |
+------------------------------+-------------------+--------------------------------------------------------------------------------------------------+
| ``SkipLayerNormalization``   | com.microsoft     | Residual add followed by layer normalisation                                                     |
+------------------------------+-------------------+--------------------------------------------------------------------------------------------------+
| ``ToComplex``                | ai.onnx.complex   | Converts a real tensor ``(..., 2)`` to complex                                                   |
+------------------------------+-------------------+--------------------------------------------------------------------------------------------------+


The full list at runtime can be printed with:

.. runpython::
    :showcode:

    import pprint
    from yobx.reference import ExtendedReferenceEvaluator

    pprint.pprint(ExtendedReferenceEvaluator.default_ops)

Basic usage
===========

:class:`~yobx.reference.ExtendedReferenceEvaluator` is a drop-in replacement
for :class:`onnx.reference.ReferenceEvaluator`.  Any model that runs with the
standard evaluator also runs here.

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    import onnx.helper as oh
    from yobx.reference import ExtendedReferenceEvaluator

    TFLOAT = onnx.TensorProto.FLOAT
    model = oh.make_model(
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
    ref = ExtendedReferenceEvaluator(model)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    (result,) = ref.run(None, {"X": x, "Y": x})
    print(result)

Contrib operators
=================

Models that use ONNX Runtime contrib operators can be run directly.
The example below uses ``FusedMatMul`` — a ``com.microsoft`` operator that
fuses matrix multiplication with optional transposition of either operand.

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    import onnx.helper as oh
    from yobx.reference import ExtendedReferenceEvaluator

    TFLOAT = onnx.TensorProto.FLOAT
    model = oh.make_model(
        oh.make_graph(
            [oh.make_node("FusedMatMul", ["X", "Y"], ["Z"], domain="com.microsoft", transA=1)],
            "fused_mm",
            [
                oh.make_tensor_value_info("X", TFLOAT, None),
                oh.make_tensor_value_info("Y", TFLOAT, None),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, None)],
        ),
        opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )
    ref = ExtendedReferenceEvaluator(model)
    a = np.arange(4, dtype=np.float32).reshape(2, 2)
    (result,) = ref.run(None, {"X": a, "Y": a})
    print(result)  # a.T @ a

Adding custom operators
=======================

Pass extra :class:`OpRun <onnx.reference.op_run.OpRun>` subclasses through
the ``new_ops`` argument.  They are *merged* with :attr:`default_ops`; you do
not need to re-list the built-in contrib operators.

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    import onnx.helper as oh
    from onnx.reference.op_run import OpRun
    from yobx.reference import ExtendedReferenceEvaluator

    TFLOAT = onnx.TensorProto.FLOAT

    class MyCustomOp(OpRun):
        op_domain = "my.domain"

        def _run(self, X):
            return (X * 2,)

    model = oh.make_model(
        oh.make_graph(
            [oh.make_node("MyCustomOp", ["X"], ["Z"], domain="my.domain")],
            "custom_graph",
            [oh.make_tensor_value_info("X", TFLOAT, [None])],
            [oh.make_tensor_value_info("Z", TFLOAT, [None])],
        ),
        opset_imports=[oh.make_opsetid("", 18), oh.make_opsetid("my.domain", 1)],
        ir_version=10,
    )
    ref = ExtendedReferenceEvaluator(model, new_ops=[MyCustomOp])
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    (result,) = ref.run(None, {"X": x})
    print(result)  # [2. 4. 6.]

Inspecting intermediate results
================================

Pass ``verbose=10`` to :class:`~yobx.reference.ExtendedReferenceEvaluator`
to print every input, every intermediate result, and every output as the
model executes.  This is useful for debugging incorrect outputs or
understanding how values flow through the graph.

The ``verbose`` parameter maps to the logging levels used internally by
:class:`onnx.reference.ReferenceEvaluator`:

* ``verbose=0`` (default) — silent
* ``verbose=2`` — prints each node as it executes (``NodeOp(inputs) -> outputs``)
* ``verbose=3`` or higher — also prints the value of every input, initializer
  constant (``+C``), and intermediate/final result (``+``, ``+I``)

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    import onnx.helper as oh
    from yobx.reference import ExtendedReferenceEvaluator

    TFLOAT = onnx.TensorProto.FLOAT
    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Add", ["X", "Y"], ["T"]),
                oh.make_node("Relu", ["T"], ["Z"]),
            ],
            "add_relu",
            [
                oh.make_tensor_value_info("X", TFLOAT, [None, None]),
                oh.make_tensor_value_info("Y", TFLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, [None, None])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )
    ref = ExtendedReferenceEvaluator(model, verbose=10)
    x = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    (result,) = ref.run(None, {"X": x, "Y": x})
    print("result:", result)

The lines prefixed with ``+I`` are model inputs; lines with ``+C`` are
initializer constants; and lines with ``+`` (after a node execution line)
are the intermediate or final outputs produced by that node.

Operator versioning
===================

When a model imports multiple versions of a domain (e.g. opset 13 and 17),
:meth:`filter_ops <yobx.reference.evaluator.ExtendedReferenceEvaluator.filter_ops>`
selects the *best* (highest version that does not exceed the model opset)
implementation from the ``new_ops`` list.

This mirrors the versioning convention used by
:class:`onnx.reference.ReferenceEvaluator` itself: operator classes whose
names end in ``_<version>`` (e.g. ``MyOp_13``, ``MyOp_17``) are treated as
versioned alternatives and the most appropriate one is chosen automatically.

.. seealso::

    :ref:`l-plot-extended-reference-evaluator` — sphinx-gallery example
    demonstrating standard operators, ``FusedMatMul``, ``QuickGelu``, and
    custom operator injection.
