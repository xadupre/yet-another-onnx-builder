.. _l-design-evaluator:

==========
Evaluators
==========

``yobx`` provides three evaluators for running ONNX models in Python.
They share a common interface (``__init__(proto, ...)`` + ``run(outputs, feeds)``)
but differ in their backend, tensor type, and primary use-case:

+------------------------------------+---------------------+-----------------+-----------------------------------+
| Class                              | Backend             | Tensor type     | Best suited for                   |
+====================================+=====================+=================+===================================+
| :class:`ExtendedReferenceEvaluator`| onnx reference      | NumPy           | unit-testing, contrib ops, pure   |
|                                    | (Python)            | ``ndarray``     | Python debugging                  |
+------------------------------------+---------------------+-----------------+-----------------------------------+
| :class:`OnnxruntimeEvaluator`      | ONNX Runtime        | NumPy or        | debugging ORT execution,          |
|                                    | (node-by-node or    | PyTorch         | inspecting intermediate results,  |
|                                    | whole, ``whole=``)  |                 | or whole-model ORT inference      |
+------------------------------------+---------------------+-----------------+-----------------------------------+
| :class:`TorchReferenceEvaluator`   | PyTorch (Python)    | ``torch.Tensor``| GPU execution, memory-efficient   |
|                                    |                     |                 | evaluation, custom PyTorch ops    |
+------------------------------------+---------------------+-----------------+-----------------------------------+

.. rubric:: Quick comparison

All three evaluators accept an ``onnx.ModelProto`` (or filename) and return a
list of outputs when called via ``run(None, feed_dict)``.  The key
differences are:

* **ExtendedReferenceEvaluator** — a pure Python, NumPy-based evaluator that
  extends :class:`onnx.reference.ReferenceEvaluator` with extra kernels for
  non-standard domains (``com.microsoft``, ``ai.onnx.complex``).  No ONNX
  Runtime installation is required.  Ideal for unit tests and operator
  prototyping.

* **OnnxruntimeEvaluator** — executes each graph node individually through
  :class:`onnxruntime.InferenceSession`.  Because every node is run in
  isolation it is easy to inspect every intermediate result and to compare
  them against a reference.  Accepts both NumPy arrays and PyTorch tensors.
  Pass ``whole=True`` to skip node-by-node splitting and hand the complete
  model to a single ORT session (faster, but intermediate results are not
  accessible).

* **TorchReferenceEvaluator** — runs every node with hand-written PyTorch
  kernels.  Inputs and outputs are :class:`torch.Tensor`.  Supports CUDA
  via ``providers=["CUDAExecutionProvider"]``.  Well-suited for evaluating
  large models on the GPU where keeping activations as PyTorch tensors avoids
  expensive NumPy round-trips.


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
------------------

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
-----------

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
-----------------

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
-----------------------

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
--------------------------------

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
                oh.make_node("Tanh", ["T"], ["Z"]),
            ],
            "add_tanh",
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
-------------------

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


OnnxruntimeEvaluator
====================

:class:`yobx.reference.OnnxruntimeEvaluator` executes an ONNX model
*node by node* using :class:`onnxruntime.InferenceSession` as the kernel
backend.  Each node is wrapped in a tiny single-node ONNX model and fed
into ORT individually, which means every intermediate tensor is accessible
after the run.  This makes the class especially useful for:

* **debugging** — comparing intermediate activations between two models or
  between ORT and a reference implementation;
* **mixed-precision inspection** — examining how casts or quantisation layers
  change the values at each step;
* **portability** — the same code runs on CPU or GPU simply by passing
  different ``providers``.

Basic usage
-----------

The API mirrors :class:`onnx.reference.ReferenceEvaluator`: pass an
``onnx.ModelProto`` (or a filename) and call :meth:`run`.

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    import onnx.helper as oh
    from yobx.reference.onnxruntime_evaluator import OnnxruntimeEvaluator

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
    ref = OnnxruntimeEvaluator(model)
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    (result,) = ref.run(None, {"X": x, "Y": x})
    print(result)

Inspecting intermediate results
---------------------------------

Pass ``verbose=2`` to print every node that executes together with its
inputs and outputs.  Pass ``intermediate=True`` to :meth:`run` to get back
a dictionary that maps *every* result name (inputs, constants, and
intermediate tensors) to its value.

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    import onnx.helper as oh
    from yobx.reference.onnxruntime_evaluator import OnnxruntimeEvaluator

    TFLOAT = onnx.TensorProto.FLOAT
    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Add", ["X", "Y"], ["T"]),
                oh.make_node("Tanh", ["T"], ["Z"]),
            ],
            "add_tanh",
            [
                oh.make_tensor_value_info("X", TFLOAT, [None, None]),
                oh.make_tensor_value_info("Y", TFLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, [None, None])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )
    ref = OnnxruntimeEvaluator(model, verbose=2)
    x = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
    all_results = ref.run(None, {"X": x, "Y": x}, intermediate=True)
    for name, value in sorted(all_results.items()):
        print(f"{name}: {value}")

Running the whole model at once
--------------------------------

By default ``OnnxruntimeEvaluator`` splits the graph into individual nodes
and runs each one separately (``whole=False``).  Passing ``whole=True``
hands the complete model to a single :class:`onnxruntime.InferenceSession`
and is equivalent to calling ORT directly.  This mode is faster but does
not allow intermediate result inspection.

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    import onnx.helper as oh
    from yobx.reference.onnxruntime_evaluator import OnnxruntimeEvaluator

    TFLOAT = onnx.TensorProto.FLOAT
    model = oh.make_model(
        oh.make_graph(
            [oh.make_node("Sigmoid", ["X"], ["Z"])],
            "sigmoid_graph",
            [oh.make_tensor_value_info("X", TFLOAT, [None])],
            [oh.make_tensor_value_info("Z", TFLOAT, [None])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )
    ref = OnnxruntimeEvaluator(model, whole=True)
    x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    (result,) = ref.run(None, {"X": x})
    print(result)


TorchReferenceEvaluator
=======================

:class:`yobx.reference.TorchReferenceEvaluator` is a pure-Python evaluator
that runs every ONNX node with hand-written :mod:`torch` kernels.  Inputs
and outputs are :class:`torch.Tensor`, which means:

* there are no NumPy round-trips between nodes;
* the model can be evaluated on CUDA by passing
  ``providers=["CUDAExecutionProvider"]``;
* intermediate tensors are freed as soon as they are no longer needed,
  which reduces peak memory usage for large models.

The available kernels can be listed with
:func:`~yobx.reference.torch_evaluator.get_kernels`.

Basic usage
-----------

.. runpython::
    :showcode:

    import onnx
    import onnx.helper as oh
    import torch
    from yobx.helpers import string_type
    from yobx.reference.torch_evaluator import TorchReferenceEvaluator

    TFLOAT = onnx.TensorProto.FLOAT
    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Sigmoid", ["Y"], ["sy"]),
                oh.make_node("Mul", ["Y", "sy"], ["ysy"]),
                oh.make_node("Mul", ["X", "ysy"], ["final"]),
            ],
            "silu",
            [
                oh.make_tensor_value_info("X", TFLOAT, [1, "b", "c"]),
                oh.make_tensor_value_info("Y", TFLOAT, ["a", "b", "c"]),
            ],
            [oh.make_tensor_value_info("final", TFLOAT, ["a", "b", "c"])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=9,
    )
    sess = TorchReferenceEvaluator(model)
    feeds = dict(X=torch.rand((4, 5)), Y=torch.rand((4, 5)))
    result = sess.run(None, feeds)
    print(string_type(result, with_shape=True, with_min_max=True))

Verbose mode
------------

Pass ``verbose=1`` to print every kernel execution and every tensor freed
during the run.  This lets you trace the exact execution order and see when
memory is reclaimed.

.. runpython::
    :showcode:

    import onnx
    import onnx.helper as oh
    import torch
    from yobx.helpers import string_type
    from yobx.reference.torch_evaluator import TorchReferenceEvaluator

    TFLOAT = onnx.TensorProto.FLOAT
    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Add", ["X", "Y"], ["T"]),
                oh.make_node("Tanh", ["T"], ["Z"]),
            ],
            "add_tanh",
            [
                oh.make_tensor_value_info("X", TFLOAT, [None, None]),
                oh.make_tensor_value_info("Y", TFLOAT, [None, None]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, [None, None])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=9,
    )
    sess = TorchReferenceEvaluator(model, verbose=1)
    feeds = dict(X=torch.tensor([[1.0, -2.0], [3.0, -4.0]]),
                 Y=torch.tensor([[1.0, -2.0], [3.0, -4.0]]))
    result = sess.run(None, feeds)
    print(string_type(result, with_shape=True))

Custom kernels
--------------

A specific ONNX op can be replaced by passing a dictionary to
``custom_kernels``.  The keys are ``(domain, op_type)`` tuples and the
values are subclasses of
:class:`~yobx.reference.torch_ops.OpRunKernel`.
This is useful, for example, to delegate a single op to ONNX Runtime while
keeping the rest of the graph in PyTorch.

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    import onnx.helper as oh
    import torch
    from yobx.helpers import string_type
    from yobx.reference.torch_evaluator import TorchReferenceEvaluator
    from yobx.reference.torch_ops import OpRunKernel, OpRunTensor

    TFLOAT = onnx.TensorProto.FLOAT

    class SigmoidCPU(OpRunKernel):
        "Custom Sigmoid that always runs on CPU."
        def run(self, x):
            t = x.tensor.cpu()
            return OpRunTensor(torch.sigmoid(t).to(x.tensor.device))

    model = oh.make_model(
        oh.make_graph(
            [oh.make_node("Sigmoid", ["X"], ["Z"])],
            "sigmoid_graph",
            [oh.make_tensor_value_info("X", TFLOAT, [None])],
            [oh.make_tensor_value_info("Z", TFLOAT, [None])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=9,
    )
    sess = TorchReferenceEvaluator(model, custom_kernels={("", "Sigmoid"): SigmoidCPU})
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    (result,) = sess.run(None, {"X": x})
    print(result)

Available kernels
-----------------

.. runpython::
    :showcode:

    from yobx.reference.torch_evaluator import get_kernels

    for k, v in sorted(get_kernels().items()):
        domain, name, version = k
        f = f"{name}({version})" if domain == "" else f"{name}[{domain}]({version})"
        add = " " * max(25 - len(f), 0)
        dd = " -- device dependent" if v.device_dependent() else ""
        print(f"{f}{add} -- {v.__name__}{dd}")

.. seealso::

    :ref:`l-plot-evaluator-comparison` — sphinx-gallery example that runs
    the same model through all three evaluators and verifies the outputs agree.
