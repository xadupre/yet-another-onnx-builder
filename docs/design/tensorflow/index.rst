.. _l-design-tensorflow-converter:

=========================
TensorFlow Export to ONNX
=========================

:func:`yobx.tensorflow.to_onnx` converts a :epkg:`TensorFlow`/:epkg:`Keras`
model into an :class:`onnx.ModelProto`.  The implementation is a
**proof-of-concept** that traces the model with
:func:`tensorflow.function` / ``get_concrete_function`` and then
converts each TF operation in the resulting computation graph to its
ONNX equivalent via a registry of op-level converters.

High-level workflow
===================

.. code-block:: text

    Keras model / layer
          │
          ▼
      to_onnx()           ← builds TensorSpecs, calls get_concrete_function
          │
          ▼
    ConcreteFunction        (TF computation graph)
          │
          ▼
    _convert_concrete_function()
          │  for every op in the graph …
          ▼
    op converter            ← emits ONNX node(s) via GraphBuilder.op.*
          │
          ▼
    GraphBuilder.to_onnx()  ← validates and returns ModelProto

The steps in detail:

1. :func:`to_onnx <yobx.tensorflow.to_onnx>` accepts the Keras model,
   a tuple of representative *numpy* inputs (used to infer dtypes
   and shapes), and optional ``input_names`` / ``dynamic_shapes``.
2. A :class:`tensorflow.TensorSpec` is built for every input.  By
   default, the batch axis (axis 0) is made dynamic; pass
   ``dynamic_shapes`` to customise which axes are dynamic.
3. :func:`get_concrete_function` traces the model with those specs,
   yielding a :class:`tensorflow.ConcreteFunction` whose ``graph``
   exposes every individual TF operation in execution order.
4. A fresh :class:`~yobx.xbuilder.GraphBuilder` is created and
   :func:`_convert_concrete_function` walks the op list:

   a. **Captured variables** (model weights) are seeded as ONNX
      initializers.
   b. **Placeholder ops** that correspond to real inputs are registered
      via :meth:`make_tensor_input
      <yobx.xbuilder.GraphBuilder.make_tensor_input>`.
   c. Every other op is dispatched to a registered converter (or to an
      entry in ``extra_converters``).
5. The ONNX outputs are declared with :meth:`make_tensor_output
   <yobx.xbuilder.GraphBuilder.make_tensor_output>`.
6. :meth:`GraphBuilder.to_onnx <yobx.xbuilder.GraphBuilder.to_onnx>`
   finalises and returns the :class:`onnx.ModelProto`.

Quick example
=============

.. code-block:: python

    import numpy as np
    import tensorflow as tf
    from yobx.tensorflow import to_onnx

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(2),
    ])

    X = np.random.rand(5, 4).astype(np.float32)
    onx = to_onnx(model, (X,))


Converter registry
==================

The registry is a module-level dictionary
``TF_OP_CONVERTERS: Dict[str, Callable]`` defined in
:mod:`yobx.tensorflow.register`.  Keys are TF op-type strings
(e.g. ``"MatMul"``, ``"Relu"``); values are converter callables.

Registering a converter
-----------------------

Use the :func:`register_tf_op_converter
<yobx.tensorflow.register.register_tf_op_converter>` decorator.  Pass a
single op-type string or a tuple of strings:

.. code-block:: python

    from yobx.tensorflow.register import register_tf_op_converter
    from yobx.xbuilder import GraphBuilder


    @register_tf_op_converter(("MyOp", "MyOpV2"))
    def convert_my_op(g: GraphBuilder, sts: dict, outputs: list, op) -> str:
        return g.op.SomeOnnxOp(op.inputs[0].name, outputs=outputs, name=op.name)

The decorator raises :class:`TypeError` if any of the given op-type
strings is already present in the global registry (including duplicates
within the same tuple, since each string is registered before the next
one is checked).

Looking up a converter
----------------------

:func:`get_tf_op_converter <yobx.tensorflow.register.get_tf_op_converter>`
accepts an op-type string and returns the registered callable, or
``None`` if none is found.

:func:`get_tf_op_converters <yobx.tensorflow.register.get_tf_op_converters>`
returns a copy of the full registry dictionary.

Converter function signature
============================

Every op converter follows the same contract:

``(g, sts, outputs, op[, verbose]) → output_name``

=============  =====================================================
Parameter      Description
=============  =====================================================
``g``          :class:`GraphBuilder <yobx.xbuilder.GraphBuilder>`
               — call ``g.op.<OpType>(…)`` to emit ONNX nodes.
``sts``        ``Dict`` of metadata (currently always ``{}``).
``outputs``    ``List[str]`` of pre-allocated output tensor names
               that the converter **must** write to.
``op``         A :class:`tensorflow.Operation` whose ``inputs``,
               ``outputs``, and attributes describe the TF op.
``verbose``    Optional verbosity level (default ``0``).
=============  =====================================================

The function should return the name of the primary output tensor.
Input tensor names are obtained via ``op.inputs[i].name`` and attribute
values via ``op.get_attr("attr_name")``.

Supported ops
=============

The built-in converters live under :mod:`yobx.tensorflow.ops` and are
loaded on first call to :func:`to_onnx <yobx.tensorflow.to_onnx>` via
:func:`register_tensorflow_converters
<yobx.tensorflow.register_tensorflow_converters>`.

+----------------------------+---------------------------------------------+
| TF op type(s)              | ONNX equivalent                             |
+============================+=============================================+
| ``MatMul``                 | ``MatMul``                                  |
| ``BatchMatMulV2``          | (with optional ``Transpose`` when           |
| ``BatchMatMul``            | ``transpose_a`` / ``transpose_b`` is set)   |
+----------------------------+---------------------------------------------+
| ``BiasAdd``                | ``Add``                                     |
+----------------------------+---------------------------------------------+
| ``Relu``                   | ``Relu``                                    |
+----------------------------+---------------------------------------------+
| ``Relu6``                  | ``Clip(min=0, max=6)``                      |
+----------------------------+---------------------------------------------+
| ``Sigmoid``                | ``Sigmoid``                                 |
+----------------------------+---------------------------------------------+
| ``Tanh``                   | ``Tanh``                                    |
+----------------------------+---------------------------------------------+
| ``Softmax``                | ``Softmax(axis=-1)``                        |
+----------------------------+---------------------------------------------+
| ``ReadVariableOp``         | ``Identity`` (resolves captured variable)   |
+----------------------------+---------------------------------------------+
| ``Const``                  | constant value embedded as initializer      |
+----------------------------+---------------------------------------------+
| ``Identity``               | ``Identity``                                |
+----------------------------+---------------------------------------------+
| ``NoOp``                   | *(nothing emitted)*                         |
+----------------------------+---------------------------------------------+

Dynamic shapes
==============

By default :func:`to_onnx <yobx.tensorflow.to_onnx>` marks axis 0 of
every input as dynamic (unnamed batch dimension).  To control which axes
are dynamic, pass ``dynamic_shapes`` — a tuple of one ``Dict[int, str]``
per input where keys are axis indices and values are symbolic dimension
names:

.. code-block:: python

    # axis 0 named "batch", axis 1 fixed
    onx = to_onnx(model, (X,), dynamic_shapes=({0: "batch"},))

Custom op converters
====================

The ``extra_converters`` parameter of :func:`to_onnx
<yobx.tensorflow.to_onnx>` accepts a mapping from TF op-type string to
converter function.  Entries here take **priority** over the built-in
registry, making it easy to override or extend coverage without
modifying the package.

.. code-block:: python

    import numpy as np
    import tensorflow as tf
    from yobx.tensorflow import to_onnx

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation="relu", input_shape=(3,))
    ])
    X = np.random.rand(5, 3).astype(np.float32)

    def custom_relu(g, sts, outputs, op):
        """Replace Relu with a Clip(0, 1) to saturate outputs at 1."""
        import numpy as np
        return g.op.Clip(
            op.inputs[0].name,
            np.array(0.0, dtype=np.float32),
            np.array(1.0, dtype=np.float32),
            outputs=outputs[:1],
            name=op.name,
        )

    onx = to_onnx(model, (X,), extra_converters={"Relu": custom_relu})

Adding a new built-in converter
================================

To extend the built-in op coverage:

1. Create a new file under ``yobx/tensorflow/ops/`` (e.g.
   ``yobx/tensorflow/ops/reduce.py``).
2. Implement a converter function following the signature above.
3. Decorate it with ``@register_tf_op_converter("ReduceSum")``
   (or a tuple for multiple op types).
4. Import the new module inside the ``register()`` function in
   ``yobx/tensorflow/ops/__init__.py``.

.. code-block:: python

    # yobx/tensorflow/ops/reduce.py
    import numpy as np
    from ..register import register_tf_op_converter
    from ...xbuilder import GraphBuilder


    @register_tf_op_converter("Sum")
    def convert_reduce_sum(g: GraphBuilder, sts: dict, outputs: list, op) -> str:
        """TF ``Sum`` → ONNX ``ReduceSum``."""
        axes = op.inputs[1].name  # axis tensor
        keepdims = int(op.get_attr("keep_dims"))
        return g.op.ReduceSum(
            op.inputs[0].name, axes, keepdims=keepdims, outputs=outputs, name=op.name
        )
