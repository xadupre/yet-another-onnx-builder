.. _l-design-litert-converter:

===================================
LiteRT / TFLite Export to ONNX
===================================

.. toctree::
   :maxdepth: 1

   supported_ops

:func:`yobx.litert.to_onnx` converts a :epkg:`TFLite`/:epkg:`LiteRT`
``.tflite`` model into an :class:`onnx.ModelProto`.  The implementation is
a **proof-of-concept** that parses the binary
`FlatBuffer <https://flatbuffers.dev/>`_ format of ``.tflite`` files with a
minimal pure-Python parser (no external library required) and converts each
operator in the graph to its ONNX equivalent via a registry of op-level
converters.

High-level workflow
===================

.. code-block:: text

    .tflite file / bytes
           │
           ▼
       to_onnx()           ← parse_tflite_model() + build TFLiteModel
           │
           ▼
    TFLiteModel             (pure-Python object hierarchy)
           │
           ▼
    _convert_subgraph()
           │  for every operator in the subgraph …
           ▼
    op converter            ← emits ONNX node(s) via GraphBuilder.op.*
           │
           ▼
    GraphBuilder.to_onnx()  ← validates and returns ModelProto


The steps in detail:

1. :func:`to_onnx <yobx.litert.to_onnx>` accepts the model as a file path
   or raw bytes together with a tuple of representative *numpy* inputs (used
   to determine input dtypes and shapes when they are not fully specified in
   the TFLite model).
2. :func:`~yobx.litert.litert_helper.parse_tflite_model` reads the binary
   FlatBuffer and returns a :class:`~yobx.litert.litert_helper.TFLiteModel`
   object containing a list of
   :class:`~yobx.litert.litert_helper.TFLiteSubgraph` objects, each with its
   tensors and operators.
3. A fresh :class:`~yobx.xbuilder.GraphBuilder` is created and
   :func:`~yobx.litert.convert._convert_subgraph` walks the operator list:

   a. **Weight tensors** (tensors that carry buffer data and are not graph
      inputs) are registered as ONNX initializers.
   b. **Input tensors** are registered via
      :meth:`~yobx.xbuilder.GraphBuilder.make_tensor_input`.
   c. Every operator is dispatched to a registered converter (or to an entry
      in ``extra_converters``).

4. The ONNX outputs are declared with
   :meth:`~yobx.xbuilder.GraphBuilder.make_tensor_output`.
5. :meth:`GraphBuilder.to_onnx <yobx.xbuilder.GraphBuilder.to_onnx>` finalises
   and returns the :class:`onnx.ModelProto`.


Quick example
=============

.. code-block:: python

    import numpy as np
    from yobx.litert import to_onnx

    X = np.random.rand(1, 4).astype(np.float32)
    onx = to_onnx("model.tflite", (X,))


TFLite FlatBuffer parser
========================

:mod:`yobx.litert.litert_helper` contains a self-contained, zero-dependency
FlatBuffer reader (:class:`~yobx.litert.litert_helper._FlatBuf`) together
with the parsed data-classes
(:class:`~yobx.litert.litert_helper.TFLiteModel`,
:class:`~yobx.litert.litert_helper.TFLiteSubgraph`,
:class:`~yobx.litert.litert_helper.TFLiteTensor`,
:class:`~yobx.litert.litert_helper.TFLiteOperator`) and the
:class:`~yobx.litert.litert_helper.BuiltinOperator` enum.

.. runpython::
    :showcode:

    from yobx.litert.litert_helper import (
        _make_sample_tflite_model,
        parse_tflite_model,
    )

    model = parse_tflite_model(_make_sample_tflite_model())
    sg = model.subgraphs[0]
    print(f"subgraphs : {len(model.subgraphs)}")
    print(f"tensors   : {[t.name for t in sg.tensors]}")
    print(f"inputs    : {sg.inputs}")
    print(f"outputs   : {sg.outputs}")
    for op in sg.operators:
        print(f"operator  : {op.name}  inputs={op.inputs}  outputs={op.outputs}")


Converter registry
==================

The registry is a module-level dictionary
``LITERT_OP_CONVERTERS: Dict[Union[int, str], Callable]`` defined in
:mod:`yobx.litert.register`.  Keys are
:class:`~yobx.litert.litert_helper.BuiltinOperator` integers
(e.g. ``BuiltinOperator.RELU`` = ``19``); custom ops use their string name.
Values are converter callables.

Registering a converter
-----------------------

Use the :func:`~yobx.litert.register.register_litert_op_converter` decorator.
Pass a single op-code integer, a custom-op name string, or a tuple thereof:

.. code-block:: python

    from yobx.litert.register import register_litert_op_converter
    from yobx.litert.litert_helper import BuiltinOperator


    @register_litert_op_converter(BuiltinOperator.RELU)
    def convert_relu(g, sts, outputs, op):
        return g.op.Relu(op.inputs[0], outputs=outputs, name="relu")


Converter function signature
============================

Every op converter follows the same contract:

``(g, sts, outputs, op) → output_name``

=============  =====================================================
Parameter      Description
=============  =====================================================
``g``          :class:`GraphBuilder <yobx.xbuilder.GraphBuilder>`
               — call ``g.op.<OpType>(…)`` to emit ONNX nodes.
``sts``        ``Dict`` of metadata (currently always ``{}``).
``outputs``    ``List[str]`` of pre-allocated output tensor names
               that the converter **must** write to.
``op``         An :class:`~yobx.litert.convert._OpProxy` whose
               ``inputs`` and ``outputs`` are **string names** of
               the tensors, and ``builtin_options`` is a decoded
               attribute dict.
=============  =====================================================

``op.inputs[i]`` is the ONNX tensor name (a string, not an integer index)
of the *i*-th operator input.  Use it directly in ``g.op.*()`` calls.

Dynamic shapes
==============

By default :func:`to_onnx <yobx.litert.to_onnx>` marks axis 0 of every
input as dynamic (unnamed batch dimension).  To control which axes are
dynamic, pass ``dynamic_shapes`` — a tuple of one ``Dict[int, str]`` per
input where keys are axis indices and values are symbolic dimension names:

.. code-block:: python

    onx = to_onnx("model.tflite", (X,), dynamic_shapes=({0: "batch"},))


Custom op converters
====================

The ``extra_converters`` parameter of :func:`to_onnx <yobx.litert.to_onnx>`
accepts a mapping from :class:`~yobx.litert.litert_helper.BuiltinOperator`
integer (or custom-op name string) to converter function.  Entries here take
**priority** over the built-in registry:

.. code-block:: python

    import numpy as np
    from yobx.litert import to_onnx
    from yobx.litert.litert_helper import BuiltinOperator

    def custom_relu(g, sts, outputs, op):
        """Replace RELU with Clip(0, 6)."""
        return g.op.Clip(
            op.inputs[0],
            np.array(0.0, dtype=np.float32),
            np.array(6.0, dtype=np.float32),
            outputs=outputs,
            name="relu6",
        )

    onx = to_onnx("model.tflite", (X,),
                  extra_converters={BuiltinOperator.RELU: custom_relu})


Adding a new built-in converter
================================

1. Create a new file under ``yobx/litert/ops/`` (e.g. ``yobx/litert/ops/cast_ops.py``).
2. Implement a converter function following the signature above.
3. Decorate it with ``@register_litert_op_converter(BuiltinOperator.CAST)``.
4. Import the new module inside the ``register()`` function in
   ``yobx/litert/ops/__init__.py``.

.. code-block:: python

    # yobx/litert/ops/cast_ops.py
    from onnx import TensorProto
    from ..register import register_litert_op_converter
    from ..litert_helper import BuiltinOperator
    from ...xbuilder import GraphBuilder

    @register_litert_op_converter(BuiltinOperator.CAST)
    def convert_cast(g: GraphBuilder, sts: dict, outputs: list, op) -> str:
        """TFLite CAST → ONNX Cast."""
        # op.builtin_options carries "in_data_type" / "out_data_type" integers.
        out_type = op.builtin_options.get("out_data_type", 0)
        from yobx.litert.litert_helper import litert_dtype_to_np_dtype
        import numpy as np
        np_dtype = litert_dtype_to_np_dtype(out_type)
        onnx_dtype = TensorProto.FLOAT  # map np_dtype → TensorProto int
        return g.op.Cast(op.inputs[0], to=onnx_dtype, outputs=outputs, name="cast")


Supported ops
=============

See :ref:`l-design-litert-supported-ops` for the full list of
built-in LiteRT op converters, generated automatically from the live registry.
