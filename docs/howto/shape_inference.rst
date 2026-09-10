.. _l-howto-shape-inference:

Native shape inference
======================

:class:`~yobx.xshape.NativeShapeInference` uses the published ``onnx-light``
wheel's ``onnx_light.onnx_core.shape_inference.ShapesContext``. Models, nodes,
and tensors must be native ``onnx-light`` protos. There is no reference
``onnx`` dependency or Python shape-engine fallback.

Symbolic shapes and costs
-------------------------

.. code-block:: python

    import numpy as np
    from onnx_light.onnx import TensorProto, helper
    from yobx.xshape import InferenceMode, NativeShapeInference

    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Add", ["X", "Y"], ["Z"])],
            "symbolic_add",
            [
                helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 3]),
                helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3]),
            ],
            [helper.make_tensor_value_info("Z", TensorProto.FLOAT, None)],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    inference = NativeShapeInference()
    costs = inference.run_model(model, inference=InferenceMode.COST)
    assert inference.get_shape("Z") == ("N", 3)
    assert inference.get_type("Z") == TensorProto.FLOAT
    concrete = inference.evaluate_cost_with_true_inputs(
        {"X": np.zeros((5, 3), dtype=np.float32)}, costs
    )
    assert concrete[0][1] == 15

``COST`` returns ``(op_type, flops, input_shapes)`` triples. FLOPs are integers,
symbolic expressions, or ``None`` when the estimate is unavailable. Python
cost formulas perform arithmetic on native inferred dimensions; they do not
perform shape inference.

Modes
-----

* ``SHAPE`` (default) exposes native shapes and element types.
* ``TYPE`` uses native inference but exposes only element types.
* ``COST`` additionally computes the per-node estimates shown above.
* ``NOTHING`` does not infer shapes or types.

Names can also be supplied as case-insensitive strings, for example
``inference="cost"``.

Serialization
-------------

``run_model`` does not rewrite the input model. To obtain a copy containing
inferred annotations, preserving the model's metadata:

.. code-block:: python

    inferred = inference.to_onnx()
    data = inferred.SerializeToString()

Alternatively, ``inference.update_shapes(model)`` writes native inferred
annotations to an existing model or graph in place.

Incremental and custom inference
---------------------------------

``set_type``, ``set_shape``, and ``run_node`` provide incremental inference.
The public ``context`` also exposes the wheel's native constraint,
shape-value, event, and custom-callback APIs:

.. code-block:: python

    def infer_custom(context, node):
        context.set(str(node.output[0]), context.get(str(node.input[0])))

    inference.context.set_custom_shape_inference_function(
        "custom", "IdentityLike", infer_custom
    )

An independent model requires a new adapter or a call to
``reset_types_and_shapes()`` first. Resetting clears the native descriptors,
constraints, and custom callbacks; custom callbacks must then be registered
again.

Limitations and unknown values
-------------------------------

Anonymous dimensions are returned as ``None`` rather than invented Python
symbols; costs requiring them are unavailable. ``value_as_shape(name)``
returns only shape values computed by the native engine, or ``None``.
Equality constraints are registered and resolved by ``ShapesContext``.
The published 0.1.26 wheel does not propagate shape-tensor values through every
operator (for example, ``Identity``, ``Mul``, and ``Slice`` leave them unavailable).
The adapter does not fill those gaps with a Python evaluator. Native expression
evaluation supports the wheel's exact-division syntax, such as ``N/:2``.
Updating an element type or shape preserves the descriptor's native shape values
and minimum/maximum value bounds.

``Squeeze`` with omitted axes removes singleton dimensions. However, an empty
axes tensor supplied as a graph input does not currently produce that same
inferred shape, even when the axes input has a declared shape of ``[0]``.

The current wheel represents unknown rank and scalar rank with the same native
descriptor. Shape/cost inference therefore rejects graph inputs without a
declared rank rather than silently treating them as scalars. Unknown dimensions
within a known rank remain supported.

Unsupported operators and native inference errors propagate; ``exc=False``
does not enable a fallback engine or swallow them. ``TYPE`` still uses the
native symbolic engine, so its operator coverage is the wheel's coverage rather
than that of the legacy Python type helper.
