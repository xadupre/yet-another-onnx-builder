
.. _l-design-shape:

============
ShapeBuilder
============

:func:`onnx.shape_inference.infer_shapes` tries to infer
shapes and types based on input shapes. It does not
supports formulas and introduces new symbols.

:class:`yobx.xshape.ShapeBuilder`
class walks through all nodes and looks into a list of functions
computing the output shapes based on the node type.
It tries as much as possible to express the new shape with formulas
based on the dimensions used to defined the inputs.
The list of functions is available in :mod:`yobx.xshape.shape_type_compute`
called from class :class:`_InferenceRuntime <yobx.xshape._inference_runtime._InferenceRuntime>`.

While doing this, every function may try to compute some tiny constants
in :class:`_BuilderRuntime <yobx.xshape._builder_runtime._BuilderRuntime>`.
This is used by :class:`_ShapeRuntime <yobx.xshape._shape_runtime._ShapeRuntime>`
to deduce some shapes.

For example, if **X** has shape ``("d1", 2)`` then ``Shape(X, start=1)`` is constant ``[2]``.
This can be later used to infer the shape after a reshape.

After getting an expression, a few postprocessing are applied to reduce
its complexity. This relies on :mod:`ast`. It is done by function
:func:`simplify_expression <yobx.xshape.simplify_expressions.simplify_expression>`.
``d + f - f`` is replaced by ``d``.

Symbolic Expressions
====================

When input shapes contain unknown (dynamic) dimensions, :class:`ShapeBuilder
<yobx.xshape.ShapeBuilder>` represents each dimension as either:

* an **integer** — for statically known sizes, or
* a **string** — for symbolic (dynamic) sizes.

Symbolic strings are valid Python arithmetic expressions built from the names
of the original dynamic dimensions.  For example, if the two inputs of a
``Concat(axis=1)`` node have shapes ``("batch", "seq1")`` and
``("batch", "seq2")``, the output shape is ``("batch", "seq1+seq2")``.

Supported operators in symbolic expressions
--------------------------------------------

* ``+``  addition (e.g. ``seq1+seq2``)
* ``-``  subtraction (e.g. ``total-seq``)
* ``*``  multiplication (e.g. ``2*seq``)
* ``//``  floor division (e.g. ``seq//2``)
* ``%``  modulo
* ``^``  used internally to represent ``max(a, b)``
  (e.g. ``a^b`` evaluates to ``max(a, b)``)

Automatic simplification
-------------------------

Before storing a symbolic dimension,
:func:`simplify_expression <yobx.xshape.simplify_expressions.simplify_expression>`
rewrites the expression to its simplest equivalent form:

.. runpython::
    :showcode:

    from yobx.xshape.simplify_expressions import simplify_expression

    print(simplify_expression("d + f - f"))       # d
    print(simplify_expression("2 * seq // 2"))    # seq
    print(simplify_expression("1024 * a // 2"))   # 512*a
    print(simplify_expression("b + a"))           # a+b  (terms sorted)

Evaluating symbolic expressions at runtime
-------------------------------------------

Once the concrete integer values of the input dimensions are known,
:func:`evaluate_expression <yobx.xshape.evaluate_expressions.evaluate_expression>`
can resolve any symbolic dimension to its actual integer value.
:meth:`evaluate_shape <yobx.xshape.ShapeBuilder.evaluate_shape>` applies this
to a whole shape at once.

.. runpython::
    :showcode:

    import onnx
    import onnx.helper as oh
    from yobx.xshape import BasicShapeBuilder
    from yobx.xshape.evaluate_expressions import evaluate_expression

    TFLOAT = onnx.TensorProto.FLOAT

    model = oh.make_model(
        oh.make_graph(
            [oh.make_node("Concat", ["X", "Y"], ["Z"], axis=1)],
            "graph",
            [
                oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq1"]),
                oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq2"]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, [None, None])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )

    builder = BasicShapeBuilder()
    builder.run_model(model)

    # Symbolic shape of Z
    sym_shape = builder.get_shape("Z")
    print("symbolic shape :", sym_shape)

    # Evaluate each dimension given concrete values
    context = dict(batch=3, seq1=5, seq2=7)
    concrete = builder.evaluate_shape("Z", context)
    print("concrete shape :", concrete)

.. seealso::

    :ref:`plot-shape-expressions` — sphinx-gallery
    example demonstrating ``Concat``, ``Reshape``, and ``Split`` symbolic
    expressions, automatic simplification, and evaluation with concrete values.

Example
=======

The following example builds a small ONNX graph, runs
:class:`BasicShapeBuilder <yobx.xshape.shape_builder_impl.BasicShapeBuilder>`
on it, and prints the inferred shapes and types.

.. runpython::
    :showcode:

    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper as onh
    import numpy as np
    from yobx.xshape import BasicShapeBuilder

    TFLOAT = onnx.TensorProto.FLOAT

    # A small model: reshape X then multiply by a weight matrix W.
    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Reshape", ["X", "shape"], ["Xr"]),
                oh.make_node("MatMul", ["Xr", "W"], ["Z"]),
            ],
            "graph",
            [oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", 64])],
            [oh.make_tensor_value_info("Z", TFLOAT, ["batch", "seq", 32])],
            [
                onh.from_array(np.array([0, 0, 64], dtype=np.int64), name="shape"),
                onh.from_array(np.random.randn(64, 32).astype(np.float32), name="W"),
            ],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )

    builder = BasicShapeBuilder()
    builder.run_model(model)

    for name in ["X", "Xr", "W", "Z"]:
        print(
            f"{name:5s}  type={builder.get_type(name)}"
            f"  shape={builder.get_shape(name)}"
        )

Comparison with ONNX shape inference
=====================================

:func:`onnx.shape_inference.infer_shapes` is ONNX's built-in shape
propagation pass.  It works well for models with fully static dimensions but
loses symbolic relationships when dimensions are dynamic: intermediate results
receive freshly generated, unrelated symbols (e.g. ``unk__0``, ``unk__1``)
instead of expressions derived from the input dimensions.

:class:`BasicShapeBuilder <yobx.xshape.shape_builder_impl.BasicShapeBuilder>`
does better in this case because it:

1. **Carries symbolic names** — every dynamic dimension keeps the name given in
   the input ``value_info`` (e.g. ``batch``, ``seq``, ``d_model``).
2. **Builds arithmetic expressions** — when an operator changes a dimension
   (e.g. ``Concat`` along an axis doubles ``d_model``) the result is stored as
   the string expression ``"2*d_model"`` rather than a new opaque symbol.
3. **Folds constants** — initializer tensors that appear as shape arguments
   (e.g. the ``[0, 0, -1]`` passed to ``Reshape``) are evaluated at
   inference-time, which lets the builder resolve the ``-1`` placeholder to
   the correct symbolic formula.
4. **Simplifies** — the resulting expression is reduced to its simplest form
   by :func:`simplify_expression <yobx.xshape.simplify_expressions.simplify_expression>`
   before being stored (``2*d_model//2`` → ``d_model``, etc.).

The table below summarises the difference for a model that applies
``Add → Concat(axis=2) → Reshape([0,0,-1])`` to inputs of shape
``(batch, seq, d_model)``:

+---------------------+--------------------------------------+-----------------------------+
| result              | ``infer_shapes``                     | ``BasicShapeBuilder``       |
+=====================+======================================+=============================+
| ``added``           | ``(batch, seq, d_model)``            | ``(batch, seq, d_model)``   |
+---------------------+--------------------------------------+-----------------------------+
| ``concat_out``      | ``(batch, seq, unk__0)``             | ``(batch, seq, 2*d_model)`` |
+---------------------+--------------------------------------+-----------------------------+
| ``Z``               | ``(batch, seq, unk__1)``             | ``(batch, seq, 2*d_model)`` |
+---------------------+--------------------------------------+-----------------------------+

See :ref:`l-plot-computed-shapes` for a runnable example that demonstrates
this comparison step by step.
