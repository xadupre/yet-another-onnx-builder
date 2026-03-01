
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

Class Hierarchy
===============

:class:`BasicShapeBuilder <yobx.xshape.shape_builder_impl.BasicShapeBuilder>`
is the main concrete implementation and is composed of four cooperative
base classes:

* :class:`ShapeBuilder <yobx.xshape.shape_builder.ShapeBuilder>` — the public
  API contract: ``get_shape``, ``get_type``, ``get_rank``, ``has_shape``,
  ``has_type``, ``has_rank``, ``set_shape``, ``set_type``, ``set_rank``,
  ``evaluate_shape``, ``compare_with_true_inputs``, ``update_shapes``.
* :class:`_InferenceRuntime <yobx.xshape._inference_runtime._InferenceRuntime>`
  — walks the graph node by node, dispatching each node to the matching
  per-operator handler in :mod:`yobx.xshape.shape_type_compute`.
* :class:`_BuilderRuntime <yobx.xshape._builder_runtime._BuilderRuntime>`
  — evaluates small constant sub-expressions (e.g. the ``[0, 0, -1]``
  passed to a ``Reshape`` node) so the builder can resolve ``-1`` to the
  correct symbolic formula.
* :class:`_ShapeRuntime <yobx.xshape._shape_runtime._ShapeRuntime>`
  — handles the special *value-as-shape* tracking needed by operators such
  as ``Shape``, ``Gather``, ``Concat``, and ``Slice`` when their output
  feeds directly into a ``Reshape``.

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

Validating computed shapes
==========================

Once the model has been run with concrete inputs you can verify that the
symbolic shapes predicted by
:class:`BasicShapeBuilder <yobx.xshape.shape_builder_impl.BasicShapeBuilder>`
agree with the actual tensor shapes using
:meth:`compare_with_true_inputs <yobx.xshape.ShapeBuilder.compare_with_true_inputs>`.
The method accepts the concrete input and output dictionaries (or lists) and
returns, for every output result, the list of ``(symbolic_expr, expected, computed)``
triples.

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    import onnx.helper as oh
    from yobx.reference import ExtendedReferenceEvaluator
    from yobx.xshape import BasicShapeBuilder

    TFLOAT = onnx.TensorProto.FLOAT

    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Add", ["X", "Y"], ["added"]),
                oh.make_node("Concat", ["added", "X"], ["Z"], axis=2),
            ],
            "add_concat",
            [
                oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", "d_model"]),
                oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", "d_model"]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, [None, None, None])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )

    builder = BasicShapeBuilder()
    builder.run_model(model)

    feeds = {
        "X": np.random.rand(2, 5, 4).astype(np.float32),
        "Y": np.random.rand(2, 5, 4).astype(np.float32),
    }
    session = ExtendedReferenceEvaluator(model)
    outputs = session.run(None, feeds)

    result = builder.compare_with_true_inputs(feeds, outputs)
    for name, dims in result.items():
        print(f"{name}: {dims}")

Each triple ``(expr, expected, computed)`` confirms that evaluating the
symbolic expression with the concrete dimension values yields the same size as
the tensor produced by the runtime.

Writing shapes back to a model
===============================

:meth:`update_shapes <yobx.xshape.ShapeBuilder.update_shapes>` writes the
inferred shapes and types back into the ``value_info`` section of the
``onnx.ModelProto``.  Inputs, outputs, and initializers are left untouched;
only intermediate results (node outputs that are neither inputs, outputs, nor
initializers) are annotated.

This is useful for visualisation tools and downstream passes that rely on
``value_info`` being populated.

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper as onh
    from yobx.xshape import BasicShapeBuilder

    TFLOAT = onnx.TensorProto.FLOAT

    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Add", ["X", "Y"], ["added"]),
                oh.make_node("MatMul", ["added", "W"], ["Z"]),
            ],
            "add_matmul",
            [
                oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", 64]),
                oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", 64]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, [None, None, None])],
            [onh.from_array(np.random.randn(64, 32).astype(np.float32), name="W")],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )

    builder = BasicShapeBuilder()
    builder.run_model(model)

    print("value_info before:", [vi.name for vi in model.graph.value_info])
    builder.update_shapes(model)
    print("value_info after :", [vi.name for vi in model.graph.value_info])

    vi = model.graph.value_info[0]
    t = vi.type.tensor_type
    shape = tuple(d.dim_param if d.dim_param else d.dim_value for d in t.shape.dim)
    print(f"  {vi.name}: dtype={t.elem_type}  shape={shape}")

Debugging
=========

:class:`BasicShapeBuilder <yobx.xshape.shape_builder_impl.BasicShapeBuilder>`
respects several environment variables that help narrow down shape-inference
problems:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Environment variable
     - Effect
   * - ``ONNXSTOPSHAPE=<name>``
     - Raises an exception the moment result ``<name>`` receives a shape.
       Useful for finding the first place where a wrong shape is assigned.
   * - ``ONNXSTOPTYPE=<name>``
     - Raises an exception the moment result ``<name>`` receives a type.
   * - ``ONNXDYNDIM=<name>``
     - Prints a message every time the dynamic dimension ``<name>`` is
       encountered during shape propagation.
   * - ``ONNXCST=1``
     - Prints which constant value is being requested during inference.
   * - ``ONNXSHAPECOMPUTE=1``
     - Raises an exception when a shape is missing for a result that should
       have one.
   * - ``ONNXSTOPVALUESHAPE=<name>``
     - Prints extra information inside the function that tracks shapes of
       results used as shape arguments (e.g. inputs to ``Reshape``).

In addition, :meth:`get_debug_msg
<yobx.xshape.shape_builder_impl.BasicShapeBuilder.get_debug_msg>` returns a
detailed text dump of the builder's internal state (known shapes, types,
constants, ranks, and the sequence of calls) which can be printed or logged
whenever an assertion fails.
