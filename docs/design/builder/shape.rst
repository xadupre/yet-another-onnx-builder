
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

The whole algorithm relies on four components:

* An **analyser for expressions** able to parse and simplify numerical expressions
  built upon name for the dynamic dimension sets of the inputs,
* A list of **functions inferring shapes**, including the numerical
  expressions for every ONNX operator,
* A very **simple runtime** able to run a short list of kernels usually used to
  handle shapes (Add, Sub, Mul, Div, Concat, Squeeze, Unsqueeze, Shape, Size, Reshape),
* An algorithm solving **constraints** after inferring function was run.
  An unknown dimension may be known or at least constrained to a short set of values
  after a binary operator (or any other) was processed. The constraint mechanism is
  put in place to implement a kind of backward pass where output dimensions
  restricts the number of possible values for input dimensions.

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

    from yobx.xexpressions import simplify_expression

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
    from yobx.xexpressions import evaluate_expression
    from yobx.xshape import BasicShapeBuilder

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

    :ref:`l-plot-shape-expressions` — sphinx-gallery
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

.. _l-design-xshape-debugging:

Debugging Shape Inference with Environment Variables
====================================================

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

Constraint Mechanism
====================

When :class:`BasicShapeBuilder <yobx.xshape.shape_builder_impl.BasicShapeBuilder>`
processes a broadcasting operation (e.g. ``Add``, ``Mul``, ``Where``) it computes
the output shape with
:func:`broadcast_shape <yobx.xshape.shape_type_compute.broadcast_shape>`.
If one input dimension is a symbolic string (unknown at graph-construction time)
and the other is a concrete integer that is not ``1``, the builder registers a
**constraint** equating the symbolic name to the concrete value.

Why constraints are needed
--------------------------

Without constraints, the shape of the broadcast result would be left as the
symbolic name (e.g. ``"d_model"``), and any operation that follows would inherit
this uncertainty.  Later, when a downstream node reveals the concrete value, the
builder would have to *backtrack* and update all earlier shapes — an expensive
and error-prone operation that is not implemented.

With constraints, the concrete value is used **immediately** as the output
dimension, and the equality ``symbolic_name = concrete_value`` is stored.
Downstream operations can propagate the concrete shape without revisiting
previous nodes.

How constraints are registered
--------------------------------

:func:`broadcast_shape <yobx.xshape.shape_type_compute.broadcast_shape>` applies
the following rules for each pair of aligned dimensions ``(a, b)``:

+---------------------------+---------------------------+--------+----------------------------------+
| ``a``                     | ``b``                     | Result | Constraint registered            |
+===========================+===========================+========+==================================+
| symbolic string           | concrete int ``n ≠ 0, 1`` | ``n``  | ``a = n``                        |
+---------------------------+---------------------------+--------+----------------------------------+
| concrete int ``n ≠ 0, 1`` | symbolic string           | ``n``  | ``b = n``                        |
+---------------------------+---------------------------+--------+----------------------------------+
| symbolic string           | ``1``                     | ``a``  | *(none — 1 broadcasts freely)*   |
+---------------------------+---------------------------+--------+----------------------------------+
| two symbolic strings      | ``a == b``                | ``a``  | *(none — already equal)*         |
+---------------------------+---------------------------+--------+----------------------------------+
| two symbolic strings      | ``a != b``                | ``a^b``| *(none — max expression)*        |
+---------------------------+---------------------------+--------+----------------------------------+

The concrete integer is always chosen as the output dimension so that subsequent
operations see a precise shape immediately.

Example: broadcasting after an unknown dimension
-------------------------------------------------

.. runpython::
    :showcode:

    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper as onh
    import numpy as np
    from yobx.xshape import BasicShapeBuilder
    from yobx.xshape.shape_type_compute import broadcast_shape

    TFLOAT = onnx.TensorProto.FLOAT

    # X has dynamic last dimension "d_model"; bias has static size 64.
    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Add", ["X", "bias"], ["Z"]),
                oh.make_node("MatMul", ["Z", "W"], ["Out"]),
            ],
            "graph",
            [oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", "d_model"])],
            [oh.make_tensor_value_info("Out", TFLOAT, [None, None, None])],
            [
                onh.from_array(np.zeros((64,), dtype=np.float32), name="bias"),
                onh.from_array(np.random.randn(64, 32).astype(np.float32), name="W"),
            ],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )

    builder = BasicShapeBuilder()
    builder.run_model(model)

    for name in ["X", "Z", "Out"]:
        print(f"{name:5s}  shape={builder.get_shape(name)}")

    # The constraint records that d_model equals 64
    print("constraints:", builder.get_registered_constraints())

When the ``Add`` node is processed, ``broadcast_shape`` aligns
``("batch", "seq", "d_model")`` with ``(64,)`` (right-padded to
``(1, 1, 64)``).  The pair ``("d_model", 64)`` triggers the constraint
``"d_model" = 64``.  The output shape ``Z`` therefore becomes
``("batch", "seq", 64)`` rather than ``("batch", "seq", "d_model")``, and
the ``MatMul`` handler can propagate the shape of ``Out`` immediately as
``("batch", "seq", 32)`` without any backtracking.

Constraint API
--------------

Three methods on :class:`ShapeBuilder <yobx.xshape.shape_builder.ShapeBuilder>`
expose the constraint registry:

* :meth:`register_constraint_dimension(dim_name, value)
  <yobx.xshape.ShapeBuilder.register_constraint_dimension>` — record that the
  symbolic dimension ``dim_name`` is equal to ``value`` (an integer or another
  symbolic name).  Called automatically by ``broadcast_shape`` when needed.
* :meth:`add_to_constraints(dim_name, value)
  <yobx.xshape.ShapeBuilder.add_to_constraints>` — lower-level helper that
  accepts a set of values as well as a single value.
* :meth:`get_registered_constraints()
  <yobx.xshape.ShapeBuilder.get_registered_constraints>` — returns the full
  mapping ``{dim_name: {values}}`` accumulated so far.

The registry is also used by
:meth:`_improves_dynamic_dimension_naming
<yobx.xshape.ShapeBuilder._improves_dynamic_dimension_naming>` to replace
internal opaque tokens (e.g. ``"s0"``, ``"DYN0"``) with user-visible names
once the relationships between them are known.

Implementing a new shape function
==================================

Adding support for a new operator (or overriding an existing one) requires
writing a small function and registering it in
:mod:`yobx.xshape.shape_type_compute`.

Shape functions signature
--------------------------

Every shape function receives two arguments:

* ``g`` — the :class:`ShapeBuilder <yobx.xshape.shape_builder.ShapeBuilder>`
  instance that holds all currently known shapes, types, ranks, and devices.
* ``node`` — the :class:`onnx.NodeProto` being processed.

A minimal shape function expects to see the following API :ref:`l-design-expected-api`
and it should do:

1. **Propagate device** — if ``g.has_device(input)`` is true, copy the
   device to the output with ``set_device``.
2. **Propagate type** — guard with ``g.has_type(input)`` before calling
   ``set_type`` on every output; return ``None`` early if the type is not yet
   known.
3. **Compute and set the shape** — guard with ``g.has_shape(input)`` before
   deriving the output shape and calling ``set_shape``.  When the full shape
   is unavailable, fall back to ``g.has_rank`` / ``set_rank``.
4. **Return the shape** (or ``True`` if only a rank was set, or ``None`` if
   nothing could be done).

Example: a custom element-wise scaling operator
-------------------------------------------------

The following example shows a shape function for a hypothetical ``Scale``
operator (domain ``"my.domain"``) that multiplies its first input ``X`` by a
scalar ``scale`` and returns a result with the same shape and type as ``X``.

.. code-block:: python

    from onnx import NodeProto
    from yobx.xshape.shape_builder import ShapeBuilder

    def _set_shape_type_scale(g: ShapeBuilder, node: NodeProto):
        "Shape function for the custom Scale operator."
        x = node.input[0]
        out = node.output[0]

        # 1. propagate device
        if g.has_device(x):
            g.set_device(out, g.get_device(x))

        # 2. propagate element type
        if not g.has_type(x):
            return None
        g.set_type(out, g.get_type(x))

        # 3. compute output shape (same shape as input)
        if g.has_shape(x):
            shape = g.get_shape(x)
            g.set_shape(out, shape)
            return shape

        # fallback: propagate rank only
        if g.has_rank(x):
            g.set_rank(out, g.get_rank(x))
            return True

        return None

To register the function so that
:class:`BasicShapeBuilder <yobx.xshape.shape_builder_impl.BasicShapeBuilder>`
calls it automatically, add it to the appropriate registry dictionary in
:mod:`yobx.xshape.shape_type_compute`:

* ``_set_shape_type_op_any_known`` — for standard ONNX operators
  (domain ``""``).
* ``_set_shape_type_op_any_custom`` — for operators in non-standard domains
  (e.g. ``"com.microsoft"``).

.. code-block:: python

    # In yobx/xshape/shape_type_compute.py:
    _set_shape_type_op_any_custom["Scale"] = _set_shape_type_scale

The function will then be called automatically whenever
:meth:`run_node <yobx.xshape.shape_builder_impl.BasicShapeBuilder.run_node>`
processes a ``Scale`` node.
