.. _l-howto-shape-inference:

Shape inference
===============

This page answers common *"how do I…"* questions for running shape inference
on an ONNX model with :class:`~yobx.xshape.BasicShapeBuilder`.

----

How to run shape inference on an ONNX model
--------------------------------------------

Instantiate :class:`~yobx.xshape.BasicShapeBuilder` and call
:meth:`~yobx.xshape.shape_builder_impl.BasicShapeBuilder.run_model`.  After
the call, query each result tensor by name with
:meth:`~yobx.xshape.ShapeBuilder.get_shape`.

Unlike :func:`onnx.shape_inference.infer_shapes`, which can only propagate
shapes for statically-known integer dimensions, :class:`~yobx.xshape.BasicShapeBuilder`
keeps each dimension as a symbolic arithmetic expression so that output shapes
are expressed in terms of the input dimension names (e.g. ``batch``, ``seq``).

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper as onh
    from yobx.xshape import BasicShapeBuilder

    TFLOAT = onnx.TensorProto.FLOAT
    TINT64 = onnx.TensorProto.INT64

    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Add", ["X", "Y"], ["added"]),
                oh.make_node("Concat", ["added", "X"], ["concat_out"], axis=2),
                oh.make_node("Reshape", ["concat_out", "reshape_shape"], ["Z"]),
            ],
            "add_concat_reshape",
            [
                oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", "d_model"]),
                oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", "d_model"]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, [None, None, None])],
            [onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="reshape_shape")],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )

    builder = BasicShapeBuilder()
    builder.run_model(model)

    for name in ["X", "Y", "added", "concat_out", "Z"]:
        print(f"  {name:15s}  shape={builder.get_shape(name)}")

----

How to compare shape inference approaches
------------------------------------------

Three tools are commonly used for ONNX shape inference:

1. :func:`onnx.shape_inference.infer_shapes` — the standard ONNX tool. It
   propagates shapes only when every dimension is a known integer; symbolic
   (dynamic) dimensions typically become ``None``.

2. `onnx-shape-inference <https://pypi.org/project/onnx-shape-inference/>`_ — a
   third-party package that uses `SymPy <https://www.sympy.org>`_ to track
   dimension expressions on the :pypi:`onnx-ir` representation of the model.

3. :class:`~yobx.xshape.BasicShapeBuilder` — the built-in yobx tool. It keeps
   dimensions as symbolic arithmetic expressions and evaluates constant-shape
   tensors (such as the ``shape`` input of ``Reshape``) to propagate
   information through the graph.

The table below illustrates the difference on a model that contains dynamic
dimensions:

.. runpython::
    :showcode:

    import numpy as np
    import pandas
    import onnx
    import onnx_ir as ir
    import onnx.helper as oh
    import onnx.numpy_helper as onh
    from onnx_shape_inference import infer_symbolic_shapes
    from yobx.xshape import BasicShapeBuilder

    TFLOAT = onnx.TensorProto.FLOAT

    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Add", ["X", "Y"], ["added"]),
                oh.make_node("Concat", ["added", "X"], ["concat_out"], axis=2),
                oh.make_node("Reshape", ["concat_out", "reshape_shape"], ["Z"]),
            ],
            "add_concat_reshape",
            [
                oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", "d_model"]),
                oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", "d_model"]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, [None, None, None])],
            [onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="reshape_shape")],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )

    # onnx.shape_inference.infer_shapes
    inferred = onnx.shape_inference.infer_shapes(model)
    onnx_shapes = {}
    for vi in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        t = vi.type.tensor_type
        if t.HasField("shape"):
            onnx_shapes[vi.name] = tuple(
                d.dim_param if d.dim_param else (d.dim_value if d.dim_value else None)
                for d in t.shape.dim
            )
        else:
            onnx_shapes[vi.name] = "unknown"

    # onnx-shape-inference
    ir_model = ir.serde.deserialize_model(model)
    ir_model = infer_symbolic_shapes(ir_model)
    ir_shapes = {}
    for v in ir_model.graph.inputs:
        ir_shapes[v.name] = str(v.shape)
    for node in ir_model.graph:
        for out in node.outputs:
            ir_shapes[out.name] = str(out.shape)

    # BasicShapeBuilder
    builder = BasicShapeBuilder()
    builder.run_model(model)

    names = ["X", "Y", "added", "concat_out", "Z"]
    rows = []
    for name in names:
        rows.append({
            "name": name,
            "onnx": str(onnx_shapes.get(name, "unknown")),
            "onnx_ir": str(ir_shapes.get(name, "unknown")),
            "basic": str(builder.get_shape(name)),
        })
    print(pandas.DataFrame(rows).set_index("name").to_string())

**Key observations:**

- For the ``added`` and ``concat_out`` intermediate results,
  :func:`onnx.shape_inference.infer_shapes` returns ``None`` for every
  dynamic dimension, whereas :class:`~yobx.xshape.BasicShapeBuilder` keeps
  the symbolic name and derives the doubled axis automatically.
- For the ``Reshape`` output ``Z``, the constant ``reshape_shape``
  tensor (``[0, 0, -1]``) allows :class:`~yobx.xshape.BasicShapeBuilder` to
  evaluate the flattening and express ``Z`` in terms of the input dimensions.
- ``onnx-shape-inference`` reaches the same conclusions for ``added`` and
  ``concat_out`` but may assign a fresh symbol to ``Z`` because it does not
  always evaluate constant-shape tensors.

----

How to evaluate symbolic shapes with concrete values
-----------------------------------------------------

Once the model has been analysed, call
:meth:`~yobx.xshape.ShapeBuilder.evaluate_shape` with a dictionary of
``{dim_name: int}`` values to substitute the symbolic variables and obtain
concrete integer shapes.

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
                oh.make_node("Concat", ["added", "X"], ["concat_out"], axis=2),
                oh.make_node("Reshape", ["concat_out", "reshape_shape"], ["Z"]),
            ],
            "add_concat_reshape",
            [
                oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", "d_model"]),
                oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", "d_model"]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, [None, None, None])],
            [onh.from_array(np.array([0, 0, -1], dtype=np.int64), name="reshape_shape")],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )

    builder = BasicShapeBuilder()
    builder.run_model(model)

    context = {"batch": 2, "seq": 5, "d_model": 8}
    for name in ["X", "Y", "added", "concat_out", "Z"]:
        sym = builder.get_shape(name)
        concrete = builder.evaluate_shape(name, context)
        print(f"  {name:15s}  symbolic={sym!s:30s}  concrete={concrete}")

----

How to estimate the computational cost of a model
--------------------------------------------------

Pass ``inference=InferenceMode.COST`` to
:meth:`~yobx.xshape.shape_builder_impl.BasicShapeBuilder.run_model`.  The
method returns a list of ``(op_type, flops, node)`` triples where *flops* is
either an integer (static shapes), a symbolic string expression (dynamic
shapes), or ``None`` (unsupported operator or unknown shapes).

Cost is expressed in *floating-point operations* (FLOPs).

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    import onnx.helper as oh
    from yobx.xshape import BasicShapeBuilder, InferenceMode

    TFLOAT = onnx.TensorProto.FLOAT

    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("MatMul", ["A", "B"], ["C"]),
                oh.make_node("Relu", ["C"], ["out"]),
            ],
            "matmul_relu",
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

To substitute concrete dimension values and obtain integer FLOPs counts, use
:meth:`~yobx.xshape.shape_builder_impl.BasicShapeBuilder.evaluate_cost_with_true_inputs`:

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    import onnx.helper as oh
    from yobx.xshape import BasicShapeBuilder, InferenceMode

    TFLOAT = onnx.TensorProto.FLOAT

    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("MatMul", ["A", "B"], ["C"]),
                oh.make_node("Relu", ["C"], ["out"]),
            ],
            "matmul_relu",
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

    rng = np.random.default_rng(0)
    feeds = {
        "A": rng.standard_normal((4, 32, 64)).astype(np.float32),
        "B": rng.standard_normal((4, 64, 16)).astype(np.float32),
    }

    concrete = builder.evaluate_cost_with_true_inputs(feeds, cost_list)
    total = 0
    print("Concrete FLOPs per node:")
    for op_type, flops, _ in concrete:
        total += flops or 0
        print(f"  {op_type:<12s}  {flops:>12,}")
    print(f"  {'TOTAL':<12s}  {total:>12,}")

**Cost considerations:**

Running :class:`~yobx.xshape.BasicShapeBuilder` itself introduces overhead
because it must interpret every node in the graph and evaluate any constant
sub-expressions it finds.  For very large models (thousands of nodes) this can
take noticeably longer than
:func:`onnx.shape_inference.infer_shapes`, which relies on a
compiled C++ backend.
:class:`~yobx.xshape.BasicShapeBuilder` is best used offline (during model
export or optimisation) rather than in a hot inference path.

----

How to work with constraints from named output dimensions
---------------------------------------------------------

Some operators — such as ``NonZero`` — introduce a *data-dependent* dimension
whose size cannot be determined from shapes alone.
:class:`~yobx.xshape.BasicShapeBuilder` assigns an internal placeholder name
(e.g. ``NEWDIM_nonzero_0``) to such a dimension.

When the graph output is annotated with **named dimensions**, the builder
detects the mismatch between the computed placeholder and the user-supplied
name, registers the *constraint* ``NEWDIM_nonzero_0 = nnz``, and renames the
placeholder throughout the graph.  Inspect registered constraints with
:meth:`~yobx.xshape.ShapeBuilder.get_registered_constraints`.

.. runpython::
    :showcode:

    import onnx
    import onnx.helper as oh
    from yobx.xshape import BasicShapeBuilder

    TFLOAT = onnx.TensorProto.FLOAT
    TINT64 = onnx.TensorProto.INT64

    nodes = [
        oh.make_node("Abs", ["X"], ["abs_out"]),
        oh.make_node("NonZero", ["abs_out"], ["nz"]),
        oh.make_node("Transpose", ["nz"], ["transposed_nz"]),
    ]
    inputs = [oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq"])]

    # --- anonymous output shapes: placeholder is kept as-is ---
    model_anon = oh.make_model(
        oh.make_graph(
            nodes,
            "nonzero_anon",
            inputs,
            [oh.make_tensor_value_info("transposed_nz", TINT64, [None, None])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )

    # --- named output shapes: constraint is registered ---
    model_named = oh.make_model(
        oh.make_graph(
            nodes,
            "nonzero_named",
            inputs,
            [oh.make_tensor_value_info("transposed_nz", TINT64, ["nnz", "rank"])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )

    b_anon = BasicShapeBuilder()
    b_anon.run_model(model_anon)
    print("anonymous — nz shape         :", b_anon.get_shape("nz"))
    print("anonymous — transposed shape  :", b_anon.get_shape("transposed_nz"))
    print("anonymous — constraints       :", b_anon.get_registered_constraints())

    print()

    b_named = BasicShapeBuilder()
    b_named.run_model(model_named)
    print("named     — nz shape         :", b_named.get_shape("nz"))
    print("named     — transposed shape  :", b_named.get_shape("transposed_nz"))
    print("named     — constraints       :", b_named.get_registered_constraints())

**When to use named output dimensions:**

Provide named dimensions in the graph output annotations whenever the graph
contains data-dependent operators and you want downstream code to be able to
reference those dimensions by a stable name.  The constraint
``NEWDIM_nonzero_0 = nnz`` acts as a *rename directive* that propagates the
user-visible name throughout the entire symbolic shape graph, making the
output of every subsequent node easier to interpret.

.. seealso::

    :ref:`l-design-shape` — design document describing the algorithm behind
    :class:`~yobx.xshape.BasicShapeBuilder` in detail.

    :ref:`l-design-cost` — explanation of how FLOPs are counted for each
    operator type.

    :ref:`l-plot-computed-shapes` — gallery example comparing all three
    shape-inference tools on a larger model.

    :ref:`l-plot-symbolic-cost` — gallery example estimating the cost of an
    attention model before and after optimisation.
