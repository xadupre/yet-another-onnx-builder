
.. _l-design-graph-builder:

============
GraphBuilder
============

:class:`yobx.xbuilder.GraphBuilder` simplifies the programmatic construction
and optimization of ONNX graphs.  It is the primary tool used to convert a
:class:`torch.fx.Graph` into a :class:`onnx.ModelProto`, but it can equally
be used standalone to build or transform any ONNX graph from scratch.

Class Hierarchy
===============

:class:`GraphBuilder <yobx.xbuilder.GraphBuilder>` is composed of three
cooperative base classes:

* :class:`_BuilderRuntime <yobx.xshape._builder_runtime._BuilderRuntime>`
  — evaluates small constant sub-expressions (e.g. the ``[0, 0, -1]`` passed
  to a ``Reshape`` node) so the builder can resolve ``-1`` to the correct
  symbolic formula and fold constants early.
* :class:`_ShapeRuntime <yobx.xshape._shape_runtime._ShapeRuntime>`
  — handles *value-as-shape* tracking needed by operators such as ``Shape``,
  ``Gather``, ``Concat``, and ``Slice`` when their outputs feed directly into
  a ``Reshape``.
* :class:`_InferenceRuntime <yobx.xshape._inference_runtime._InferenceRuntime>`
  — walks the graph node by node, dispatching each node to the matching
  per-operator handler in :mod:`yobx.xshape.shape_type_compute` so that
  shapes and types are tracked for every intermediate result.

Two helper classes round out the public API:

* :class:`FunctionOptions <yobx.xbuilder.FunctionOptions>` — controls whether
  (and how) a sub-graph is exported as a reusable ONNX local function.
* :class:`OptimizationOptions <yobx.xbuilder.OptimizationOptions>` — selects
  which optimization passes run inside :meth:`to_onnx
  <yobx.xbuilder.GraphBuilder.to_onnx>`.

Building a graph from scratch
==============================

The simplest workflow is:

1. Construct a :class:`GraphBuilder` with an opset version.
2. Call :meth:`make_tensor_input <yobx.xbuilder.GraphBuilder.make_tensor_input>`
   to declare each graph input.
3. Call :meth:`make_node <yobx.xbuilder.GraphBuilder.make_node>` (or the
   short-hand ``g.op.<OpType>(…)`` syntax) to add operators.
4. Call :meth:`make_tensor_output <yobx.xbuilder.GraphBuilder.make_tensor_output>`
   to declare each graph output.
5. Call :meth:`to_onnx <yobx.xbuilder.GraphBuilder.to_onnx>` to obtain a
   :class:`onnx.ModelProto`.

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    from yobx.xbuilder import GraphBuilder

    TFLOAT = onnx.TensorProto.FLOAT

    # 1. create builder targeting opset 18
    g = GraphBuilder(18, ir_version=10)

    # 2. declare inputs
    g.make_tensor_input("X", TFLOAT, ("batch", "seq", 64), is_dimension=False)
    g.make_tensor_input("W", TFLOAT, (64, 32), is_dimension=False)

    # 3. add a MatMul node via the short-hand op accessor
    result = g.op.MatMul("X", "W")

    # 4. declare the output and export
    g.make_tensor_output(result, elem_type=TFLOAT, shape=("batch", "seq", 32),
                         indexed=False, is_dimension=False)
    model = g.to_onnx()
    print(f"nodes  : {len(model.graph.node)}")
    print(f"opset  : {model.opset_import[0].version}")
    print(f"output : {model.graph.output[0].name}")

Loading an existing model
=========================

Passing an existing :class:`onnx.ModelProto` to the constructor loads it into
the builder so its nodes and initializers can be inspected, modified, or
re-optimized.

.. runpython::
    :showcode:

    import onnx
    import onnx.helper as oh
    from yobx.xbuilder import GraphBuilder

    TFLOAT = onnx.TensorProto.FLOAT

    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Add", ["X", "Y"], ["T"]),
                oh.make_node("Relu", ["T"], ["Z"]),
            ],
            "add_relu",
            [
                oh.make_tensor_value_info("X", TFLOAT, ["batch", 4]),
                oh.make_tensor_value_info("Y", TFLOAT, ["batch", 4]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, ["batch", 4])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )

    g = GraphBuilder(model)
    print("input  shapes:", {n: g.get_shape(n) for n in g.input_names})
    print("nodes        :", [nd.op_type for nd in g.nodes])

Initializers
============

Initializers (model weights and constants) are added with
:meth:`make_initializer <yobx.xbuilder.GraphBuilder.make_initializer>`.
The builder deduplicates small integer arrays automatically: if the same
value is added twice it returns the name of the first occurrence rather than
creating a duplicate node.

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    from yobx.xbuilder import GraphBuilder

    TFLOAT = onnx.TensorProto.FLOAT

    g = GraphBuilder(18, ir_version=10)
    g.make_tensor_input("X", TFLOAT, ("batch", 64), is_dimension=False)

    # Add a weight matrix as an initializer
    W = np.random.randn(64, 32).astype(np.float32)
    w_name = g.make_initializer("W", W, source="example")

    result = g.op.MatMul("X", w_name)
    g.make_tensor_output(result, elem_type=TFLOAT, shape=("batch", 32),
                         indexed=False, is_dimension=False)
    model = g.to_onnx()
    print("initializer name :", list(g.initializers_dict)[0])
    print("initializer shape:", list(g.initializers_dict.values())[0].shape)

Shape and type tracking
=======================

:class:`GraphBuilder` inherits the full
:class:`ShapeBuilder <yobx.xshape.ShapeBuilder>` interface.  Shapes and types
are registered for every intermediate result as nodes are added, and are used
during optimization and for populating ``value_info`` in the exported proto.

Key methods:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``g.set_shape(name, shape)``
     - Register the shape of result ``name``.
       Shape dimensions may be integers (static) or strings (symbolic).
   * - ``g.get_shape(name)``
     - Return the shape as a tuple of integers / symbolic strings.
   * - ``g.has_shape(name)``
     - Return ``True`` if the shape is already known.
   * - ``g.set_type(name, itype)``
     - Register the element type (an ONNX ``TensorProto.*`` integer).
   * - ``g.get_type(name)``
     - Return the element type.
   * - ``g.has_type(name)``
     - Return ``True`` if the element type is already known.
   * - ``g.set_rank(name, rank)``
     - Register only the rank when the full shape is not yet available.
   * - ``g.get_rank(name)``
     - Return the rank as an integer.
   * - ``g.has_rank(name)``
     - Return ``True`` if the rank is known.

Dynamic shapes
==============

When some input dimensions are unknown at graph-construction time, they are
represented as strings (e.g. ``"batch"``, ``"seq"``).  For graphs that are
later exported for dynamic-shape inference with ``torch.export``, the builder
accepts a ``dynamic_shapes`` dictionary that maps input names to per-axis
dimension objects (:class:`torch.export.Dim` or :class:`WrapDim
<yobx.xbuilder.GraphBuilder.WrapDim>`).

:meth:`register_dynamic_objects_from_shape
<yobx.xbuilder.GraphBuilder.register_dynamic_objects_from_shape>`
registers any string dimension names encountered in a shape so that they are
tracked as symbolic dimensions.

.. runpython::
    :showcode:

    import onnx
    from yobx.xbuilder import GraphBuilder

    TFLOAT = onnx.TensorProto.FLOAT

    g = GraphBuilder(18, ir_version=10)
    g.make_tensor_input("X", TFLOAT, ("batch", "seq", 64), is_dimension=False)
    g.make_tensor_input("Y", TFLOAT, ("batch", "seq", 64), is_dimension=False)

    # symbolic dimensions are tracked automatically once shapes are set
    result = g.op.Add("X", "Y")
    g.make_tensor_output(result, elem_type=TFLOAT, shape=("batch", "seq", 64),
                         indexed=False, is_dimension=False)
    model = g.to_onnx()

    out = model.graph.output[0]
    dims = [
        d.dim_param if d.dim_param else d.dim_value
        for d in out.type.tensor_type.shape.dim
    ]
    print("output shape:", dims)

Optimizations
=============

:meth:`to_onnx <yobx.xbuilder.GraphBuilder.to_onnx>` runs a sequence of
optimization passes by default.  The set of passes is controlled by
:class:`OptimizationOptions <yobx.xbuilder.OptimizationOptions>`.

Default passes (in order):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Pass
     - Effect
   * - ``remove_unused``
     - Remove nodes whose outputs are never consumed.
   * - ``constant_folding``
     - Evaluate operators such as ``Transpose``, ``Cast``, ``Reshape``,
       ``Concat``, ``Add``, ``Mul``, etc. when all inputs are constants and
       fold the result into an initializer.
   * - ``remove_identity``
     - Remove ``Identity`` nodes.
   * - ``remove_duplicated_shape``
     - Merge identical ``Shape`` nodes that produce the same result.
   * - ``patterns``
     - Apply user-supplied or built-in fusion patterns (e.g.
       ``"default"`` enables the default set of ONNX-to-ONNX rewrites).

.. runpython::
    :showcode:

    import onnx
    import onnx.helper as oh
    from yobx.xbuilder import GraphBuilder, OptimizationOptions

    TFLOAT = onnx.TensorProto.FLOAT

    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Identity", ["X"], ["X2"]),
                oh.make_node("Relu", ["X2"], ["Z"]),
            ],
            "id_relu",
            [oh.make_tensor_value_info("X", TFLOAT, [None, 4])],
            [oh.make_tensor_value_info("Z", TFLOAT, [None, 4])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )

    opts = OptimizationOptions(remove_identity=True)
    g = GraphBuilder(model, optimization_options=opts)
    optimized = g.to_onnx()
    print("nodes before:", len(model.graph.node))
    print("nodes after :", len(optimized.graph.node))

Local functions
===============

A sub-graph can be exported as a reusable ONNX local function (a
``FunctionProto``) by passing a :class:`FunctionOptions
<yobx.xbuilder.FunctionOptions>` instance to
:meth:`to_onnx <yobx.xbuilder.GraphBuilder.to_onnx>`.

.. runpython::
    :showcode:

    import onnx
    from yobx.xbuilder import GraphBuilder, FunctionOptions

    TFLOAT = onnx.TensorProto.FLOAT

    g = GraphBuilder(18, ir_version=10, as_function=True)
    g.make_tensor_input("X", TFLOAT, ("batch", 64), is_dimension=False)
    r = g.op.Relu("X")
    g.make_tensor_output(r, is_dimension=False, indexed=False)

    func = g.to_onnx(
        function_options=FunctionOptions(
            export_as_function=True,
            name="MyRelu",
            domain="my.domain",
        ),
        inline=False,
    )
    print(type(func).__name__)
    print("function name  :", func.name)
    print("function domain:", func.domain)

Debugging
=========

:class:`GraphBuilder <yobx.xbuilder.GraphBuilder>` respects several
environment variables that help narrow down construction or optimization
problems:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Environment variable
     - Effect
   * - ``ONNXSTOP=<name>``
     - Raises an exception the moment result ``<name>`` is created.
   * - ``ONNXSTOPSHAPE=<name>``
     - Raises an exception the moment result ``<name>`` receives a shape.
   * - ``ONNXSTOPTYPE=<name>``
     - Raises an exception the moment result ``<name>`` receives a type.
   * - ``ONNXSTOPOUTPUT=<name>``
     - Raises an exception the moment a node produces output ``<name>``.
   * - ``ONNXSTOPVALUESHAPE=<name>``
     - Prints extra information for shape-as-value tracking (e.g. inputs
       to ``Reshape``).
   * - ``ONNXCST=1``
     - Prints which constant is being evaluated.
   * - ``ONNXFUNC=1``
     - Prints details when nodes from a local function domain are added.
   * - ``ONNXSHAPECOMPUTE=1``
     - Raises an exception when a shape is missing for a result that should
       have one.
   * - ``NULLSHAPE=1``
     - Raises an exception as soon as a null/empty shape is encountered.
   * - ``ONNXDYNDIM=<name>``
     - Prints a message every time dynamic dimension ``<name>`` is used.
   * - ``PRINTNAME=<name>``
     - Prints a message every time a node producing ``<name>`` is added.

In addition,
:meth:`get_debug_msg <yobx.xshape.shape_builder_impl.BasicShapeBuilder.get_debug_msg>`
returns a detailed text dump of the builder's internal state (known shapes,
types, ranks, constants, and node list) which can be printed or logged whenever
an assertion fails.

:meth:`pretty_text <yobx.xbuilder.GraphBuilder.pretty_text>` returns a
human-readable representation of the whole graph (inputs, initializers, nodes,
outputs) and is useful for quick visual inspection:

.. runpython::
    :showcode:

    import onnx
    import onnx.helper as oh
    from yobx.xbuilder import GraphBuilder

    TFLOAT = onnx.TensorProto.FLOAT

    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Add", ["X", "Y"], ["T"]),
                oh.make_node("Relu", ["T"], ["Z"]),
            ],
            "add_relu",
            [
                oh.make_tensor_value_info("X", TFLOAT, ["batch", 4]),
                oh.make_tensor_value_info("Y", TFLOAT, ["batch", 4]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, ["batch", 4])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )

    g = GraphBuilder(model)
    print(g.pretty_text())
