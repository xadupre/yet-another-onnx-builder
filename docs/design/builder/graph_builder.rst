
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

Protocol API
============

:class:`GraphBuilder <yobx.xbuilder.GraphBuilder>` satisfies a hierarchy of
protocols defined in :mod:`yobx.typing`.  Callers that do not need the
full concrete class should type-annotate against the narrowest protocol that
covers the methods they actually use:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Protocol
     - Scope
   * - :class:`GraphBuilderProtocol <yobx.typing.GraphBuilderProtocol>`
     - Core construction API: inputs, outputs, initializers, nodes,
       opset registration, type/shape/sequence accessors, and
       :meth:`~yobx.typing.GraphBuilderProtocol.to_onnx`.
   * - :class:`GraphBuilderExtendedProtocol <yobx.typing.GraphBuilderExtendedProtocol>`
     - Extends the core protocol with ``main_opset``,
       :attr:`~yobx.typing.GraphBuilderExtendedProtocol.op` (the
       :class:`~yobx.typing.OpsetProtocol` helper),
       :meth:`~yobx.typing.GraphBuilderExtendedProtocol.set_type_shape_unary_op`,
       constant queries, and :meth:`~yobx.typing.GraphBuilderExtendedProtocol.get_debug_msg`.
       Required by the :mod:`yobx.sklearn` converters.
   * - :class:`GraphBuilderTorchProtocol <yobx.typing.GraphBuilderTorchProtocol>`
     - Extends :class:`~yobx.typing.GraphBuilderExtendedProtocol` with the
       full torch-exporter surface: rank helpers, device helpers, dynamic-shape
       helpers, sub-builder / local-function support, and miscellaneous
       utilities used by :class:`~yobx.torch.interpreter.FxGraphInterpreter`.

The :attr:`~yobx.typing.GraphBuilderExtendedProtocol.op` property returns an
object that satisfies :class:`OpsetProtocol <yobx.typing.OpsetProtocol>`,
which resolves ``g.op.Add(x, y)``-style attribute-access dispatch to ONNX
node creation.

.. _builder-api-make:

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
    from yobx._onnx_shim import onnx
    from yobx.helpers.onnx_helper import pretty_onnx
    from yobx.xbuilder import GraphBuilder

    TFLOAT = onnx.TensorProto.FLOAT

    # 1. create builder targeting opset 18
    g = GraphBuilder(18, ir_version=10)

    # 2. declare inputs
    g.make_tensor_input("X", TFLOAT, ("batch", "seq", 64))
    g.make_tensor_input("W", TFLOAT, (64, 32))

    # 3. add a MatMul node via the short-hand op accessor
    result = g.op.MatMul("X", "W")

    # 4. declare the output and export
    g.make_tensor_output(result, elem_type=TFLOAT, shape=("batch", "seq", 32),
                            indexed=False)
    model = g.to_onnx()
    print(f"nodes  : {len(model.graph.node)}")
    print(f"opset  : {model.opset_import[0].version}")
    print(f"output : {model.graph.output[0].name}")
    print(pretty_onnx(model))

Loading an existing model
=========================

Passing an existing :class:`onnx.ModelProto` to the constructor loads it into
the builder so its nodes and initializers can be inspected, modified, or
re-optimized.

.. runpython::
    :showcode:

    from yobx._onnx_shim import onnx
    import onnx_light.onnx.helper as oh
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
    print("nodes        :", [node.op_type for node in g.nodes])

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
    from yobx._onnx_shim import onnx
    from yobx.xbuilder import GraphBuilder

    TFLOAT = onnx.TensorProto.FLOAT

    g = GraphBuilder(18, ir_version=10)
    g.make_tensor_input("X", TFLOAT, ("batch", 64))

    # Add a weight matrix as an initializer
    W = np.random.randn(64, 32).astype(np.float32)
    w_name = g.make_initializer("W", W, source="example")

    result = g.op.MatMul("X", w_name)
    g.make_tensor_output(result, elem_type=TFLOAT, shape=("batch", 32),
                         indexed=False)
    model = g.to_onnx()
    print("initializer name :", list(g.initializers_dict)[0])
    print("initializer shape:", tuple(g.initializers_dict[w_name].dims))

.. _builder-api:

Shape and type tracking
=======================

:class:`GraphBuilder` inherits the full
:class:`ShapeBuilder <yobx.xshape.ShapeBuilder>` interface.  Shapes and types
are registered for every intermediate result as nodes are added, and are used
during optimization and for populating ``value_info`` in the exported proto.
See :ref:`l-design-expected-api`.

Native optimization
===================

.. _l-design-graph-builder-optimization:

``GraphBuilder`` delegates optimization to the native ``onnx-light``
``GraphGraph`` engine through :class:`~yobx.xbuilder.OptimizationOptions`
(published as :class:`~yobx.builder.onnxlight.OnnxLightOptimizationOptions`
in the native bridge).  ``patterns=None`` or ``"default"`` selects the
wheel's registered standard patterns, an empty sequence disables pattern
rewrites, and every other value must be an exact registered pattern name.
Python pattern classes, legacy pattern groups, and historical Python
registries are no longer supported.

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

    from yobx._onnx_shim import onnx
    from yobx.xbuilder import GraphBuilder

    TFLOAT = onnx.TensorProto.FLOAT

    g = GraphBuilder(18, ir_version=10)
    g.make_tensor_input("X", TFLOAT, ("batch", "seq", 64))
    g.make_tensor_input("Y", TFLOAT, ("batch", "seq", 64))

    # symbolic dimensions are tracked automatically once shapes are set
    result = g.op.Add("X", "Y")
    g.make_tensor_output(result, elem_type=TFLOAT, shape=("batch", "seq", 64),
                         indexed=False)
    model = g.to_onnx()

    out = model.graph.output[0]
    dims = [
        d.dim_param if d.dim_param else d.dim_value
        for d in out.type.tensor_type.shape.dim
    ]
    print("output shape:", dims)

Optimizations
=============

:meth:`to_onnx <yobx.xbuilder.GraphBuilder.to_onnx>` delegates optimization to
the native ``onnx-light`` ``GraphGraph`` engine through
:class:`OptimizationOptions <yobx.xbuilder.OptimizationOptions>`.

Only the native pattern selection knobs are exposed here:

* ``patterns=None`` or ``patterns="default"`` selects the wheel's registered
  standard patterns.
* ``patterns=[]`` disables rewrites.
* any other value must be an exact registered pattern name such as
  ``"TransposeTranspose"`` or ``"MulMulMatMul"``.
* ``max_iter`` limits the number of native rewrite iterations.

Python pattern classes, legacy pass flags, and historical reorder /
constant-folding options are not supported by the native bridge.
With ``onnx-light>=0.1.25``, dependency ordering and lifetime metadata are
finalized by the native engine; the adapter does not reorder serialized nodes
or reconstruct the optimized graph to repair them. Local function imports,
nested calls, and referenced attributes likewise use the native builder
without replaying typed function bodies in Python.

.. runpython::
    :showcode:

    from onnx_light import onnx
    from onnx_light.onnx import helper
    from onnx_light.onnx_core.optimization import standard_pattern_names
    from yobx.builder.onnxlight import (
        OnnxLightGraphBuilder,
        OnnxLightOptimizationOptions,
    )

    assert "TransposeTranspose" in standard_pattern_names()

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Transpose", ["X"], ["T"], perm=[1, 0]),
                helper.make_node("Transpose", ["T"], ["Y"], perm=[1, 0]),
            ],
            "transpose_transpose",
            [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 3])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=8,
    )

    builder = OnnxLightGraphBuilder(
        model,
        optimization_options=OnnxLightOptimizationOptions(patterns=["TransposeTranspose"]),
    )
    optimized = builder.to_onnx()
    print("nodes before:", len(model.graph.node))
    print("nodes after :", len(optimized.proto.graph.node))

Optimization report
===================

Passing ``return_optimize_report=True`` to
:meth:`to_onnx <yobx.xbuilder.GraphBuilder.to_onnx>` returns an
:class:`~yobx.container.ExportArtifact` whose ``report.stats`` list contains one
entry per native rewrite.  Each entry records the rewrite name, the number of
nodes added or removed, and the time spent in that rewrite.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Description
   * - ``pattern``
     - Native rewrite name such as ``"TransposeTranspose"`` or
       ``"MulMulMatMul"``.
   * - ``added``
     - Number of nodes added by the rewrite.
   * - ``removed``
     - Number of nodes removed by the rewrite.
   * - ``time_in``
     - Wall-clock time spent in this rewrite (seconds).

The list can be converted to a :class:`pandas.DataFrame` for quick
exploration:

.. runpython::
    :showcode:

    import pandas
    from onnx_light import onnx
    from onnx_light.onnx import helper
    from yobx.builder.onnxlight import OnnxLightGraphBuilder

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["X"], ["X2"]),
                helper.make_node("Transpose", ["X2"], ["T"], perm=[1, 0]),
                helper.make_node("Transpose", ["T"], ["Y"], perm=[1, 0]),
            ],
            "demo",
            [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3, 4])],
            [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [3, 4])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=8,
    )

    art = OnnxLightGraphBuilder(model).to_onnx(return_optimize_report=True)

    df = pandas.DataFrame(art.report.stats)
    if df.empty:
        print("no rewrites")
    else:
        print(df[["pattern", "added", "removed", "time_in"]].to_string(index=False))
    print(f"\nnodes before: {len(model.graph.node)}")
    print(f"nodes after : {len(art.proto.graph.node)}")

The report can be aggregated by rewrite name:

.. runpython::
    :showcode:

    import pandas
    from onnx_light import onnx
    from onnx_light.onnx import helper
    from yobx.builder.onnxlight import OnnxLightGraphBuilder

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["X"], ["X2"]),
                helper.make_node("Transpose", ["X2"], ["T"], perm=[1, 0]),
                helper.make_node("Transpose", ["T"], ["Y"], perm=[1, 0]),
            ],
            "demo",
            [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3, 4])],
            [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [3, 4])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=8,
    )

    art = OnnxLightGraphBuilder(model).to_onnx(return_optimize_report=True)

    df = pandas.DataFrame(art.report.stats)
    if df.empty:
        print("no rewrites")
    else:
        for c in ["added", "removed"]:
            df[c] = df[c].fillna(0).astype(int)
        agg = df.groupby("pattern")[["added", "removed", "time_in"]].sum()
        agg = agg[(agg["added"] > 0) | (agg["removed"] > 0)].sort_values(
            "removed", ascending=False
        )
        print(agg.to_string())

Local functions
===============

A sub-graph can be exported as a reusable ONNX local function (a
``FunctionProto``) by passing a :class:`FunctionOptions
<yobx.xbuilder.FunctionOptions>` instance to
:meth:`to_onnx <yobx.xbuilder.GraphBuilder.to_onnx>`.

.. runpython::
    :showcode:

    from yobx._onnx_shim import onnx
    from yobx.xbuilder import GraphBuilder, FunctionOptions

    TFLOAT = onnx.TensorProto.FLOAT

    g = GraphBuilder(18, ir_version=10, as_function=True)
    g.make_tensor_input("X", TFLOAT, ("batch", 64))
    r = g.op.Relu("X")
    g.make_tensor_output(r, indexed=False)

    func = g.to_onnx(
        function_options=FunctionOptions(
            export_as_function=True,
            name="MyRelu",
            domain="my.domain",
        ),
        inline=False,
    )
    proto = func.proto
    print(type(proto).__name__)
    print("function name  :", proto.name)
    print("function domain:", proto.domain)

.. _l-graphbuilder-debugging-env:

Debugging GraphBuilder with Environment Variables
=================================================

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
:meth:`get_debug_msg <yobx.xshape.NativeShapeInference.get_debug_msg>`
returns a detailed text dump of the builder's internal state (known shapes,
types, ranks, constants, and node list) which can be printed or logged whenever
an assertion fails.

:meth:`pretty_text <yobx.xbuilder.GraphBuilder.pretty_text>` returns a
human-readable representation of the whole graph (inputs, initializers, nodes,
outputs) and is useful for quick visual inspection:

.. runpython::
    :showcode:

    from yobx._onnx_shim import onnx
    import onnx_light.onnx.helper as oh
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

.. seealso::

    :class:`GraphBuilderProtocol <yobx.typing.GraphBuilderProtocol>`,
    :class:`GraphBuilderExtendedProtocol <yobx.typing.GraphBuilderExtendedProtocol>`,
    :class:`GraphBuilderTorchProtocol <yobx.typing.GraphBuilderTorchProtocol>`,
    :class:`OpsetProtocol <yobx.typing.OpsetProtocol>` —
    formal protocol interfaces satisfied by
    :class:`GraphBuilder <yobx.xbuilder.GraphBuilder>`.

    :ref:`l-design-expected-api` — the full list of methods and attributes
    every graph builder must expose for use with :func:`~yobx.sklearn.to_onnx`.

    :ref:`l-design-graph-builder-optimization` — how native optimization
    selects registered pattern names through :class:`~yobx.xbuilder.OptimizationOptions`.
