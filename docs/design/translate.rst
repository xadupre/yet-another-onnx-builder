.. _l-design-translate:

=========
Translate
=========

:func:`translate <yobx.translate.translate>` converts an existing
:class:`onnx.ModelProto` into Python source code that, when executed,
recreates the same graph.  This is useful for:

* **debugging** — turn an opaque ``.onnx`` file into readable Python so you
  can inspect or modify individual nodes;
* **sharing** — paste a self-contained snippet into a bug report or a unit
  test without attaching a binary file;
* **migration** — port a model written with one builder API to another.

Architecture overview
=====================

The translation is split into two independent layers:

.. code-block:: text

    ModelProto / GraphProto / FunctionProto
           │
           ▼
       Translator          ← walks the proto, fires events
           │
           ▼
        Emitter            ← converts events into code strings

:class:`Translator <yobx.translate.translate.Translator>` knows the
structure of ONNX protos; it never emits text directly.  Instead it fires a
sequence of typed events (``BEGIN_GRAPH``, ``INITIALIZER``, ``NODE``,
``END_GRAPH``, …) defined by
:class:`EventType <yobx.translate.base_emitter.EventType>`.

:class:`BaseEmitter <yobx.translate.base_emitter.BaseEmitter>` receives those
events and returns lists of code strings.  Each concrete emitter subclass
overrides the ``_emit_*`` methods to produce code in the desired target API.
The default implementation raises :class:`NotImplementedError` for every
event, so any missing override is caught early.

.. runpython::
    :showcode:

    from yobx.translate.base_emitter import EventType

    for name, value in sorted(EventType.__members__.items(), key=lambda kv: kv[1]):
        print(f"{value:2d}  {name}")


Available emitters
==================

Four concrete emitters are shipped:

+--------------------------------------------------+------------------------------------+
| Class                                            | Output API                         |
+==================================================+====================================+
| :class:`~yobx.translate.inner_emitter.           | :mod:`onnx.helper` (``oh.make_*``) |
| InnerEmitter`                                    |                                    |
+--------------------------------------------------+------------------------------------+
| :class:`~yobx.translate.inner_emitter.           | Same as above, but replaces large  |
| InnerEmitterShortInitializer`                    | initializers (> 16 elements) with  |
|                                                  | ``np.random.randn(…)``             |
+--------------------------------------------------+------------------------------------+
| :class:`~yobx.translate.light_emitter.           | ``onnx_array_api.light_api``       |
| LightEmitter`                                    | fluent chain                       |
+--------------------------------------------------+------------------------------------+
| :class:`~yobx.translate.builder_emitter.         | ``onnx_array_api.graph_api.        |
| BuilderEmitter`                                  | GraphBuilder``                     |
+--------------------------------------------------+------------------------------------+

The four APIs are exposed through the single convenience function
:func:`translate <yobx.translate.translate>`:

.. runpython::
    :showcode:

    import onnx.helper as oh
    import onnx
    from yobx.translate import translate

    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Add", ["X", "Y"], ["T"]),
                oh.make_node("Relu", ["T"], ["Z"]),
            ],
            "add_relu",
            [
                oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, 4]),
                oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [None, 4]),
            ],
            [oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [None, 4])],
        ),
        opset_imports=[oh.make_opsetid("", 17)],
        ir_version=9,
    )

    code = translate(model, api="onnx")
    print(code)


Event sequence
==============

For a :class:`~onnx.ModelProto` the ``Translator`` fires events in this
order:

1. ``START`` — emitter initialises its state (opset list, ir_version).
2. ``BEGIN_GRAPH`` — emitter declares graph-level containers.
3. ``INITIALIZER`` × N — one event per initializer tensor.
4. ``BEGIN_SIGNATURE`` — separator before inputs.
5. ``INPUT`` × N — one event per graph input.
6. ``END_SIGNATURE`` — separator after inputs.
7. ``NODE`` × N — one event per operator node.
8. ``BEGIN_RETURN`` — separator before outputs.
9. ``OUTPUT`` × N — one event per graph output.
10. ``END_RETURN`` — separator after outputs.
11. ``END_GRAPH`` — emitter assembles the graph object.
12. Optionally, for each local function: ``BEGIN_FUNCTION`` … ``END_FUNCTION``.
13. ``TO_ONNX_MODEL`` — emitter assembles the final ``ModelProto``.

``FunctionProto`` inputs and outputs use the ``FUNCTION_INPUT`` /
``FUNCTION_OUTPUT`` variants and the sequence is bookended by
``BEGIN_FUNCTION`` / ``END_FUNCTION`` instead of ``BEGIN_GRAPH`` /
``END_GRAPH``.


InnerEmitter
============

:class:`~yobx.translate.inner_emitter.InnerEmitter` produces standard
:mod:`onnx.helper` code.  Every initializer is written as an exact
``np.array(…)`` literal.

:class:`~yobx.translate.inner_emitter.InnerEmitterShortInitializer` inherits
from :class:`~yobx.translate.inner_emitter.InnerEmitter` and overrides only
``_emit_initializer``: tensors with **more than 16 elements** are replaced by
a ``np.random.randn(…)`` or ``np.random.randint(…)`` call, so the snippet
stays readable for large weight matrices.

.. runpython::
    :showcode:

    import numpy as np
    import onnx.helper as oh
    import onnx.numpy_helper as onh
    import onnx
    from yobx.translate import translate

    big_w = onh.from_array(np.random.randn(8, 5).astype(np.float32), name="W")
    model = oh.make_model(
        oh.make_graph(
            [oh.make_node("MatMul", ["X", "W"], ["Z"])],
            "mm",
            [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, 8])],
            [oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [None, 5])],
            [big_w],
        ),
        opset_imports=[oh.make_opsetid("", 17)],
    )
    full  = translate(model, api="onnx")
    short = translate(model, api="onnx-short")
    print(f"full  : {len(full):>5} chars")
    print(f"short : {len(short):>5} chars")
    # Show only the initializer line from the short version
    for line in short.splitlines():
        if "randn" in line or "randint" in line:
            print("short initializer:", line.strip())
            break


LightEmitter
============

:class:`~yobx.translate.light_emitter.LightEmitter` generates a fluent
method chain for the ``onnx_array_api.light_api`` (``start(…).vin(…).…``).
The chain is either written as a multi-line indented block (default) or
collapsed to a single ``.``-joined line when ``single_line=True``.

.. runpython::
    :showcode:

    import onnx.helper as oh
    import onnx
    from yobx.translate import translate

    model = oh.make_model(
        oh.make_graph(
            [oh.make_node("Relu", ["X"], ["Z"])],
            "relu",
            [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, 4])],
            [oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [None, 4])],
        ),
        opset_imports=[oh.make_opsetid("", 17)],
    )
    print(translate(model, api="light"))


BuilderEmitter
==============

:class:`~yobx.translate.builder_emitter.BuilderEmitter` generates code that
uses ``onnx_array_api.graph_api.GraphBuilder``.  The graph body is wrapped
in a Python function named after the ONNX graph, and a small driver block
constructs the ``GraphBuilder``, calls the function, and finalises the model
with ``g.to_onnx()``.

.. runpython::
    :showcode:

    import onnx.helper as oh
    import onnx
    from yobx.translate import translate

    model = oh.make_model(
        oh.make_graph(
            [oh.make_node("Relu", ["X"], ["Z"])],
            "relu",
            [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, 4])],
            [oh.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [None, 4])],
        ),
        opset_imports=[oh.make_opsetid("", 17)],
    )
    print(translate(model, api="builder"))


Round-trip verification
=======================

The generated ``"onnx"`` code is fully self-contained; running it recreates
the original model:

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    import onnx.helper as oh
    from yobx.translate import translate, translate_header

    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Transpose", ["X"], ["Y"], perm=[1, 0]),
            ],
            "transpose",
            [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3, 4])],
            [oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [4, 3])],
        ),
        opset_imports=[oh.make_opsetid("", 17)],
        ir_version=9,
    )

    code = translate(model, api="onnx")
    header = translate_header("onnx")
    ns: dict = {}
    exec(compile(header + "\n" + code, "<translate>", "exec"), ns)  # noqa: S102
    recreated = ns["model"]

    assert len(recreated.graph.node) == len(model.graph.node)
    print("nodes       :", len(recreated.graph.node))
    print("opset       :", recreated.opset_import[0].version)
    print("ir_version  :", recreated.ir_version)
    print("Round-trip  : OK")

.. seealso::

    :ref:`l-plot-translate-comparison` — sphinx-gallery example that builds
    a ``Gemm → Relu`` model, translates it with all four APIs, prints the
    generated snippets, verifies the round-trip, and plots a bar chart of
    generated code sizes.
