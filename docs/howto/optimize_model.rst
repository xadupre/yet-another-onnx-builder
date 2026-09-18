.. _l-howto-optimize-model:

Optimize an existing ONNX model
===============================

This page answers common *"how do I…"* questions for optimizing an
existing :class:`onnx.ModelProto` with the native ``onnx-light``
optimizer.  The underlying engine is ``GraphGraph``; the native bridge
classes are :class:`~yobx.builder.onnxlight.OnnxLightGraphBuilder` and
:class:`~yobx.builder.onnxlight.OnnxLightOptimizationOptions`.

How to optimize a model with the default patterns
--------------------------------------------------

Load the model into :class:`~yobx.builder.onnxlight.OnnxLightGraphBuilder`
and call :meth:`to_onnx <yobx.builder.onnxlight.OnnxLightGraphBuilder.to_onnx>`
with ``optimize=True``.  The default native pattern catalogue is applied
automatically.

.. runpython::
    :showcode:

    from onnx_light import onnx
    from onnx_light.onnx import helper
    from onnx_light.onnx_core.optimization import standard_pattern_names
    from yobx.builder.onnxlight import OnnxLightGraphBuilder

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

    print("before:", [node.op_type for node in model.graph.node])
    builder = OnnxLightGraphBuilder(model)
    optimized = builder.to_onnx()
    print("after :", [node.op_type for node in optimized.proto.graph.node])

How to choose which native patterns to apply
--------------------------------------------

Use :class:`~yobx.builder.onnxlight.OnnxLightOptimizationOptions` to choose
the native registered pattern names.  The selection rules are:

* ``patterns=None`` or ``patterns="default"`` — use the wheel's standard
  registered patterns.
* ``patterns=[]`` — disable pattern rewrites.
* any other string or sequence — must contain exact native pattern names such
  as ``"TransposeTranspose"``.

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
    opt_onx = builder.to_onnx()
    print([node.op_type for node in opt_onx.proto.graph.node])

The native bridge validates the names eagerly and raises an error when a
legacy Python pattern group or a custom pattern object is supplied.

How to inspect what the optimizer did
-------------------------------------

Calling :meth:`to_onnx <yobx.xbuilder.GraphBuilder.to_onnx>` with
``return_optimize_report=True`` returns an artifact whose report contains one
row per native rewrite, with timings and the number of nodes added or removed.
The rows can be aggregated with :mod:`pandas` to get a per-pattern summary.

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
        print(agg[(agg["added"] > 0) | (agg["removed"] > 0)])

The same information is also available on the native ``GraphGraph`` report
returned by the bridge classes.

Next steps
----------

* :ref:`l-design-graph-builder` — build and optimize ONNX graphs programmatically.
* :ref:`l-design-shape` — symbolic shape expressions for dynamic shapes.
* :ref:`l-design-translate` — translate ONNX graphs back to Python code.
* :doc:`api/index` — full API reference.
