ONNX translation
================

Model-to-source translation is implemented by :mod:`onnx_light.tools`, not by
YOBX. This keeps graph serialization and native ``GraphBuilder`` generation
next to the ONNX representation they target.

.. runpython::
    :showcode:

    import onnx_light.onnx as onnx
    import onnx_light.onnx.helper as oh
    from onnx_light.tools import translate

    model = oh.make_model(
        oh.make_graph(
            [oh.make_node("Relu", ["X"], ["Y"])],
            "relu",
            [oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, 3])],
            [oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [None, 3])],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
    )
    print(translate(model, api="onnx-compact"))

The supported output APIs are ``"onnx-compact"``, ``"builder"``, and
``"cpp"``. Mermaid source is produced separately by
:func:`onnx_light.tools.to_mermaid`.
