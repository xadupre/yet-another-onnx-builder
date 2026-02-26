.. _l-design-helpers:

===================
Interesting Helpers
===================

ONNX Serialization of Nested Structured with Tensors
====================================================

The main goal is to serialize any Python structure into ONNX
format. This relies on :class:`MiniOnnxBuilder <yobx.helpers.mini_onnx_builder.MiniOnnxBuilder>`.
Example :ref:`l-plot-mini-onnx-builder` shows an example.

ONNX Graph Visualization
========================

:func:`to_dot <yobx.helpers.dot_helper.to_dot>` converts an
:class:`onnx.ModelProto` into a `DOT <https://graphviz.org/doc/info/lang.html>`_
string suitable for rendering with `Graphviz <https://graphviz.org/>`_.

.. runpython::
    :showcode:

    import numpy as np
    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper as onh
    from yobx.helpers.dot_helper import to_dot

    TFLOAT = onnx.TensorProto.FLOAT
    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Add", ["X", "Y"], ["added"]),
                oh.make_node("MatMul", ["added", "W"], ["mm"]),
                oh.make_node("Relu", ["mm"], ["Z"]),
            ],
            "add_matmul_relu",
            [
                oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", 4]),
                oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", 4]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, ["batch", "seq", 2])],
            [
                onh.from_array(
                    np.zeros((4, 2), dtype=np.float32),
                    name="W",
                )
            ],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )
    dot = to_dot(model)
    print(dot[:200], "...")

The resulting DOT source can be rendered directly in the documentation with the
``gdot`` directive from :epkg:`sphinx-runpython`:

.. gdot::
    :script: DOT-SECTION
    :process:

    import numpy as np
    import onnx
    import onnx.helper as oh
    import onnx.numpy_helper as onh
    from yobx.helpers.dot_helper import to_dot

    TFLOAT = onnx.TensorProto.FLOAT
    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Add", ["X", "Y"], ["added"]),
                oh.make_node("MatMul", ["added", "W"], ["mm"]),
                oh.make_node("Relu", ["mm"], ["Z"]),
            ],
            "add_matmul_relu",
            [
                oh.make_tensor_value_info("X", TFLOAT, ["batch", "seq", 4]),
                oh.make_tensor_value_info("Y", TFLOAT, ["batch", "seq", 4]),
            ],
            [oh.make_tensor_value_info("Z", TFLOAT, ["batch", "seq", 2])],
            [
                onh.from_array(
                    np.zeros((4, 2), dtype=np.float32),
                    name="W",
                )
            ],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=10,
    )
    dot = to_dot(model)
    print("DOT-SECTION", dot)

.. seealso::

    :ref:`l-plot-dot-graph` — sphinx-gallery example demonstrating
    :func:`to_dot <yobx.helpers.dot_helper.to_dot>` on a simple hand-built model.

