
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
    from yobx.xshape.shape_builder_impl import BasicShapeBuilder

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
