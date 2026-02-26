
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
