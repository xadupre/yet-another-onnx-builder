yobx.xbuilder
=============

API for the graph builder used to construct and optimize ONNX graphs.

.. toctree::
    :maxdepth: 1
    :caption: modules

    function_options
    graph_builder
    graph_builder_opset
    infer_shapes_options
    optimization_options
    order_optim

GraphBuilder
++++++++++++

.. autoclass:: yobx.xbuilder.GraphBuilder
    :members:
    :no-undoc-members:
    :exclude-members: WrapSym, WrapDim, VirtualTensor

GraphBuilderProtocol
++++++++++++++++++++

.. autoclass:: yobx.typing.GraphBuilderProtocol
    :members:
    :no-undoc-members:

GraphBuilderExtendedProtocol
++++++++++++++++++++++++++++

.. autoclass:: yobx.typing.GraphBuilderExtendedProtocol
    :members:
    :no-undoc-members:

FunctionOptions
+++++++++++++++

.. autoclass:: yobx.xbuilder.FunctionOptions
    :members:
    :no-undoc-members:

InferShapesOptions
++++++++++++++++++

.. autoclass:: yobx.xbuilder.InferShapesOptions
    :members:
    :no-undoc-members:

OptimizationOptions
+++++++++++++++++++

.. autoclass:: yobx.xbuilder.OptimizationOptions
    :members:
    :no-undoc-members:

OrderAlgorithm
++++++++++++++

.. autoclass:: yobx.xbuilder.OrderAlgorithm
    :members:
    :no-undoc-members:

Intermediate Classes
++++++++++++++++++++

InitializerInfo
---------------

.. autoclass:: yobx.xbuilder._initializer_info.InitializerInfo
    :members:
    :no-undoc-members:

VirtualTensor
-------------

.. autoclass:: yobx.xbuilder._virtual_tensor.VirtualTensor
    :members:
    :no-undoc-members:

WrapDim
-------

.. autoclass:: yobx.xbuilder._wrap_dim.WrapDim
    :members:
    :no-undoc-members:

WrapSym
-------

.. autoclass:: yobx.xbuilder._wrap_sym.WrapSym
    :members:
    :no-undoc-members:
