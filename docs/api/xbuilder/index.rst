yobx.xbuilder
=============

API for the graph builder used to construct and optimize ONNX graphs.

``GraphBuilder`` and ``OptimizationOptions`` expose the native
``onnx-light`` adapter. The historical ``yobx.xbuilder.graph_builder`` and
``yobx.xbuilder.optimization_options`` imports resolve to the same classes.
Graph construction, shape inference and pattern optimization do not fall back
to the former Python engines.

Native optimization options select registered pattern names through
``patterns`` and limit iterations through ``max_iter``. Python pattern objects,
legacy pattern groups such as ``"default+onnxruntime"``, and unsupported
legacy options raise explicit errors.

The historical ``graph_builder_opset.Opset`` import resolves to the native
adapter too. The Python ``OrderOptimization`` engine and ``OrderAlgorithm``
selection have been retired; shape-node ordering uses the native builder's
``move_shape_and_size_nodes`` operation.

.. toctree::
    :maxdepth: 1
    :caption: modules

    builder_stats_helper
    function_options
    graph_builder
    graph_builder_opset
    infer_shapes_options
    optimization_options

GraphBuilder
++++++++++++

.. autoclass:: yobx.xbuilder.GraphBuilder
    :members:
    :no-undoc-members:
    :exclude-members: WrapSym, WrapDim, VirtualTensor

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
