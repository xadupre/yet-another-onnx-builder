GraphBuilder to build and optimize ONNX Models
==============================================

This section covers the tools for building and optimizing ONNX computation graphs
programmatically. It includes the :class:`GraphBuilder <yobx.xbuilder.GraphBuilder>` class
for constructing ONNX graphs from scratch, shape inference mechanisms for tracking tensor
dimensions throughout the graph, and a pattern-based optimizer for rewriting and simplifying
graphs.

.. toctree::
   :maxdepth: 1

   shape
   cost
   graph_builder
   optimizer
   optimizer_patterns
