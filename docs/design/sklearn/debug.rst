.. _l-design-sklearn-debug-env-vars:

====================================
Debugging with Environment Variables
====================================

Debugging depends on the builder used to build the ONNX model.
The default option is :class:`~yobx.xbuilder.GraphBuilder`
and offers debugging options through environment variables.

GraphBuilder Environment Variables
===================================

The following environment variables are recognized by
:class:`~yobx.xbuilder.GraphBuilder` (see :ref:`l-graphbuilder-debugging-env`
for the full documentation):

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
:meth:`get_debug_msg <yobx.xshape.shape_builder_impl.BasicShapeBuilder.get_debug_msg>`
returns a detailed text dump of the builder's internal state (known shapes,
types, ranks, constants, and node list).

Pattern Optimization Environment Variables
==========================================

When the ONNX graph is optimized with the pattern optimizer, additional
environment variables can help trace which patterns are applied and
narrow down optimization problems
(see :ref:`l-design-pattern-optimizer-debugging` for the full documentation):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Environment variable
     - Effect
   * - ``LOG_PATTERN_OPTIMIZE=10``
     - Sets the verbosity level for all patterns.  ``10`` gives the most
       detailed output.
   * - ``PATTERN=<ClassName>``
     - Increases the verbosity to ``10`` for one or more specific patterns
       (comma-separated). Useful to focus on a single pattern.
   * - ``<ClassName>=10``
     - Sets the verbosity for a single pattern whose class name matches the
       variable name (e.g. ``ReshapeReshapePattern=10``).
   * - ``DROPPATTERN=<ClassName>``
     - Comma-separated list of pattern class names to exclude from the
       optimizer. Useful to bisect which pattern is causing a wrong result.
   * - ``DUMPPATTERNS=<folder>``
     - When set to a folder path, the optimizer writes every matched subgraph
       and its replacement to that folder.
   * - ``PATTERNNOREMOVE=<name>``
     - Raises an exception if an optimization step removes the named result
       from the graph.
   * - ``PATTERNSTEP=1``
     - Runs one optimization step at a time.
