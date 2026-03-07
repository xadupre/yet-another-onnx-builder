.. _l-design-sklearn-debug-env-vars:

============================
Debugging with Environment Variables
============================

When a conversion produces unexpected results or raises an error, a set of
environment variables can be used to get more diagnostic output **without
modifying any source code**.  All variables are read once at start-up — set
them in your shell (or prepend them to the ``python`` command) before running
the failing script.

Graph-builder variables
=======================

The variables below are handled by
:class:`GraphBuilder <yobx.xbuilder.GraphBuilder>` and
:class:`BasicShapeBuilder <yobx.xshape.BasicShapeBuilder>`.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Effect
   * - ``ONNXSTOP=<name>``
     - Raises an exception the first time variable *name* receives a type.
       Useful to locate the node that first assigns a type to a result.
   * - ``ONNXSTOPSHAPE=<name>``
     - Raises an exception the first time variable *name* receives a shape.
   * - ``ONNXSTOPTYPE=<name>``
     - Raises an exception the first time variable *name* receives a type
       (same as ``ONNXSTOP``; kept for symmetry with ``ONNXSTOPSHAPE``).
   * - ``ONNXSTOPSEQUENCE=<name>``
     - Raises an exception when *name* is assigned as a sequence variable.
   * - ``ONNXSTOPOUTPUT=<name>``
     - Raises an exception when a node emits the output *name*.
   * - ``ONNXSTOPVALUESHAPE=<name>``
     - Prints extra information every time the *value-shape* of *name* is
       updated (helps trace ``Shape``/``Gather`` constant-folding).
   * - ``ONNXCST=1``
     - Prints the name of every constant that is requested during the build.
   * - ``ONNXSHAPECOMPUTE=1``
     - Raises an exception the first time a shape is required but not known.
   * - ``ONNXCONSTANTFOLD=1``
     - Enables verbose output for constant-folding steps.
   * - ``ONNXFOLDNOT=1``
     - Enables verbose output for the *NOT* constant-folding path.
   * - ``ONNXFUNC=1``
     - Prints debug messages related to local ONNX functions.
   * - ``ONNXDYNDIM=<names>``
     - Comma-separated list of dynamic-dimension names.  Raises an exception
       whenever any listed dimension is used, which helps trace how a
       particular symbolic dimension propagates through the graph.
   * - ``ONNXNODETYPE=<op_type>``
     - Filters console output to nodes of the given operator type.
   * - ``ONNXQUIET=1``
     - Suppresses most console output (reverses the effect of verbose modes).
   * - ``NULLSHAPE=1``
     - Raises an exception the first time a null/empty shape is encountered.
   * - ``PRINTNAME=<names>``
     - Comma-separated list of result names.  Prints every node that produces
       or consumes any listed name.
   * - ``NOTORCH=1``
     - Disables all :epkg:`torch` usage even when PyTorch is installed.

Pattern-optimizer variables
============================

The variables below are handled by
:class:`GraphBuilderPatternOptimization <yobx.xoptim.GraphBuilderPatternOptimization>`.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Effect
   * - ``PATTERN=<names>``
     - Comma-separated list of pattern class names.  Increases verbosity for
       each listed pattern so you can follow every match attempt.
   * - ``LOG_PATTERN_OPTIMIZE=<level>``
     - Integer verbosity level (e.g. ``10``).  Overrides the ``verbose``
       argument passed at construction time for **all** patterns.
   * - ``DROPPATTERN=<names>``
     - Comma-separated list of pattern class names to **skip entirely**.
       Remove patterns one by one to isolate which pattern corrupts the
       model.
   * - ``DUMPPATTERNS=<folder>``
     - Dumps every matched subgraph and its replacement as a
       :epkg:`FunctionProto` file inside *folder*.  Useful for offline
       inspection of what each pattern is doing.
   * - ``PATTERNSTEP=1``
     - Enables step-by-step mode: the optimizer pauses after each applied
       pattern so you can inspect intermediate states.
   * - ``PATTERNNOREMOVE=<name>``
     - Raises an exception if the optimizer removes result *name* during
       any optimization step.

Quick examples
==============

Find where variable ``result_0`` first gets a shape:

.. code-block:: bash

    ONNXSTOPSHAPE=result_0 python convert_my_model.py

Show all constant requests and disable pattern optimization for
``MatMulAddPattern`` to check whether it introduces a numerical error:

.. code-block:: bash

    ONNXCST=1 DROPPATTERN=MatMulAddPattern python convert_my_model.py

Dump every matched pattern to ``/tmp/patterns`` for offline inspection:

.. code-block:: bash

    DUMPPATTERNS=/tmp/patterns python convert_my_model.py

Run with maximum pattern verbosity to see every match attempt:

.. code-block:: bash

    LOG_PATTERN_OPTIMIZE=10 python convert_my_model.py

.. seealso::

    :ref:`l-design-sklearn-converter` — overview of the conversion pipeline.

    :ref:`l-design-expected-api` — :class:`GraphBuilder <yobx.xbuilder.GraphBuilder>`
    API reference.
