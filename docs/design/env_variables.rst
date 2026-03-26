.. _l-design-env-variables:

=====================================
Environment Variables for Unit Tests
=====================================

Several environment variables can be set to control the behavior of unit
tests and to enable extra diagnostic output when debugging the library.
They are grouped below by the component they affect.

Test-Runner Variables (``yobx/ext_test_case.py``)
==================================================

These variables are read by the helpers in :mod:`yobx.ext_test_case` and
control which tests run and how their output is handled.

.. list-table::
   :widths: 25 10 65
   :header-rows: 1

   * - Variable
     - Default
     - Effect
   * - ``UNITTEST_GOING``
     - ``0``
     - Set to ``1`` to signal that the test suite is actively running.
       Several tests use :func:`~yobx.ext_test_case.unit_test_going` to
       shorten expensive computations and skip slow branches.
   * - ``UNHIDE``
     - *(unset)*
     - Set to ``1`` (or ``True``) to disable the
       :func:`~yobx.ext_test_case.hide_stdout` decorator so that standard
       output is **not** suppressed.  Useful for inspecting print statements
       inside tests that are normally silenced.
   * - ``LONGTEST``
     - ``0``
     - Set to ``1`` to enable tests decorated with
       :func:`~yobx.ext_test_case.long_test`.  By default those tests are
       skipped.
   * - ``NEVERTEST``
     - ``0``
     - Set to ``1`` to enable tests decorated with
       :func:`~yobx.ext_test_case.never_test`.  By default those tests are
       always skipped.
   * - ``VERBOSE``
     - ``0``
     - Integer verbosity level exposed via the
       :attr:`~yobx.ext_test_case.ExtTestCase.verbose` property.  Tests may
       print additional diagnostic information when this value is greater than
       zero.
   * - ``DEBUG``
     - *(unset)*
     - Set to ``1`` (or ``True``) to enable extra debug output in tests that
       check the :attr:`~yobx.ext_test_case.ExtTestCase.debug` property.
   * - ``NOTORCH``
     - ``0``
     - Set to ``1`` to pretend that :epkg:`torch` is not installed.  Tests
       decorated with :func:`~yobx.ext_test_case.requires_torch` are
       skipped, and the builder switches to the pure-Python fallback path.

GraphBuilder Variables (``yobx/xbuilder/``, ``yobx/xshape/``)
==============================================================

These variables activate diagnostic features inside
:class:`~yobx.xbuilder.GraphBuilder` and the shape-inference engine.
They are read once at construction time so they take effect for each new
builder instance.

.. list-table::
   :widths: 30 10 60
   :header-rows: 1

   * - Variable
     - Default
     - Effect
   * - ``ONNXSTOP``
     - *(unset)*
     - Raise an exception when a result whose name contains the given string
       is registered.  Example: ``ONNXSTOP=attn_output python ...``.
   * - ``ONNXSTOPSHAPE``
     - *(unset)*
     - Raise an exception when the *shape* of a result whose name matches the
       given string is set.
   * - ``ONNXSTOPTYPE``
     - *(unset)*
     - Raise an exception when the *type* of a result whose name matches the
       given string is set.
   * - ``ONNXSTOPSEQUENCE``
     - *(unset)*
     - Raise an exception when a sequence result whose name matches the given
       string is registered.
   * - ``ONNXSTOPVALUESHAPE``
     - *(unset)*
     - Raise an exception when the value-shape of a result whose name matches
       the given string is set.
   * - ``ONNXSTOPOUTPUT``
     - *(unset)*
     - Raise an exception when a node output whose name matches the given
       string is produced.
   * - ``ONNXNODETYPE``
     - *(unset)*
     - Filter verbose output to only show nodes of the given op-type.
   * - ``ONNXCST``
     - ``0``
     - Set to ``1`` to print each constant that is computed or looked up
       during graph construction.
   * - ``ONNXFOLDNOT``
     - ``0``
     - Set to ``1`` to enable verbose output for ``not``-folding decisions.
   * - ``ONNXFUNC``
     - ``0``
     - Set to ``1`` to enable verbose output for local-function registration.
   * - ``ONNXQUIET``
     - ``0``
     - Set to ``1`` to suppress most informational messages from the builder.
   * - ``ONNXSHAPECOMPUTE``
     - ``0``
     - Set to ``1`` to print a warning every time a shape cannot be inferred.
   * - ``ONNXCONSTANTFOLD``
     - ``0``
     - Set to ``1`` to print verbose output for constant-folding decisions.
   * - ``ONNXDYNDIM``
     - *(unset)*
     - Comma-separated list of dimension names that should be kept dynamic
       even when a static value is known.
   * - ``PRINTNAME``
     - *(unset)*
     - Comma-separated list of result names: the builder prints a diagnostic
       line each time one of these names is added to the graph.
   * - ``NULLSHAPE``
     - ``0``
     - Set to ``1`` to raise an exception as soon as a result is assigned a
       null/empty shape.
   * - ``CONTINUE``
     - ``0``
     - Set to ``1`` to allow the builder to continue past certain internal
       assertion failures instead of raising immediately.
   * - ``DUMPMSG``
     - *(unset)*
     - Set to a file-path prefix; the builder writes the internal graph
       module and graph text to ``<prefix>.module.txt`` and
       ``<prefix>.graph.txt`` when ``process.graph_module`` is available in
       the debug context.
   * - ``ONNX_BUILDER_PROGRESS``
     - ``0``
     - Set to ``1`` to print a progress line after each node is added to the
       graph.
   * - ``NOTF``
     - ``0``
     - Set to ``1`` to disable :epkg:`tensorflow` inside the shape-inference
       runtime even if it is installed.

Pattern-Optimizer Variables (``yobx/xoptim/``)
===============================================

These variables are read by the graph-optimization engine
(:class:`~yobx.xoptim.GraphBuilderOptim` and the pattern API).

.. list-table::
   :widths: 30 10 60
   :header-rows: 1

   * - Variable
     - Default
     - Effect
   * - ``LOG_PATTERN_OPTIMIZE``
     - ``0``
     - Integer verbosity level for the pattern optimizer.  Set to ``1`` or
       higher to print which patterns are matched and applied.
   * - ``PATTERN``
     - *(unset)*
     - Comma-separated list of pattern class names to restrict optimization
       to.  All other patterns are ignored.
   * - ``AMBIGUITIES``
     - ``0``
     - Set to ``1`` to report cases where more than one pattern matches the
       same set of nodes (ambiguity detection).
   * - ``DROPPATTERN``
     - *(unset)*
     - Comma-separated list of pattern class names that should be skipped
       entirely during optimization.
   * - ``PATTERNSTEP``
     - ``0``
     - Set to ``1`` to pause and log after each individual pattern-match
       step.
   * - ``PATTERNNOREMOVE``
     - *(unset)*
     - Name of a result that the optimizer must not remove, even when it
       decides the result is otherwise dead.
   * - ``DUMPPATTERNS``
     - *(unset)*
     - Set to a file path to dump the list of applied patterns to that file
       after each optimization run.

PyTorch Exporter Variables (``yobx/torch/``)
=============================================

These variables control diagnostic output from the PyTorch-to-ONNX export
pipeline.

.. list-table::
   :widths: 30 10 60
   :header-rows: 1

   * - Variable
     - Default
     - Effect
   * - ``PRINT_EXPORTED_PROGRAM``
     - ``0``
     - Set to ``1`` to print the :class:`torch.export.ExportedProgram` to
       stdout before conversion begins.
   * - ``PRINT_GRAPH_MODULE``
     - ``0``
     - Set to ``1`` to print the :class:`torch.fx.GraphModule` to stdout
       during export.
   * - ``ONNXVERBOSE``
     - *(unset)*
     - Integer verbosity override for the ONNX export step; the effective
       verbosity is ``max(requested, ONNXVERBOSE)``.
   * - ``ATENDEBUG``
     - ``0``
     - Integer level.  Set to ``1`` or higher to enable verbose output from
       the aten-function interpreter.
   * - ``NOHTTP``
     - *(unset)*
     - Set to ``1`` to block all HTTP requests inside the Transformers
       configuration helper
       (:func:`~yobx.torch.in_transformers.models.configs.get_cached_configuration`).
       Useful for offline CI runs.

Miscellaneous Variables
=======================

.. list-table::
   :widths: 30 10 60
   :header-rows: 1

   * - Variable
     - Default
     - Effect
   * - ``DUPLICATE``
     - *(unset)*
     - Set to a non-empty string to log duplicate rows detected by the
       :class:`~yobx.helpers.cube_helper.CubeLogs` helper.
   * - ``FIGSIZEH``
     - ``1``
     - Float ratio applied to the default figure height when
       :class:`~yobx.helpers.cube_helper.CubeLogs` renders plots.
