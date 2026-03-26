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

Other Variables
===============

Environment variables used by the graph builder, shape-inference engine,
pattern optimizer, and PyTorch exporter are documented in the dedicated
sections below:

* :ref:`l-graphbuilder-debugging-env` — ``GraphBuilder`` and shape-inference
  environment variables (``ONNXSTOP*``, ``ONNXCST``, ``NULLSHAPE``,
  ``ONNXSHAPECOMPUTE``, ``PRINTNAME``, …).
* :ref:`l-design-xshape-debugging` — shape-inference debugging variables.
* :ref:`l-design-pattern-optimizer-debugging` — pattern-optimizer variables
  (``LOG_PATTERN_OPTIMIZE``, ``PATTERN``, ``DROPPATTERN``, …).
* :ref:`l-design-torch-converter` — PyTorch exporter variables
  (``PRINT_EXPORTED_PROGRAM``, ``PRINT_GRAPH_MODULE``, ``ONNXVERBOSE``,
  ``ATENDEBUG``, …).
