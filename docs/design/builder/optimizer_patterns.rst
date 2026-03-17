.. _l-design-pattern-optimizer-patterns:

==================
Available Patterns
==================

Default Patterns
================

.. runpython::
    :showcode:
    :rst:

    from yobx.xoptim import get_pattern_list

    names = sorted([(pat.__class__.__name__, pat.__module__)
                    for pat in get_pattern_list("default")])
    for i, (name, module) in enumerate(names):
        print(f"* {i+1}: :class:`{name} <{module}.{name}>`")

Patterns specific to onnxruntime
================================

.. runpython::
    :showcode:
    :rst:

    from yobx.xoptim import get_pattern_list

    names = sorted([(pat.__class__.__name__, pat.__module__)
                    for pat in get_pattern_list("onnxruntime")])
    for i, (name, module) in enumerate(names):
        print(f"* {i+1}: :class:`{name} <{module}.{name}>`")

Patterns specific to ai.onnx.ml
===============================

.. runpython::
    :showcode:
    :rst:

    from yobx.xoptim import get_pattern_list

    names = sorted([(pat.__class__.__name__, pat.__module__)
                    for pat in get_pattern_list("ml")])
    for i, (name, module) in enumerate(names):
        print(f"* {i+1}: :class:`{name} <{module}.{name}>`")

Experimental Patterns
=====================

This works on CUDA with :epkg:`onnx-extended`.

.. runpython::
    :showcode:
    :rst:

    from yobx.xoptim import get_pattern_list

    names = sorted([(pat.__class__.__name__, pat.__module__)
                    for pat in get_pattern_list("experimental")])
    for i, (name, module) in enumerate(names):
        print(f"* {i+1}: :class:`{name} <{module}.{name}>`")

.. _l-debug-patterns:

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