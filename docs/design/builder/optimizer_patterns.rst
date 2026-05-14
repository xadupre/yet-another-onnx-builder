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

This works on CUDA with :epkg:`yet-another-onnxruntime-extensions`.

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
(see :ref:`l-design-pattern-optimizer-debugging` for the full documentation).
