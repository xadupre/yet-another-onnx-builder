.. _l-design-tensorflow-supported-ops:

====================
Supported TF Ops
====================

The following TF op types have a built-in converter in
:mod:`yobx.tensorflow.ops`.  The list is generated programmatically
from the live converter registry.

.. runpython::
    :showcode:
    :rst:

    from yobx.tensorflow import register_tensorflow_converters
    register_tensorflow_converters()
    from yobx.tensorflow.register import get_tf_op_converters

    converters = get_tf_op_converters()

    # Group op-type strings by converter function (module → list of op types)
    groups = {}
    for op_type, fct in sorted(converters.items()):
        groups.setdefault(fct, []).append(op_type)

    # Sort groups by the first op-type name within each group
    for fct, op_types in sorted(groups.items(), key=lambda kv: sorted(kv[1])[0]):
        op_str = ", ".join(f"``{t}``" for t in sorted(op_types))
        print(
            f"* {op_str} → "
            f":func:`{fct.__name__} <{fct.__module__}.{fct.__name__}>`"
        )
    print()
