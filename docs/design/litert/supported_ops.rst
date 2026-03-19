.. _l-design-litert-supported-ops:

====================
Supported LiteRT Ops
====================

The following TFLite op types have a built-in converter in
:mod:`yobx.litert.ops`.  The list is generated programmatically
from the live converter registry.

.. runpython::
    :showcode:
    :rst:

    import re
    from yobx.litert import register_litert_converters
    from yobx.litert.register import LITERT_OP_CONVERTERS
    from yobx.litert.litert_helper import builtin_op_name

    register_litert_converters()

    PATTERN = re.compile(r"TFLite\s+``(\w+)``\s+→\s+ONNX\s+(.+)")
    MODULE_LABELS = {
        "yobx.litert.ops.activations": "Activations",
        "yobx.litert.ops.elementwise": "Element-wise",
        "yobx.litert.ops.nn_ops": "Neural network",
        "yobx.litert.ops.reshape_ops": "Shape / tensor manipulation",
    }
    MODULE_ORDER = list(MODULE_LABELS.keys())

    groups = {m: [] for m in MODULE_ORDER}
    for code, fn in LITERT_OP_CONVERTERS.items():
        mod = fn.__module__
        doc = (fn.__doc__ or "").strip().splitlines()[0].strip().rstrip(".")
        m = PATTERN.match(doc)
        tflite_op = m.group(1) if m else (builtin_op_name(code) if isinstance(code, int) else code)
        onnx_op = m.group(2).rstrip(".") if m else "?"
        if mod in groups:
            groups[mod].append((tflite_op, onnx_op))

    for mod in MODULE_ORDER:
        label = MODULE_LABELS[mod]
        items = sorted(groups[mod])
        if not items:
            continue
        print(f"**{label}**")
        print()
        for tflite_op, onnx_op in items:
            print(f"* ``{tflite_op}`` → {onnx_op}")
        print()
