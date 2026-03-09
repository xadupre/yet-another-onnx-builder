.. _l-design-torch-supported-aten-functions:

===========================
Supported Aten Functions
===========================

The following functions have a registered converter in
:mod:`yobx.torch.interpreter`.  The list is generated programmatically
from the live converter registry.

Aten Functions
==============

Functions registered via :mod:`yobx.torch.interpreter._aten_functions`.

.. runpython::
    :showcode:
    :rst:

    import yobx.torch.interpreter._aten_functions as _af

    rows = []
    for k, v in sorted(_af.__dict__.items()):
        if not k.startswith("aten_") or not callable(v):
            continue
        rows.append(
            f"* ``{k}`` → "
            f":func:`{v.__name__} <{v.__module__}.{v.__name__}>`"
        )

    print("\n".join(rows))

Attention Functions
===================

Functions registered via
:mod:`yobx.torch.interpreter._aten_functions_attention`.

.. runpython::
    :showcode:
    :rst:

    import yobx.torch.interpreter._aten_functions_attention as _afa

    rows = []
    for k, v in sorted(_afa.__dict__.items()):
        if not k.startswith("aten_") or not callable(v):
            continue
        rows.append(
            f"* ``{k}`` → "
            f":func:`{v.__name__} <{v.__module__}.{v.__name__}>`"
        )

    print("\n".join(rows))

Non-Aten Functions
==================

Functions registered via :mod:`yobx.torch.interpreter._non_aten_functions`.
These converters handle ONNX-specific symbolic operations that are not
part of the standard :epkg:`aten` namespace.

.. runpython::
    :showcode:
    :rst:

    import inspect
    import yobx.torch.interpreter._non_aten_functions as _naf

    rows = []
    for k, v in sorted(_naf.__dict__.items()):
        if not callable(v) or not inspect.isfunction(v):
            continue
        if v.__module__ != _naf.__name__:
            continue
        rows.append(
            f"* ``{k}`` → "
            f":func:`{v.__name__} <{v.__module__}.{v.__name__}>`"
        )

    print("\n".join(rows))

Prims Functions
===============

Functions registered via :mod:`yobx.torch.interpreter._prims_functions`.

.. runpython::
    :showcode:
    :rst:

    import yobx.torch.interpreter._prims_functions as _pf

    rows = []
    for k, v in sorted(_pf.__dict__.items()):
        if not k.startswith("prims_") or not callable(v):
            continue
        rows.append(
            f"* ``{k}`` → "
            f":func:`{v.__name__} <{v.__module__}.{v.__name__}>`"
        )

    print("\n".join(rows))

Math Functions
==============

Functions registered via :mod:`yobx.torch.interpreter._math_functions`.

.. runpython::
    :showcode:
    :rst:

    import yobx.torch.interpreter._math_functions as _mf

    rows = []
    for k, v in sorted(_mf.__dict__.items()):
        if not k.startswith("math_") or not callable(v):
            continue
        rows.append(
            f"* ``{k}`` → "
            f":func:`{v.__name__} <{v.__module__}.{v.__name__}>`"
        )

    print("\n".join(rows))

Aten Methods
============

Tensor-method converters registered via
:mod:`yobx.torch.interpreter._aten_methods`.

.. runpython::
    :showcode:
    :rst:

    import yobx.torch.interpreter._aten_methods as _am

    rows = []
    for k, v in sorted(_am.__dict__.items()):
        if not k.startswith("aten_meth_") or not callable(v):
            continue
        rows.append(
            f"* ``{k}`` → "
            f":func:`{v.__name__} <{v.__module__}.{v.__name__}>`"
        )

    print("\n".join(rows))
