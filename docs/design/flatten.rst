.. _l-design-flatten:

============================
Flattening Functionalities
============================

:func:`torch.export.export` and the :epkg:`torch` pytree machinery require every
Python object that appears as a model input or output to be **registered** as a
pytree node.  When a class is not registered, exporting fails with a cryptic
error.

:mod:`yobx.torch.flatten_helper` provides utilities to register, unregister,
and compose such registrations cleanly.

Why flattening matters
======================

:func:`torch.export.export` traces a PyTorch model into a portable
:class:`torch.fx.GraphModule`.  During tracing every input and every output
must be decomposable into a flat list of :class:`torch.Tensor` objects.  The
decomposition is handled by ``torch.utils._pytree``, which knows about
built-in Python containers (``list``, ``tuple``, ``dict``) but not about
arbitrary user-defined classes.

If a model returns — or receives as input — a class like
``transformers.DynamicCache``, exporting will fail unless that class has been
registered as a pytree node with:

* a **flatten** function — extracts the tensors and a serialisable *context*
  object that describes the structure,
* an **unflatten** function — recreates the original object from the flat list
  and the context,
* a **flatten-with-keys** function — same as flatten but pairs each tensor with
  a :class:`torch.utils._pytree.KeyEntry` that names it.

Core helpers
============

register_class_flattening
--------------------------

:func:`yobx.torch.flatten_helper.register_class_flattening` is a thin wrapper
around ``torch.utils._pytree.register_pytree_node`` that:

* skips the registration silently when the class is already registered (avoids
  duplicate-registration errors),
* optionally runs a user-supplied *check* callable to verify the round-trip
  immediately after registration.

.. runpython::
    :showcode:

    import dataclasses
    from typing import Any, List, Tuple
    import torch
    from yobx.torch.flatten_helper import (
        register_class_flattening,
        unregister_class_flattening,
    )

    # A minimal dict-like container.
    class MyOutput(dict):
        """Simple dict subclass that can hold named tensors."""

    def flatten_my_output(obj):
        keys = list(obj.keys())
        return [obj[k] for k in keys], keys

    def flatten_with_keys_my_output(obj):
        keys = list(obj.keys())
        values = [obj[k] for k in keys]
        return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(keys, values)], keys

    def unflatten_my_output(values, context, output_type=None):
        return MyOutput(zip(context, values))

    register_class_flattening(
        MyOutput,
        flatten_my_output,
        unflatten_my_output,
        flatten_with_keys_my_output,
    )

    # Flatten and unflatten a MyOutput object.
    obj = MyOutput(a=torch.tensor([1.0, 2.0]), b=torch.tensor([3.0]))
    flat, spec = torch.utils._pytree.tree_flatten(obj)
    print("flat tensors:", [t.tolist() for t in flat])

    restored = torch.utils._pytree.tree_unflatten(flat, spec)
    print("restored keys:", list(restored.keys()))

    # Clean up so that subsequent runs of the docs are not affected.
    unregister_class_flattening(MyOutput)

make_flattening_function_for_dataclass
--------------------------------------

:func:`yobx.torch.flatten_helper.make_flattening_function_for_dataclass`
auto-generates the three required callables for any class that exposes
``.keys()`` / ``.values()`` like a mapping (e.g. ``transformers.ModelOutput``
subclasses).

.. runpython::
    :showcode:

    import torch
    from yobx.torch.flatten_helper import (
        make_flattening_function_for_dataclass,
        register_class_flattening,
        unregister_class_flattening,
    )

    class HiddenState(dict):
        """Dict-like container for a transformer hidden state."""

    flatten_fn, flatten_with_keys_fn, unflatten_fn = (
        make_flattening_function_for_dataclass(HiddenState, set())
    )

    print("generated function names:")
    print(" ", flatten_fn.__name__)
    print(" ", flatten_with_keys_fn.__name__)
    print(" ", unflatten_fn.__name__)

    register_class_flattening(
        HiddenState, flatten_fn, unflatten_fn, flatten_with_keys_fn
    )

    obj = HiddenState(last_hidden_state=torch.zeros(2, 3))
    flat, spec = torch.utils._pytree.tree_flatten(obj)
    print("flat:", [t.shape for t in flat])

    unregister_class_flattening(HiddenState)

register_cache_flattening and the context manager
--------------------------------------------------

:func:`yobx.torch.flatten_helper.register_cache_flattening` registers a
collection of classes in one call and returns a ``dict`` that can be passed to
:func:`yobx.torch.flatten_helper.unregister_cache_flattening` to undo every
registration.

:func:`yobx.torch.flatten_helper.register_flattening_functions` wraps both in
a :func:`contextlib.contextmanager` so that registrations are automatically
undone when the ``with`` block exits:

.. code-block:: python

    from yobx.torch import register_flattening_functions

    with register_flattening_functions(patch_transformers=True) as fix_inputs:
        # Inside this block:
        # * DynamicCache, StaticCache, EncoderDecoderCache, and BaseModelOutput
        #   are registered as pytree nodes.
        # * fix_inputs is a callable that replaces non-standard containers in
        #   a nested input structure with their plain Python equivalents before
        #   handing them to torch.export.export.
        inputs = fix_inputs(my_inputs)
        exported = torch.export.export(model, (inputs,))

After the ``with`` block all registrations are rolled back, leaving
``torch.utils._pytree.SUPPORTED_NODES`` exactly as it was before.

Transformers-specific registrations
=====================================

When ``patch_transformers=True`` is passed to
:func:`~yobx.torch.flatten_helper.register_cache_flattening` (or
:func:`~yobx.torch.flatten_helper.register_flattening_functions`), the
following classes from :epkg:`transformers` are registered:

+-----------------------------+-------------------------------------------------------------+
| Class                       | Description                                                 |
+=============================+=============================================================+
| ``DynamicCache``            | Key-value cache whose layers grow as new tokens are decoded |
+-----------------------------+-------------------------------------------------------------+
| ``StaticCache``             | Pre-allocated key-value cache with a fixed maximum length   |
+-----------------------------+-------------------------------------------------------------+
| ``EncoderDecoderCache``     | Wraps a self-attention and a cross-attention cache          |
+-----------------------------+-------------------------------------------------------------+
| ``BaseModelOutput``         | Generic output container (dict-like dataclass)              |
+-----------------------------+-------------------------------------------------------------+

The flatten functions are defined in
:mod:`yobx.torch.transformers.flatten_class`.  The module also patches
registrations that are already present but known to be incompatible with
:func:`torch.export.export` (see ``WRONG_REGISTRATIONS``).

``DynamicCache`` layers
------------------------

.. note::
    The layer-type-aware flattening described below relies on the ``layers``
    attribute of :class:`transformers.cache_utils.DynamicCache`, which was
    introduced in **transformers >= 4.50**.  On older versions of
    ``transformers`` the cache is serialized with plain ``key_<i>`` /
    ``value_<i>`` keys and no per-layer type information is preserved.  Use
    :func:`~yobx.torch.transformers.flatten_class.flatten_dynamic_cache` only
    with ``transformers >= 4.50`` if you need to round-trip mixed layer types.

A ``DynamicCache`` can contain layers of different types
(``DynamicLayer``, ``DynamicSlidingWindowLayer``, etc.).  The flatten
context encodes each layer type as a short letter code so that the
correct layer class and its kwargs are recreated on unflatten:

+-------------------------------+------+
| Layer class                   | Code |
+===============================+======+
| ``DynamicLayer``              | D    |
+-------------------------------+------+
| ``DynamicSlidingWindowLayer`` | W    |
+-------------------------------+------+
| ``StaticLayer``               | S    |
+-------------------------------+------+
| ``StaticSlidingWindowLayer``  | X    |
+-------------------------------+------+

.. seealso::

    :ref:`l-plot-flattening` — sphinx-gallery example demonstrating
    registration of a custom class and the round-trip flatten / unflatten.

    :ref:`l-design-helpers` — the :class:`MiniOnnxBuilder
    <yobx.helpers.mini_onnx_builder.MiniOnnxBuilder>` which serialises
    pytree-flattened tensors to ONNX.
