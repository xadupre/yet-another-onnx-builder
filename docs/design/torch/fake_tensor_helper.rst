.. _l-design-fake-tensor-helper:

================
FakeTensorHelper
================

.. note::
    This section covers functionality that is **specific to PyTorch**.
    It is only relevant when exporting :class:`torch.nn.Module` models with
    :func:`torch.export.export` and has no bearing on ONNX models built
    directly with the builder APIs.

:func:`torch.export.export` traces a model symbolically; it does not need real
tensor *data*, only tensor *metadata* (dtype, shape, device).
:class:`~torch._subclasses.fake_tensor.FakeTensor` objects carry exactly that
metadata without allocating any storage.  Replacing real input tensors with
fake counterparts is therefore the recommended approach when exporting very
large models (e.g. LLMs) where loading weights just to trace the graph would be
prohibitively expensive.

:mod:`yobx.torch.fake_tensor_helper` provides two thin convenience wrappers on
top of PyTorch's :class:`~torch._subclasses.fake_tensor.FakeTensorMode`:

* :func:`~yobx.torch.fake_tensor_helper.make_fake` — replaces every real
  tensor in an arbitrary Python structure with a fake tensor that has the
  same dtype, shape, and device metadata but holds no data.

* :func:`~yobx.torch.fake_tensor_helper.make_fake_with_dynamic_dimensions` —
  like ``make_fake`` but also accepts a ``dynamic_shapes`` specification, so
  that selected tensor dimensions become *symbolic* (represented by
  :class:`~torch.SymInt` objects) rather than concrete integers.

Both functions return the modified structure **and** a
:class:`~yobx.torch.fake_tensor_helper.FakeTensorContext` that must be kept
alive for as long as the fake tensors are in use.

Why a wrapper is needed
=======================

PyTorch's :class:`~torch._subclasses.fake_tensor.FakeTensorMode` is a powerful
but low-level primitive.  Using it directly requires:

1. Creating a :class:`~torch.fx.experimental.symbolic_shapes.ShapeEnv` and a
   :class:`~torch._subclasses.fake_tensor.FakeTensorMode` and keeping them
   alive as long as any fake tensor exists.
2. Ensuring that **dimension sharing** is consistent: if two tensors both have a
   ``batch`` dimension of size 2, the symbolic integer assigned to ``batch``
   must be the *same* :class:`~torch.SymInt` object in both.
3. Handling composite input types (``list``, ``dict``, transformers caches …)
   by recursively replacing every leaf tensor.

:class:`~yobx.torch.fake_tensor_helper.FakeTensorContext` handles all three
concerns.  The ``make_fake*`` module-level helpers are thin wrappers that
create a fresh :class:`~yobx.torch.fake_tensor_helper.FakeTensorContext`
when none is supplied.

Basic usage — ``make_fake``
============================

.. code-block:: python

    import torch
    from yobx.torch.fake_tensor_helper import make_fake

    real_inputs = {
        "input_ids": torch.randint(30360, size=(2, 3), dtype=torch.int64),
        "attention_mask": torch.ones((2, 33), dtype=torch.int64),
    }

    fake_inputs, ctx = make_fake(real_inputs)

    # Every tensor has been replaced — no real data is stored.
    assert isinstance(fake_inputs["input_ids"], torch.Tensor)
    # Shapes are symbolic SymInt objects (not plain Python ints).
    print(type(fake_inputs["input_ids"].shape[0]))  # <class 'torch.SymInt'>

Advanced usage — ``make_fake_with_dynamic_dimensions``
======================================================

Pass a ``dynamic_shapes`` mapping to control which dimensions become symbolic
and how shared dimensions are matched:

.. code-block:: python

    import torch
    from yobx.torch.fake_tensor_helper import make_fake_with_dynamic_dimensions

    real_tensor = torch.rand((4, 8, 32, 64), dtype=torch.float32)

    # Mark dimension 0 as "batch" and dimension 2 as "seq_length".
    fake_t, ctx = make_fake_with_dynamic_dimensions(
        real_tensor,
        {0: "batch", 2: "seq_length"},
    )

    # The symbolic dimensions share names across multiple tensors.
    kv = torch.rand((4, 8, 16, 64), dtype=torch.float32)
    fake_kv, _ = make_fake_with_dynamic_dimensions(
        kv,
        {0: "batch", 2: "cache_length"},
        context=ctx,  # reuse the same context → same SymInt for "batch"
    )

    # fake_t.shape[0] and fake_kv.shape[0] are the *same* symbolic object.
    assert fake_t.shape[0] is fake_kv.shape[0]

Using the context
=================

:class:`~yobx.torch.fake_tensor_helper.FakeTensorContext` is a plain Python
object.  It must be kept alive for as long as the fake tensors produced by it
are in scope.  Letting it be garbage-collected while fake tensors still exist
can lead to hard-to-debug errors or — in some PyTorch builds — to a
**segmentation fault** because the underlying C++ ``ShapeEnv`` and
``FakeTensorMode`` objects are destroyed while references to them still exist
inside the :class:`~torch.SymInt` objects embedded in the fake tensor shapes.

.. warning::

    Always store the returned context alongside the fake inputs:

    .. code-block:: python

        # ✓  Correct — keep `ctx` alive alongside `fake_inputs`.
        fake_inputs, ctx = make_fake(real_inputs)
        exported = torch.export.export(model, (), kwargs=fake_inputs)

    .. code-block:: python

        # ✗  Dangerous — discarding `ctx` immediately may free the
        #    underlying ShapeEnv / FakeTensorMode before `fake_inputs`
        #    are used, potentially causing a segmentation fault.
        fake_inputs, _ = make_fake(real_inputs)
        exported = torch.export.export(model, (), kwargs=fake_inputs)  # may crash!

Segmentation-fault scenario
============================

The following minimal example reproduces the crash.  It creates fake tensors,
immediately discards the context that keeps the underlying C++ objects alive,
forces a garbage-collection cycle, and then tries to use the fake tensors:

.. code-block:: python

    import gc
    import torch
    from yobx.torch.fake_tensor_helper import make_fake

    real = {"x": torch.rand(2, 3)}

    # Step 1 — create fake tensors but discard the context at once.
    fake_inputs, ctx = make_fake(real)
    del ctx          # the FakeTensorMode and ShapeEnv may now be freed
    gc.collect()     # encourage the garbage collector to reclaim them

    # Step 2 — attempt to use a fake tensor with a symbolic shape.
    # On some PyTorch builds this dereferences the freed ShapeEnv and
    # causes a segmentation fault.
    try:
        shape = fake_inputs["x"].shape[0]   # SymInt that references freed memory
        _ = shape + 1                        # arithmetic on a dangling SymInt
    except Exception as exc:
        print(f"Raised {type(exc).__name__}: {exc}")

The correct pattern is to hold the context for the full lifetime of the fake
tensors:

.. code-block:: python

    import torch
    from yobx.torch.fake_tensor_helper import make_fake

    real = {"x": torch.rand(2, 3)}
    fake_inputs, ctx = make_fake(real)

    # Keep `ctx` alive until after `torch.export.export` completes.
    with torch.no_grad():
        exported = torch.export.export(
            model, (), kwargs=fake_inputs, dynamic_shapes=...
        )
    # Only now is it safe to let `ctx` go out of scope.
    del ctx

Common pitfalls
===============

**Dimension aliasing across tensors with the same concrete size.**
When two tensors share the same concrete integer value for what are actually
*different* semantic dimensions (e.g. ``batch=2`` and ``head_dim=2``),
:class:`~yobx.torch.fake_tensor_helper.FakeTensorContext` maps them to the
same :class:`~torch.SymInt`.  Use
:func:`~yobx.torch.fake_tensor_helper.make_fake_with_dynamic_dimensions`
with explicit dimension names to prevent this aliasing:

.. code-block:: python

    import torch
    from yobx.torch.fake_tensor_helper import make_fake_with_dynamic_dimensions

    t1 = torch.rand(2, 2)
    fake_t1, ctx = make_fake_with_dynamic_dimensions(
        t1, {0: "batch", 1: "head_dim"}
    )
    # batch and head_dim are *distinct* SymInts even though both equal 2.
    assert fake_t1.shape[0] is not fake_t1.shape[1]

**Using fake tensors after exporting.**
:class:`~torch._subclasses.fake_tensor.FakeTensor` objects are intended for
tracing only.  Do not attempt to read their data (e.g. via
:meth:`~torch.Tensor.item`, :meth:`~torch.Tensor.numpy`, or
:meth:`~torch.Tensor.tolist`) — these operations raise
:class:`~torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode`
or :class:`~torch._subclasses.fake_tensor.UnsupportedFakeTensorException`
depending on the PyTorch version, and can trigger a segmentation fault in
some builds when the underlying symbolic shapes machinery asserts an
invariant in native C++ code.

**Mixing FakeTensorMode contexts.**
A :class:`~torch._subclasses.fake_tensor.FakeTensor` created by one
:class:`~yobx.torch.fake_tensor_helper.FakeTensorContext` must not be mixed
with another context.  Operations that combine fake tensors from different
:class:`~torch._subclasses.fake_tensor.FakeTensorMode` instances raise
:class:`~torch._subclasses.fake_tensor.UnsupportedFakeTensorException` or
silently produce incorrect shapes.  Always pass an explicit ``context``
argument to reuse a single context across all inputs:

.. code-block:: python

    import torch
    from yobx.torch.fake_tensor_helper import make_fake_with_dynamic_dimensions

    ctx = None
    fake_x, ctx = make_fake_with_dynamic_dimensions(
        torch.rand(2, 8), {0: "batch"}, context=ctx
    )
    fake_y, ctx = make_fake_with_dynamic_dimensions(
        torch.rand(2, 4), {0: "batch"}, context=ctx  # reuse ctx!
    )

.. seealso::

    :ref:`l-design-flatten` — registering pytree nodes for
    ``DynamicCache`` and other transformers classes, which is typically
    needed alongside :func:`~yobx.torch.fake_tensor_helper.make_fake`
    when working with LLMs.

    :ref:`l-design-input-observer` — automating the construction of
    ``args``, ``kwargs``, and ``dynamic_shapes`` for
    :func:`torch.export.export` from real forward passes, which is an
    alternative to constructing fake tensors by hand.

    :mod:`yobx.torch.fake_tensor_helper` — API reference for
    :class:`~yobx.torch.fake_tensor_helper.FakeTensorContext`,
    :func:`~yobx.torch.fake_tensor_helper.make_fake`, and
    :func:`~yobx.torch.fake_tensor_helper.make_fake_with_dynamic_dimensions`.
