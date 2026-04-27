yobx.torch.fake_tensor_helper
==============================

.. warning::

    :class:`~yobx.torch.fake_tensor_helper.FakeTensorContext` holds the C++
    ``ShapeEnv`` and ``FakeTensorMode`` objects that back every
    :class:`~torch.SymInt` embedded in fake tensor shapes.  Letting the
    context be garbage-collected while fake tensors are still in use causes
    the ``SymInt`` objects to reference freed memory, which **can produce a
    segmentation fault** on some PyTorch builds.

    Always keep the returned context alive for the full lifetime of the fake
    tensors:

    .. code-block:: python

        # ✓  Correct — keep ``ctx`` alive alongside the fake inputs.
        fake_inputs, ctx = make_fake(real_inputs)
        torch.export.export(model, (), kwargs=fake_inputs)
        del ctx  # safe to release only after export is complete

    .. code-block:: python

        # ✗  Dangerous — discarding the context with ``_`` immediately may
        #    free the underlying C++ objects before the fake tensors are used.
        fake_inputs, _ = make_fake(real_inputs)
        torch.export.export(model, (), kwargs=fake_inputs)  # may segfault!

    See :ref:`l-design-fake-tensor-helper` for a full discussion.

.. automodule:: yobx.torch.fake_tensor_helper
    :members:
    :no-undoc-members:
